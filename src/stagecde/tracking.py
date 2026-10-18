"""Physical particle-step refinement, separated from steady RANS iterations."""
import argparse,hashlib,json,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from assembly import ROOT
from stageb.rain import simulate,load_case

OUT=ROOT/'results/research/C/rain'
def one(level,dt,n):
    tag=f'reference-{level}-n{n}-dt{dt:g}-bundle6-union.json';old=ROOT/'results/stage-b/rain'/tag
    if (OUT/tag).exists():
        r=json.loads((OUT/tag).read_text());_,_,_,meta,_=load_case(ROOT/'results/stage-b/cfd'/('reference-'+level))
        if meta['U_sha256']!=r['CFD_U_sha256']:raise ValueError('Cached particle case no longer matches the CFD velocity field')
        return r
    if old.exists():
        r=json.loads(old.read_text())
        if r['seed']==7 and r['source_scheme']=='union_of_upstream_and_top_boundary_bundles_v2':
            r['reused_source']=str(old.relative_to(ROOT));r['reused_source_sha256']=hashlib.sha256(old.read_bytes()).hexdigest();(OUT/tag).write_text(json.dumps(r,indent=2));return r
    start=time.time();r=simulate(ROOT/'results/stage-b/cfd'/('reference-'+level),n,5.,7,dt,6.,.5,False,OUT)
    r['runtime_s']=time.time()-start;(OUT/tag).write_text(json.dumps(r,indent=2));return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--grids',nargs='+',default=['coarse','medium','fine']);p.add_argument('--dts',nargs='+',type=float,default=[.1,.05,.025]);p.add_argument('--samples',type=int,default=8192);p.add_argument('--workers',type=int,default=2);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(one,g,dt,a.samples) for g in a.grids for dt in a.dts]
        for f in as_completed(fs):rows.append(f.result())
    comparison=[]
    for g in a.grids:
        selected=sorted([r for r in rows if 'reference-'+g in r['case']],key=lambda r:-r['dt_s'])
        for x,y in zip(selected,selected[1:]):
            delta=abs(x['total_specific_catch_ratio']-y['total_specific_catch_ratio']);scale=max(x['total_specific_catch_ratio'],y['total_specific_catch_ratio'])
            comparison.append({'grid':g,'dt_from':x['dt_s'],'dt_to':y['dt_s'],'absolute_eta_difference':float(delta),'relative_difference':float(delta/scale),'two_percent_difference_pass':bool(delta/scale<.02),
              'paired_sampling_uncertainty_quantified':False,'sampling_uncertainty_note':'same random source draws reduce variance of differences; paired covariance is not estimated'})
    grids=[]
    order=['coarse','medium','fine']
    for dt in a.dts:
        z=sorted([r for r in rows if r['dt_s']==dt],key=lambda r:order.index(r['case'].rsplit('reference-',1)[1]))
        for x,y in zip(z,z[1:]):
            delta=abs(x['total_specific_catch_ratio']-y['total_specific_catch_ratio']);scale=max(x['total_specific_catch_ratio'],y['total_specific_catch_ratio'])
            grids.append({'dt':dt,'from':x['case'],'to':y['case'],'absolute_eta_difference':float(delta),'relative_difference':float(delta/scale),'two_percent_difference_pass':bool(delta/scale<.02)})
    result={'runs':rows,'time_comparisons':comparison,'grid_comparisons':grids,'SIMPLE_deltaT_is_physical_time':False,'grid_independence_established':False,
      'particle_capture_accuracy_qualified':False,'why_not':'finite source bundle, sampling error and CFD grid dependence remain separate gates'}
    (ROOT/'results/research/C'/('tracking-'+('-'.join(a.grids))+'.json')).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
