"""Frozen-model diagnostic intervals and identical-action held-out rollouts."""
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from assembly import ROOT
from physics import PanelModel
from weather import model_config
from stagecde.transitions import CACHE
from stagecde.surrogate import Emulator,load_split,physical
from stagecde.plant import conserve_water

OUT=ROOT/'results/research/E'
def openloop_job(group,commands):
    f=pd.read_csv(CACHE/(group+'.csv.gz'),index_col=0,parse_dates=True);rows=[]
    for kind in ['fixed','rule','mpc','ppo']:
        model=PanelModel(model_config());model.reset(float(f.iloc[0].air_C));seq=commands[commands.controller==kind].reset_index(drop=True)
        for i in range(len(f)-1):
            try:r=model.step_hour(f.iloc[i+1],int(seq.iloc[i].requested_action))
            except (ValueError,FloatingPointError) as exc:
                rows.append({'group':group,'controller':kind,'time':str(f.index[i+1]),'solver_failed':True,'error':str(exc)});break
            rows.append({'group':group,'controller':kind,'time':str(f.index[i+1]),**r})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--open-loop',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    net=Emulator();x,y,files=load_split('test');pred=net.predict(x);raw=physical(pred,x[:,18]);truth=physical(y,x[:,18]);projected=raw['water'].copy()
    for i in range(len(x)):projected[i]=conserve_water(x[i,8:11],projected[i])
    de=projected[:,3]-truth['water'][:,3];dt=raw['T']-truth['T'];pe=raw['pv']-truth['pv']
    result={'model_sha256':hashlib.sha256((net.directory/'model.json').read_bytes()).hexdigest(),'test_used_to_retune':False,
      'after_mass_projection':{'drain_MAE_L':float(np.mean(abs(de))),'drain_RMSE_L':float(np.sqrt(np.mean(de**2))),
      'drain_abs_error_p95_L':float(np.quantile(abs(de),.95)),'reference_mean_drain_L':float(truth['water'][:,3].mean()),
      'nearzero_reference_count':int((truth['water'][:,3]<1e-6).sum()),
      'nearzero_false_positive_above_0_0001L':int(((truth['water'][:,3]<1e-6)&(projected[:,3]>1e-4)).sum()),
      'surface_abs_error_p95_K':np.quantile(abs(dt[:,[0,6]]),.95,axis=0).tolist(),'PV_abs_error_p95_Wh':float(np.quantile(abs(pe),.95))},
      'interval_interpretation':'empirical held-out error quantiles; not physical confidence intervals, not conformal coverage guarantee'}
    f=pd.read_csv(OUT/'control/test-hourly.csv.gz');table=f.pivot_table(index=['group','time'],columns='controller',values='water_supported',aggfunc='min').fillna(0)
    common=set(table.index[table.min(1)>0]);mask=[tuple(r) in common for r in f[['group','time']].itertuples(index=False,name=None)];z=f[mask]
    agg=z.groupby(['group','controller']).drain_L.sum().unstack('controller').fillna(0);agg['city']=[n.removeprefix('test-').rsplit('-2023-',1)[0] for n in agg.index]
    city=agg.groupby('city')[['fixed','rule','mpc','ppo']].mean();rng=np.random.default_rng(512);intervals={}
    for base in ['fixed','rule','mpc']:
        diff=(city.ppo-city[base]).to_numpy();boot=np.mean(rng.choice(diff,(5000,len(diff)),replace=True),1)
        intervals['ppo_minus_'+base]={'mean_city_episode_drain_difference_L':float(diff.mean()),'city_cluster_bootstrap_95_interval_L':np.quantile(boot,[.025,.975]).tolist(),'cities':len(diff)}
    result['conditional_controller_difference_intervals']=intervals;result['bootstrap_scope']='eight cities, single test year, common supported hours; not annual climate confidence or proof of product/RL superiority'
    if a.open_loop:
        dest=OUT/'control/open-loop-teacher.csv.gz'
        if dest.exists():real=pd.read_csv(dest)
        else:
            rows=[];groups=['test-singapore-2023-e0','test-rasht-2023-e0']
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                fs=[pool.submit(openloop_job,g,f[f.group==g]) for g in groups]
                for job in as_completed(fs):rows.extend(job.result())
            real=pd.DataFrame(rows);real.to_csv(dest,index=False,compression='gzip')
        joined=f.merge(real,on=['group','controller','time'],suffixes=('_emulator','_teacher'));scores=[]
        for (g,k),v in joined.groupby(['group','controller']):
            e=v.surface_dew_C_emulator-v.surface_dew_C_teacher;w=v.drain_L_emulator.sum()-v.drain_L_teacher.sum();energy=v.pv_gross_Wh_emulator.sum()-v.pv_gross_Wh_teacher.sum()
            scores.append({'group':g,'controller':k,'hours':len(v),'emitter_temperature_RMSE_K':float(np.sqrt(np.mean(e**2))),
              'drain_cumulative_error_L':float(w),'PV_cumulative_error_Wh':float(energy),'teacher_unsupported_hours':int((v.water_supported_teacher==0).sum())})
        result['identical_command_multistep_replay']=scores
        result['rollout_note']='identical exogenous requested actions and initial states; physical interlocks can resolve commands differently as states diverge'
    result['qualified_for_product_or_controller_superiority_claim']=False
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
