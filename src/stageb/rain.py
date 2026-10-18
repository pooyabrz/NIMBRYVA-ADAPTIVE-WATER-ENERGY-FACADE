"""B2: multi-size 3D Lagrangian rain on an EXECUTED OpenFOAM field.

Flux-weighted source bundles are sampled on the actual upstream/top boundary.
Drag/gravity, first solid impact, size-wise catch, splash regime and uncertain
retention are separate. The source bundle is not the entire boundary: zero-hit
and bundle-truncation uncertainty are reported, never called validated zero.
"""
import argparse,hashlib,json,math,re
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from assembly import ROOT


def field(path):
    text=Path(path).read_text()
    if 'format binary' in text:raise ValueError('Export ASCII OpenFOAM fields first')
    m=re.search(r'internalField\s+nonuniform\s+List<(vector|scalar)>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not m:raise ValueError('Expected nonuniform cell field: '+str(path))
    values=np.fromstring(m.group(3).replace('(',' ').replace(')',' '),sep=' ')
    n=int(m.group(2));dims=3 if m.group(1)=='vector' else 1
    if values.size!=n*dims:raise ValueError('OpenFOAM field length mismatch')
    return values.reshape(n,dims) if dims==3 else values


def case_signature(meta):
    keys=['release','tested_patch','wind10_m_s','z0_m','wind_direction_deg','panel_angle_deg','empty_domain','bounds_m',
       'panel_vertices_m','building_vertices_m','T_ambient_K','T_wall_K','T_panel_K','Yv_inlet','Yv_panel','Pr','Prt','Sc_molecular','Sct']
    if 'RAS_model' in meta:keys+=['RAS_model']
    return hashlib.sha256(json.dumps({k:meta[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_case(path,require_execution=True):
    path=Path(path);meta=json.loads((path/'case.json').read_text())
    execution_path=path/'execution.json'
    if require_execution:
        if not execution_path.exists():raise ValueError('No execution record; synthetic flow substitution is forbidden')
        execution=json.loads(execution_path.read_text())
        if not execution['status'].startswith('EXECUTED'):raise ValueError('CFD run/mesh was rejected')
        qpath=path/'quality.json'
        if not qpath.exists():raise ValueError('Run stage-b quality first: mesh/residual/physical-geometry gate required')
        q=json.loads(qpath.read_text())
        if not q['flow_available_for_exploratory_particle_tracking']:raise ValueError('CFD quality/geometry gate rejected this flow')
        export=path/'compact-export.json'
        if export.exists() and json.loads(export.read_text()).get('case_definition_sha256')!=case_signature(meta):
            raise ValueError('Case definition changed after CFD execution/export; regenerate a separate case')
    times=sorted([p for p in path.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',p.name) and float(p.name)>0],key=lambda x:float(x.name))
    if not times or not (times[-1]/'C').exists():
        cp=path/'reference-field.npz';provenance=path/'compact-export.json'
        if not cp.exists() or not provenance.exists():raise ValueError('No solved CFD field; rerun cfd or use a compact exported case')
        provenance=json.loads(provenance.read_text())
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=provenance['NPZ_sha256']:raise ValueError('CFD compact export checksum mismatch')
        data=np.load(cp,allow_pickle=False);centres=data['centres'];u=data['U'];extras={k:data[k] for k in data.files if k not in ['centres','U']}
        meta['solved_time']=provenance['source_time'];meta['cells']=len(centres);meta['U_sha256']=provenance['U_sha256'];meta['compact_fields_used']=True
        return centres,u,extras,meta,path/provenance['source_time']
    latest=times[-1];centres=field(latest/'C');u=field(latest/'U')
    extras={name:field(latest/name) for name in ['T','Yv','nut','k','epsilon','omega','p'] if (latest/name).exists()}
    meta['solved_time']=latest.name;meta['cells']=len(centres);meta['U_sha256']=hashlib.sha256((latest/'U').read_bytes()).hexdigest()
    return centres,u,extras,meta,latest


class VelocityField:
    def __init__(self,centres,u):self.centres=centres;self.u=u;self.tree=cKDTree(centres)
    def __call__(self,x):
        d,i=self.tree.query(x,k=4);weights=1/np.maximum(d,1e-9)**2
        return np.sum(self.u[i]*weights[:,:,None],axis=1)/weights.sum(1)[:,None]


def drag_factor(re):
    re=np.asarray(re)
    return np.where(re<1000,1+.15*re**.687,.44*re/24)


def terminal_speed(d,rho=1.2,mu=1.81e-5):
    tau=1000*d*d/(18*mu)
    return brentq(lambda v:v*float(drag_factor(rho*v*d/mu))/tau-9.80665*(1-rho/1000),1e-8,30)


def advance_particles(x,v,air,d,dt):
    """Frozen-drag exponential step; accuracy checked against adaptive ODE."""
    slip=air-v;speed=np.linalg.norm(slip,axis=1);re=1.2*speed*d/1.81e-5
    tau=1000*d*d/(18*1.81e-5*drag_factor(re));g=np.array([0.,0.,-9.80665*(1-1.2/1000)])
    equilibrium=air+tau[:,None]*g;decay=np.exp(-dt/tau)
    nxt=x+equilibrium*dt+(v-equilibrium)*tau[:,None]*(1-decay)[:,None]
    vn=equilibrium+(v-equilibrium)*decay[:,None]
    return nxt,vn


def distribution(rain_mm_h,diameters_mm):
    d=np.asarray(diameters_mm,dtype=float)
    if rain_mm_h<=0 or np.any(d<=0):raise ValueError('Positive rain and diameters required')
    edges=np.r_[max(.05,d[0]-(d[1]-d[0])/2),(d[:-1]+d[1:])/2,d[-1]+(d[-1]-d[-2])/2]
    widths=np.diff(edges);v=np.array([terminal_speed(x/1000) for x in d])
    number=8000*np.exp(-4.1*rain_mm_h**(-.21)*d)
    flux=number*np.pi/6*(d/1000)**3*1000*v*widths
    weights=flux/flux.sum()
    return weights,v,{'model':'Marshall-Palmer shape; flux renormalized to supplied rain rate',
      'unscaled_flux_kg_m2_s':float(flux.sum()),'specified_flux_kg_m2_s':rain_mm_h/3600,
      'diameter_range_mm':d.tolist(),'not_measured_local_DSD':True,'spherical_drag_above_1mm_uncertain':True}


def triangles(vertices):
    q=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    return [vertices[list(t)] for f in q for t in [(f[0],f[1],f[2]),(f[0],f[2],f[3])]]


def intersections(start,end,objects):
    """First segment/triangle intersection; catches thin surfaces crossed in dt."""
    distance=np.full(len(start),np.inf);which=np.full(len(start),-1);normal=np.zeros_like(start)
    delta=end-start
    for obj_id,tri in objects:
        a,b,c=tri;e1=b-a;e2=c-a
        cross=np.cross(delta,e2);det=cross@e1;valid=np.abs(det)>1e-12
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);off=start-a
        bary_u=np.sum(off*cross,axis=1)*inv;q=np.cross(off,e1)
        bary_v=np.sum(delta*q,axis=1)*inv;t=(q@e2)*inv
        hit=valid&(bary_u>=-1e-9)&(bary_v>=-1e-9)&(bary_u+bary_v<=1+1e-9)&(t>=0)&(t<=1)&(t<distance)
        n=np.cross(e1,e2);n/=np.linalg.norm(n)
        distance[hit]=t[hit];which[hit]=obj_id;normal[hit]=n
    return which,distance,normal


def impact(d,velocity,normal,retention_after_splash=.5):
    vn=abs(float(np.dot(velocity,normal)));rho=1000.;mu=.001;sigma=.072
    re=rho*vn*d/mu;we=rho*vn*vn*d/sigma;k=math.sqrt(we)*re**.25
    splashes=k>57.7
    # The criterion alone does NOT identify a deposited fraction on CNF/glass.
    return {'Re':re,'We':we,'Mundo_K':k,'splash_indicator':bool(splashes),
      'retained_fraction_scenario':retention_after_splash if splashes else 1.,
      'retention_empirically_validated_for_this_surface':False}


def source_regions(meta,target,fall,bundle):
    """Keep BOTH inflow planes if the proposal crosses their common corner."""
    lo,hi=np.asarray(meta['bounds_m']);z0=meta['z0_m'];coef=meta['wind10_m_s']/math.log((10+z0)/z0)
    def travel_x(z):
        def primitive(x):return (x+z0)*(math.log((x+z0)/z0)-1)
        return coef*(primitive(z)-primitive(target[2]))/fall
    dx=target[0]-lo[0];upper=hi[2]
    while travel_x(upper)<dx:upper*=2
    upstream_z=brentq(lambda z:travel_x(z)-dx,target[2],upper)
    ylo=max(lo[1],target[1]-bundle);yhi=min(hi[1],target[1]+bundle)
    zlo=max(0,upstream_z-bundle);zhi=min(hi[2],upstream_z+bundle)
    xcentre=target[0]-travel_x(hi[2]);xlo=max(lo[0],xcentre-bundle);xhi=min(hi[0],xcentre+bundle)
    regions=[]
    if zhi>zlo:regions.append(('upstream',zlo,zhi,0))
    if xhi>xlo:regions.append(('top',xlo,xhi,2))
    if not regions:raise ValueError('Seed bundle outside both inflow boundaries; enlarge bundle/domain')
    return regions,ylo,yhi


def simulate(case,samples=1024,rain_mm_h=5.,seed=7,dt=.05,bundle_halfwidth=6.,retention=.5,capture_paths=False,output_dir=None):
    if samples<16 or dt<=0 or bundle_halfwidth<=0 or not 0<=retention<=1:raise ValueError('Invalid particle settings')
    case=Path(case).resolve()
    centres,u,extra,meta,latest=load_case(case);vf=VelocityField(centres,u)
    lo,hi=np.array(meta['bounds_m']);pverts=np.array(meta['panel_vertices_m']);bverts=np.array(meta['building_vertices_m'])
    target=pverts.mean(0);objects=[(0,t) for t in triangles(bverts)]+[(1,t) for t in triangles(pverts)]
    if meta['empty_domain']:raise ValueError('Use uniform manufactured verification for empty geometry')
    rng=np.random.default_rng(seed);diam=np.array([.25,.5,.75,1.,1.5,2.,3.,4.])
    weights,vt,dsd=distribution(rain_mm_h,diam);rows=[];paths=[]
    profile=np.zeros((2,32));edge_retained=0.
    chord=pverts[3]-pverts[0];chord_length=np.linalg.norm(chord);chord/=chord_length
    normal_axis=pverts[4]-pverts[0];half_thickness=np.linalg.norm(normal_axis)/2;normal_axis/=2*half_thickness
    for bi,(dmm,fall,fraction) in enumerate(zip(diam,vt,weights)):
        d=dmm/1000;wind=meta['wind10_m_s']
        # Log-profile integral, rather than a constant U10 ballistic line.
        # This only centres the importance bundle; all trajectories use CFD.
        regions,ylo,yhi=source_regions(meta,target,fall,bundle_halfwidth)
        positions=[];massflows=[];strata=[];source_area=0.;concentration=(rain_mm_h/3600)*fraction/fall
        for ri,(plane,low,high,axis) in enumerate(regions):
            ns=samples//len(regions)+(samples%len(regions) if ri==0 else 0)
            if axis==0:ps=np.c_[np.full(ns,lo[0]+1e-6),rng.uniform(ylo,yhi,ns),rng.uniform(low,high,ns)]
            else:ps=np.c_[rng.uniform(low,high,ns),rng.uniform(ylo,yhi,ns),np.full(ns,hi[2]-1e-6)]
            vv=vf(ps);vv[:,2]-=fall;area=(high-low)*(yhi-ylo);source_area+=area
            inward=np.maximum(vv[:,0],0) if axis==0 else np.maximum(-vv[:,2],0)
            positions.append(ps);massflows.append(concentration*inward*area/ns);strata.extend([ri]*ns)
        pos=np.concatenate(positions);massflow=np.concatenate(massflows);strata=np.array(strata)
        plane='+'.join(r[0] for r in regions);velocity=vf(pos);velocity[:,2]-=fall
        active=np.ones(samples,dtype=bool);hits=np.full(samples,-2);retained=np.zeros(samples);wevalues=[]
        trace_indices=np.arange(min(12,samples));trace=[];elapsed=0;step=0
        tmax=4*max((target[0]-lo[0])/wind,hi[2]/fall)+30
        while active.any() and elapsed<tmax:
            ids=np.flatnonzero(active);x=pos[ids];v=velocity[ids];air=vf(x)
            nxt,vn=advance_particles(x,v,air,d,dt)
            obj,dist,normals=intersections(x,nxt,objects)
            collided=obj>=0
            for local in np.flatnonzero(collided):
                idx=ids[local];hits[idx]=obj[local]
                if obj[local]==1:
                    imp=impact(d,vn[local],normals[local],retention);retained[idx]=imp['retained_fraction_scenario'];wevalues.append(imp['We'])
                    point=x[local]+dist[local]*(nxt[local]-x[local]);offset=float((point-target)@normal_axis)
                    if abs(abs(offset)-half_thickness)<1e-6:
                        side=0 if offset>0 else 1;fractional=np.clip(.5-float((point-target)@chord)/chord_length,0,1-1e-12)
                        profile[side,int(fractional*32)]+=massflow[idx]*retained[idx]
                    else:edge_retained+=massflow[idx]*retained[idx]
                pos[idx]=x[local]+dist[local]*(nxt[local]-x[local]);active[idx]=False
            moving=~collided;idx=ids[moving];pos[idx]=nxt[moving];velocity[idx]=vn[moving]
            outside=np.any((pos[idx]<lo)|(pos[idx]>hi),axis=1);hits[idx[outside]]=-1;active[idx[outside]]=False
            if step%10==0:trace.append(pos.copy() if capture_paths else pos[trace_indices].copy())
            elapsed+=dt;step+=1
        # Unresolved particles are not counted as successful escape/capture.
        unresolved=int(active.sum());hit=hits==1;building=hits==0;escape=hits==-1
        caught=float(massflow[hit].sum());kept=float((massflow*retained).sum())
        denominator=rain_mm_h/3600*fraction*meta['area_m2']
        eta=caught/denominator
        draws=massflow*hit/denominator
        standard_error=math.sqrt(sum(int((strata==i).sum())*float(np.var(draws[strata==i],ddof=1)) for i in range(len(regions))))
        n=int(hit.sum());phat=n/samples;z=1.96;den=1+z*z/samples
        centre=(phat+z*z/(2*samples))/den;half=z*math.sqrt(phat*(1-phat)/samples+z*z/(4*samples*samples))/den
        scaling=float(massflow.sum()/denominator)
        rows.append({'diameter_mm':dmm,'horizontal_rain_flux_fraction':float(fraction),'terminal_m_s':fall,
          'source_plane':plane,'source_area_m2':source_area,'samples':samples,'panel_hits':n,'building_hits':int(building.sum()),
          'escaped':int(escape.sum()),'unresolved':unresolved,'source_massflow_kg_s':float(massflow.sum()),
          'panel_impact_kg_s':caught,'panel_retained_kg_s_scenario':kept,'specific_catch_ratio':eta,
          'weighted_sampling_standard_error_catch_ratio':standard_error,
          'impact_mass_accounting_residual_kg_s':float(massflow.sum()-massflow[hits==1].sum()-massflow[hits==0].sum()-massflow[hits==-1].sum()-massflow[hits==-2].sum()),
          'sampling_only_approx_interval':[max(0,centre-half)*scaling,min(1,centre+half)*scaling],
          'zero_hits_is_unresolved_not_validated_zero':n==0,'maximum_impact_We':max(wevalues,default=None),
          'particle_count_residual':samples-n-int(building.sum())-int(escape.sum())-unresolved})
        chosen=np.r_[np.flatnonzero(hit)[:6],np.flatnonzero(~hit)[:6]] if capture_paths else None
        paths.append(np.array(trace)[:,chosen] if capture_paths else np.array(trace))
        print('rain',Path(case).name,dmm,'mm, hits',n,'unresolved',unresolved,flush=True)
    out=Path(output_dir) if output_dir else ROOT/'results/stage-b/rain';out.mkdir(parents=True,exist_ok=True);tag=Path(case).name+f'-n{samples}-dt{dt:g}-bundle{bundle_halfwidth:g}-union'
    if capture_paths:tag+='-capturepaths'
    total_eta=sum(r['horizontal_rain_flux_fraction']*r['specific_catch_ratio'] for r in rows)
    standard_error=math.sqrt(sum((r['horizontal_rain_flux_fraction']*r['weighted_sampling_standard_error_catch_ratio'])**2 for r in rows))
    result={'case':str(Path(case).relative_to(ROOT)),'CFD_cells':len(centres),'source_time':latest.name,
      'CFD_U_sha256':meta['U_sha256'],'seed':seed,'dt_s':dt,'bundle_halfwidth_m':bundle_halfwidth,
      'source_scheme':'union_of_upstream_and_top_boundary_bundles_v2','sampling_interval_method':'independent stratified weighted Monte Carlo standard error; normal approximation',
      'trajectory_selection':'up to 6 actual panel hits and 6 other trajectories per size' if capture_paths else 'first 12 samples per size',
      'drop_distribution':dsd,'bins':rows,'total_panel_impact_kg_s':sum(r['panel_impact_kg_s'] for r in rows),
      'total_retained_kg_s_scenario':sum(r['panel_retained_kg_s_scenario'] for r in rows),
      'total_specific_catch_ratio':total_eta,'sampling_only_standard_error_catch_ratio':standard_error,
      'sampling_only_approx95_catch_ratio':[max(0,total_eta-1.96*standard_error),total_eta+1.96*standard_error],
      'sampling_relative95_halfwidth':1.96*standard_error/total_eta if total_eta>0 else None,
      'sampling_five_percent_gate':bool(total_eta>0 and 1.96*standard_error/total_eta<.05 and all(r['panel_hits']>=10 for r in rows)),
      'deposited_profile_kg_s_scenario':profile.tolist(),'edge_deposit_kg_s_scenario':float(edge_retained),
      'profile_coordinate':'32 equal strips from high to low elevation; positive/negative thickness faces; no transverse film resolution',
      'profile_mass_residual_kg_s':float(profile.sum()+edge_retained-sum(r['panel_retained_kg_s_scenario'] for r in rows)),
      'limitations':['finite focused boundary bundle: enlarge until catch converges','point-particle spherical drag','mean RANS, no stochastic dispersion',
        'no airborne evaporation/coalescence/breakup','Mundo threshold outside calibration on large drops/CNF is an indicator; retained fraction is a scenario',
        'secondary splashed parcels are a loss ledger, not re-tracked; no entrainment of existing film'],
      'independent_experimental_validation':False,'not_a_final_collection_efficiency':True}
    (out/(tag+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/(tag+'-paths.npz'),**{f'diameter_{i}':x for i,x in enumerate(paths)})
    if not (Path(case)/'reference-field.npz').exists():np.savez_compressed(out/(Path(case).name+'-cfd.npz'),centres=centres,U=u,**extra)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--rain',type=float,default=5);p.add_argument('--seed',type=int,default=7);p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--bundle',type=float,default=6);p.add_argument('--retention',type=float,default=.5);p.add_argument('--capture-paths',action='store_true');a=p.parse_args()
    simulate(a.case,a.samples,a.rain,a.seed,a.dt,a.bundle,a.retention,a.capture_paths)


if __name__=='__main__':main()
