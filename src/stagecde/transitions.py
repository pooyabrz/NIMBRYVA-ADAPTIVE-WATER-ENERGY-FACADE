"""Teacher transitions from the conditional v2 solver, never labelled CFD truth."""
import argparse,copy,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT
from weather import model_config,load_weather
from physics import PanelModel
from stageb.thermal import SurfaceScene
from stagecde.datasets import jobs

OUT=ROOT/'results/research/E';CACHE=ROOT/'data/research/episodes'
PARAMS=['external_h_multiplier','coating_condition','collector_efficiency','bridge_multiplier','motor_running_upper_W']
WEATHER=['air_C','rh_pct','pressure_Pa','wind_m_s','wind_x','wind_z','rain_mm_h','dni_W_m2','dhi_W_m2','ghi_W_m2','lw_down_W_m2','sun_x','sun_y','sun_z','wall_C','ground_C']
TARGETS=['T0_offset','T1_offset','T2_offset','T3_offset','T4_offset','T5_offset','T6_offset','T7_offset',
 'rain_log','cond_log','evap_log','drain_log','loss_log','film0_log','film1_log','film2_log','pv_Wh','cavity_C_offset','cavity_w','supported']
RANGES={'external_h_multiplier':[.7,1.3],'coating_condition':[.5,1.],'collector_efficiency':[.7,1.],
        'bridge_multiplier':[.5,2.],'motor_running_upper_W':[60.,84.]}

def features(state,w,action,c):
    return np.r_[state.temperature_C,state.film_kg,state.battery_Wh,np.sin(np.radians(state.angle_deg)),np.cos(np.radians(state.angle_deg)),
      state.cavity_temperature_C,state.cavity_mixing_ratio,state.wet_hours,state.uv_dose_MJ_m2,
      [float(w[k]) for k in WEATHER],np.eye(10)[int(action)],[c[k] for k in PARAMS]].astype(float)

def targets(state,result,ta):
    water=np.array([result[k] for k in ['rain_input_L','condensed_L','evaporated_L','drain_L','uncollected_departure_L']]+list(state.film_kg))
    return np.r_[state.temperature_C-ta,np.log1p(np.maximum(water,0)/.01),result['pv_gross_Wh'],
                 state.cavity_temperature_C-ta,state.cavity_mixing_ratio,result['water_supported']]

def episode(city,year,index,role,hours,seed):
    rng=np.random.default_rng(seed);c=model_config()
    if role=='train':c.update({k:float(rng.uniform(*RANGES[k])) for k in PARAMS})
    f=load_weather(city,year,c);valid=(f.air_C>=10)&(f.snowfall_cm_h==0)&(f.rh_pct<=100)&(f.wind_m_s<10)
    warmup=12;bad=np.r_[0,np.cumsum(~valid.to_numpy())];starts=np.arange(warmup,len(f)-hours-1)
    starts=starts[(bad[starts+hours+1]-bad[starts-warmup])==0]
    if not len(starts):return {'status':'NO_SUPPORTED_WINDOW','city':city,'year':year,'role':role}
    # Deterministic seasonal coverage. Selection is a conditional domain, not annual climatology.
    desired=int((index+.5)/4*len(f));start=int(starts[np.argmin(abs(starts-desired))])
    part=f.iloc[start-warmup:start+hours+1].copy();scene=SurfaceScene(initial_C=float(part.iloc[0].air_C))
    walls=[];grounds=[]
    for _,w in part.iterrows():
        sun=max(0,w.sun_z) if w.sun_y>0 else 0;sw=w.dni_W_m2*sun+.5*w.dhi_W_m2+.1*w.ghi_W_m2
        for _ in range(60):b=scene.step(w.air_C,sw,w.ghi_W_m2,w.lw_down_W_m2,w.wind_m_s,60)
        walls.append(b['wall_C']);grounds.append(b['ground_C'])
    part['wall_C']=walls;part['ground_C']=grounds
    model=PanelModel(c);model.reset(float(part.iloc[0].air_C));model.state.battery_Wh=float(rng.uniform(80,160)) if role=='train' else 100.
    warmup_resets=[]
    for i in range(1,warmup+1):
        try:model.step_hour(part.iloc[i],0)
        except FloatingPointError as exc:
            warmup_resets.append({'time':str(part.index[i]),'error':str(exc)})
            model.reset(float(part.iloc[i].air_C))
    saved=part.iloc[warmup:].copy();tag=f'{role}-{city}-{year}-e{index}'
    CACHE.mkdir(parents=True,exist_ok=True);saved.to_csv(CACHE/(tag+'.csv.gz'),index_label='time',compression='gzip')
    # Controller evaluation uses a prescribed equal initial state, independent of random teacher rollouts.
    meta={'city':city,'year':year,'role':role,'episode':index,'seed':seed,'hours':hours,'start':str(saved.index[0]),
      'config':{k:c[k] for k in PARAMS},'warmup_resets':warmup_resets,'conditional_warm_window':True,'selection_uses_future_for_domain_only':True,
      'forecast_contains_future_reanalysis':False,'initial_eval_battery_Wh':100.,'case_group':tag}
    (CACHE/(tag+'.json')).write_text(json.dumps(meta,indent=2))
    xs=[];ys=[];rowmeta=[];failures=[]
    for i in range(warmup+1,len(part)):
        old=copy.deepcopy(model.state);current=part.iloc[i-1];w=part.iloc[i]
        mask=model.valid_actions(current);validactions=np.flatnonzero(mask)
        actions=rng.choice(validactions,size=min(3,len(validactions)),replace=False)
        next_for_rollout=None
        for a in actions:
            model.state=copy.deepcopy(old)
            try:r=model.step_hour(w,int(a))
            except (FloatingPointError,ValueError) as exc:
                failures.append({'time':str(w.name),'action':int(a),'error':str(exc)});continue
            resolved=int(r['face'])*len(c['tilts_deg'])+int(np.argmin(abs(np.array(c['tilts_deg'])-r['tilt_deg'])))
            xs.append(features(old,w,resolved,c));ys.append(targets(model.state,r,float(w.air_C)))
            rowmeta.append({'group':tag,'city':city,'year':year,'role':role,'time':str(w.name),'action':int(a),'resolved_action':resolved,'supported':bool(r['water_supported'])})
            if next_for_rollout is None:next_for_rollout=copy.deepcopy(model.state)
        model.state=next_for_rollout if next_for_rollout is not None else old
        if next_for_rollout is None:model.reset(float(w.air_C))
    dest=OUT/'transitions';dest.mkdir(parents=True,exist_ok=True)
    if xs:np.savez_compressed(dest/(tag+'.npz'),X=np.array(xs),Y=np.array(ys))
    (dest/(tag+'.json')).write_text(json.dumps({'episode':meta,'rows':rowmeta,'failures':failures,'teacher':'cassette-v2.0 + B1 hourly wall/ground boundary; no CFD closure qualification'},indent=2))
    return {'status':'COMPLETED','group':tag,'role':role,'samples':len(xs),'unsupported_samples':sum(not x['supported'] for x in rowmeta),'failed_steps':len(failures)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--hours',type=int,default=72)
    p.add_argument('--roles',nargs='+',default=['train','selection','test']);p.add_argument('--episodes-train',type=int,default=4);p.add_argument('--episodes-other',type=int,default=2);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);todo=[]
    if (OUT/'surrogate/model.json').exists():raise FileExistsError('A model is already fitted; preserve its teacher data and create a new research version')
    for j,(city,year,role) in enumerate(jobs()):
        if role in a.roles:
            for e in range(a.episodes_train if role=='train' else a.episodes_other):todo.append((city,year,e,role,a.hours,1937+j*17+e))
    records=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(episode,*j) for j in todo]
        for f in as_completed(fs):
            try:r=f.result()
            except Exception as exc:r={'status':'FAILED','error':repr(exc)}
            records.append(r);print(json.dumps(r),flush=True)
    result={'records':records,'runtime_s':time.time()-start,'parameters':PARAMS,'ranges':RANGES,'targets':TARGETS,
      'teacher_is_independent_experiment':False,'CFD_surrogate':False,'source':'conditional v2 solver emulator for research only'}
    (OUT/('transitions-'+('-'.join(a.roles))+'.json')).write_text(json.dumps(result,indent=2))
    if any(r['status']=='FAILED' for r in records):raise SystemExit(1)
if __name__=='__main__':main()
