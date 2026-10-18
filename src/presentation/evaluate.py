"""Frozen-policy 2025 holdout, exact teacher replay, transparent domain censoring."""
import argparse,copy,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT
from weather import cities,model_config,load_weather
from fetch_weather import download
from prepare_weather import prepare
from stageb.thermal import SurfaceScene
from physics import PanelModel
from stagecde.plant import FastPlant
from stagecde.surrogate import Emulator
from stagecde.control import rule,mpc
from environment import observation
from actuator import load_envelope,transition,pose
from flow import air_properties
from presentation.accounting import AuditObserver

OUT=ROOT/'results/presentation';CACHE=ROOT/'data/presentation/episodes'
def protocol():return json.loads((ROOT/'config/presentation/protocol.json').read_text())

def fetch():
    p=protocol();citymap={c['id']:c for c in cities()};records=[];OUT.mkdir(parents=True,exist_ok=True)
    def one(cid):
        info=download(citymap[cid],p['test_year']);_,qc=prepare(citymap[cid],p['test_year']);path=ROOT/f'data/normalized/{cid}-{p["test_year"]}.csv.gz'
        return {'city':cid,'year':p['test_year'],'status':qc['status'],'hours':qc['hours'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':info}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,c) for c in p['cities']]):r=f.result();records.append(r);print(r['city'],r['status'],flush=True)
    (OUT/'weather.json').write_text(json.dumps({'datasets':records,'reanalysis_not_experimental_truth':True,'issued_forecasts_used':False},indent=2))
    if any(r['status']!='passed' for r in records):raise SystemExit(1)

def make_episode(cid,index):
    p=protocol();c=model_config();f=load_weather(cid,p['test_year'],c);h=p['episode_hours'];warm=p['surface_warmup_hours'];q=p['weather_preselection']
    valid=(f.air_C>=q['air_min_C'])&(f.wind_m_s<q['wind_max_m_s'])&(f.snowfall_cm_h==0)&(f.rh_pct<=100)
    bad=np.r_[0,np.cumsum(~valid.to_numpy())];starts=np.arange(warm,len(f)-h-1);starts=starts[(bad[starts+h+1]-bad[starts-warm])==0]
    if not len(starts):raise ValueError('No supported warm window '+cid)
    desired=int(p['seasonal_targets_fraction_of_year'][index]*len(f));start=int(starts[np.argmin(abs(starts-desired))]);part=f.iloc[start-warm:start+h+1].copy()
    scene=SurfaceScene(initial_C=float(part.iloc[0].air_C));walls=[];grounds=[]
    for _,w in part.iterrows():
        sw=w.dni_W_m2*max(0,w.sun_z)*(w.sun_y>0)+.5*w.dhi_W_m2+.1*w.ghi_W_m2
        for _ in range(60):b=scene.step(w.air_C,sw,w.ghi_W_m2,w.lw_down_W_m2,w.wind_m_s,60)
        walls.append(b['wall_C']);grounds.append(b['ground_C'])
    part['wall_C']=walls;part['ground_C']=grounds;frame=part.iloc[warm:].copy();tag=f'test-{cid}-{p["test_year"]}-e{index}'
    CACHE.mkdir(parents=True,exist_ok=True);frame.to_csv(CACHE/(tag+'.csv.gz'),index_label='time',compression='gzip')
    meta={'city':cid,'year':p['test_year'],'episode':index,'group':tag,'start':str(frame.index[0]),'end':str(frame.index[-1]),'hours':h,
      'intended_annual_fraction':p['seasonal_targets_fraction_of_year'][index],'start_offset_from_target_hours':start-desired,
      'annual_weather_hours':len(f),'annual_weather_outside_preselection_hours':int((~valid).sum()),
      'annual_weather_outside_preselection_percent':100*float((~valid).mean()),'annual_thermodynamic_device_scope_evaluated':False,
      'future_weather_used_for_window_domain_selection_only':True,'site_elevation_m':f.attrs['site_elevation_m'],'lat':f.attrs['city']['lat']}
    (CACHE/(tag+'.json')).write_text(json.dumps(meta,indent=2));return meta

def episodes(workers):
    p=protocol();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(make_episode,c,e) for c in p['cities'] for e in range(p['episodes_per_city'])]):r=f.result();records.append(r);print(r['group'],r['start'],flush=True)
    (OUT/'episodes.json').write_text(json.dumps({'episodes':records,'annual_yield_estimate':False,'selection':'predeclared seasonal targets, closest warm/snow-free window'},indent=2))

def obs(frame,t,state,c,meta,model):
    w=frame.iloc[t];x=observation(w,state,c,w.name,meta['site_elevation_m'],meta['lat'],model.valid_actions(w))
    return np.r_[x,w.wall_C/60,w.ground_C/60].astype(np.float32)

def decide(kind,model,planner,w,policy,observation_value):
    c=model.c;s=model.state
    if kind=='fixed':raw=2
    elif kind=='emitter':raw=7
    elif kind=='rule':raw=rule(w,s,c)
    elif kind=='mpc':planner.state=copy.deepcopy(s);raw=mpc(planner,w)
    else:raw=int(policy.predict(observation_value,deterministic=True)[0])
    mask=model.valid_actions(w);move=transition(c,s.angle_deg,pose(raw//5,c['tilts_deg'][raw%5]));rho,_,_=air_properties(float(w.air_C),float(w.pressure_Pa));wind=load_envelope(c,model.assembly,float(w.wind_m_s),rho,film_mass=float(s.film_kg.sum()))
    interlocked=not bool(mask[raw]);a=s.face*5+s.tilt_index if interlocked else raw
    power=c['motor_running_upper_W'] if c['motor_energy_policy']=='upper_running' else c['motor_running_W']
    return a,{'policy_requested_action':int(raw),'causal_action_interlock':int(interlocked),
      'causal_wind_interlock':int(interlocked and not wind['screen_pass']),
      'causal_energy_interlock':int(interlocked and move['duration_s']*power/3600>s.battery_Wh*c['discharge_efficiency']),
      'requested_move':int(move['duration_s']>0)}

def replay_one(group,kind,scenario=None):
    import torch
    torch.set_num_threads(1);start=time.time();c=model_config();meta=json.loads((CACHE/(group+'.json')).read_text());f=pd.read_csv(CACHE/(group+'.csv.gz'),index_col=0,parse_dates=True)
    if scenario:
        key,value=scenario
        if key in c:c[key]=value
        elif key=='rain_factor':f.rain_mm_h*=value
        elif key=='LW_offset':f.lw_down_W_m2+=value
    trace=AuditObserver(c);model=PanelModel(c,trace);model.reset(float(f.iloc[0].air_C));policy=None;planner=None
    if kind=='ppo':
        from stable_baselines3 import PPO
        policy=PPO.load(ROOT/protocol()['frozen_policy'],device='cpu')
    if kind=='mpc':planner=FastPlant(Emulator(),c)
    rows=[];valid_prefix=True;prefix_count=0
    for t in range(len(f)-1):
        w=f.iloc[t];a,decision=decide(kind,model,planner,w,policy,obs(f,t,model.state,c,meta,model))
        try:r=model.step_hour(f.iloc[t+1],a)
        except (ValueError,FloatingPointError) as exc:
            rows.append({'group':group,'city':meta['city'],'controller':kind,'hour':t+1,'time':str(f.index[t+1]),'solver_failed':True,'error':str(exc),'water_supported':0,'valid_prefix':False});break
        valid_prefix=valid_prefix and bool(r['water_supported']);prefix_count+=int(valid_prefix)
        rows.append({'group':group,'city':meta['city'],'controller':kind,'hour':t+1,'time':str(f.index[t+1]),'valid_prefix':valid_prefix,'requested_action':a,**decision,**r})
    summary={'group':group,'controller':kind,'hours':len(rows),'valid_prefix_hours':prefix_count,'seconds':time.time()-start,'scenario':scenario,
      'complete':len(rows)==len(f)-1,'unsupported_hours':sum(not r['water_supported'] for r in rows)}
    return rows,summary

def run(workers):
    if (OUT/'evaluation.json').exists():raise FileExistsError('Preserve F1 holdout; use a new output version for reruns')
    p=protocol();selection=json.loads((ROOT/'results/research/E/control/selection.json').read_text());policy_path=ROOT/p['frozen_policy']
    if policy_path.name!=selection['chosen_policy']:raise ValueError('Protocol must use previously selected frozen policy')
    rows=[];summaries=[];groups=sorted(x.stem for x in CACHE.glob('*.json'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(replay_one,g,k) for g in groups for k in p['controllers']]
        for f in as_completed(fs):r,s=f.result();rows.extend(r);summaries.append(s);print(s['group'],s['controller'],s['unsupported_hours'],flush=True)
    frame=pd.DataFrame(rows).sort_values(['group','controller','hour']);frame.to_csv(OUT/'hourly.csv.gz',index=False,compression='gzip')
    result={'episodes':summaries,'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'frozen_selection_sha256':hashlib.sha256((ROOT/'results/research/E/control/selection.json').read_bytes()).hexdigest(),
      'protocol_sha256':hashlib.sha256((ROOT/'config/presentation/protocol.json').read_bytes()).hexdigest(),
      'new_weather_holdout_year':p['test_year'],'retrained_on_holdout':False,'independent_physical_validation':False,'RL_improvement_claim_allowed':False}
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2))

SCENARIOS=[('nominal',None),('h_low',('external_h_multiplier',.7)),('h_high',('external_h_multiplier',1.3)),
 ('coating_degraded',('coating_condition',.5)),('collector_low',('collector_efficiency',.7)),('collector_high',('collector_efficiency',1.)),
 ('bridge_low',('bridge_multiplier',.5)),('bridge_high',('bridge_multiplier',2.)),('motor_low',('motor_running_upper_W',60)),('motor_high',('motor_running_upper_W',84)),
 ('rain_low',('rain_factor',.75)),('rain_high',('rain_factor',1.25)),('LW_low',('LW_offset',-15)),('LW_high',('LW_offset',15))]

def sensitivity_one(city,name,scenario):
    g=f'test-{city}-{protocol()["test_year"]}-e0';rows,meta=replay_one(g,'rule',scenario);f=pd.DataFrame(rows)
    return {'city':city,'scenario':name,'hours':len(f),'unsupported_hours':meta['unsupported_hours'],
      **{k:float(f[k].sum()) for k in ['collected_rain_L','collected_condensation_L','evaporated_L','uncollected_departure_L','pv_gross_Wh','pv_useful_in_system_Wh','motor_Wh','idle_demand_Wh']}}

def sensitivity(workers):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(sensitivity_one,c,n,s) for c in ['singapore','rasht','dubai','phoenix'] for n,s in SCENARIOS]
        for f in as_completed(fs):r=f.result();rows.append(r);print(r['city'],r['scenario'],flush=True)
    (OUT/'scenarios.json').write_text(json.dumps({'runs':rows,'range_type':'one-at-a-time epistemic/forcing scenarios, not a probabilistic CI','controller':'frozen threshold rule','independent_parameter_probability_distribution':False},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['fetch','episodes','run','sensitivity']);p.add_argument('--workers',type=int,default=4);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.task=='fetch':fetch()
    elif a.task=='episodes':episodes(a.workers)
    elif a.task=='run':run(a.workers)
    else:sensitivity(a.workers)
if __name__=='__main__':main()
