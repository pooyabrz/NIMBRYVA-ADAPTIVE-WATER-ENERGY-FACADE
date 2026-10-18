"""Common-information controller comparison. No realized future as a forecast."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from assembly import ROOT
from stagecde.plant import ResearchEnv,FastPlant,reward
from stagecde.surrogate import Emulator

OUT=ROOT/'results/research/E/control'

def rule(w,state,c):
    if w.rain_mm_h>.2:return 5
    if w.ghi_W_m2>100:return int(np.argmax([w[f'poa_{i}'] for i in range(5)]))
    if w.rh_pct>65:return 7
    return state.face*5+state.tilt_index

def mpc(plant,w,horizon=3,beam=3):
    # Persistence only: this function receives ONE current row, never a frame.
    root=copy.deepcopy(plant.state);c=plant.c;states=[(0.,root,None)]
    try:
        for depth in range(horizon):
            candidates=[]
            for total,s,first in states:
                plant.state=copy.deepcopy(s)
                for a in np.flatnonzero(plant.valid_actions(w)):
                    plant.state=copy.deepcopy(s);r=plant.step_hour(w,int(a))
                    score=total+reward(r,c)
                    if depth==horizon-1:score-=c['reward_final_SOC_deficit_per_Wh']*max(0,root.battery_Wh-plant.state.battery_Wh)
                    candidates.append((score,copy.deepcopy(plant.state),int(a) if first is None else first))
            states=sorted(candidates,key=lambda x:x[0],reverse=True)[:beam]
        return states[0][2]
    finally:plant.state=root

def action(kind,env,policy=None):
    w=env.frame.iloc[env.t];s=env.plant.state
    if kind=='fixed':a=2
    elif kind=='rule':a=rule(w,s,env.c)
    elif kind=='mpc':a=mpc(env.plant,w)
    elif kind=='ppo':a=int(policy.predict(env._obs(),deterministic=True)[0])
    else:raise ValueError(kind)
    if not env.plant.valid_actions(w)[a]:a=s.face*5+s.tilt_index
    return int(a)

def rollout(kind,role,policy=None,groups=None,teacher=False):
    env=ResearchEnv(role,False);rows=[];summaries=[]
    for tag in groups or sorted(env.frames):
        env.reset(seed=200,options={'group':tag})
        if teacher:
            from physics import PanelModel
            real=PanelModel(env.c);real.reset(float(env.frame.iloc[0].air_C));planner=env.plant;env.plant=real
        episode=[];total=0.
        for i in range(len(env.frame)-1):
            # MPC is always the same frozen emulator planner, initialized with the observed state.
            if teacher and kind=='mpc':
                planner.state=copy.deepcopy(env.plant.state);a=mpc(planner,env.frame.iloc[env.t])
            else:a=action(kind,env,policy)
            if teacher:
                env.t+=1
                try:r=env.plant.step_hour(env.frame.iloc[env.t],a)
                except (FloatingPointError,ValueError) as exc:
                    episode.append({'group':tag,'controller':kind,'time':str(env.frame.index[env.t]),'solver_failed':True,'error':str(exc),'water_supported':0});break
                value=reward(r,env.c)
                if env.t==len(env.frame)-1:value-=env.c['reward_final_SOC_deficit_per_Wh']*max(0,100-r['battery_Wh'])
            else:_,value,_,_,r=env.step(a)
            rec={'group':tag,'controller':kind,'time':str(env.frame.index[env.t]),'requested_action':a,'reward':value,**r};episode.append(rec);total+=value
        rows.extend(episode);f=pd.DataFrame(episode)
        sums={k:float(f[k].sum()) if k in f else None for k in ['drain_L','pv_gross_Wh','motor_Wh','unserved_Wh','rain_input_L','condensed_L']}
        summaries.append({'group':tag,'controller':kind,'reward':float(total),'hours':len(f),'unsupported_hours':int((f.water_supported==0).sum()),
          'final_battery_Wh':float(env.plant.state.battery_Wh),'complete':len(f)==len(env.frame)-1,
          'mass_residual_max_kg':float(f.get('mass_residual_kg',pd.Series([np.nan])).abs().max()),**sums})
    return rows,summaries

def train(steps,seeds):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise FileExistsError('Selection already frozen; use a new protocol for new candidates')
    records=[]
    for seed in seeds:
        start=time.time();env=ResearchEnv('train',True)
        policy=PPO('MlpPolicy',env,seed=seed,device='cpu',n_steps=512,batch_size=256,n_epochs=5,learning_rate=.0003,
          gamma=.99,gae_lambda=.95,clip_range=.2,target_kl=.03,policy_kwargs={'net_arch':[64,64]},verbose=0)
        policy.learn(total_timesteps=steps);dest=OUT/f'ppo-{seed}';policy.save(dest)
        rows,summ=rollout('ppo','selection',policy);pd.DataFrame(rows).to_csv(OUT/f'selection-ppo-{seed}.csv.gz',index=False,compression='gzip')
        rec={'seed':seed,'actual_timesteps':policy.num_timesteps,'seconds':time.time()-start,'selection_mean_return':float(np.mean([r['reward'] for r in summ])),
          'selection_unsupported_hours':sum(r['unsupported_hours'] for r in summ),'episodes':summ,'policy':dest.name+'.zip'}
        records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='episodes'}),flush=True)
        (OUT/'training-progress.json').write_text(json.dumps(records,indent=2))
    chosen=max(records,key=lambda x:x['selection_mean_return']);baseline=[]
    for kind in ['fixed','rule','mpc']:
        rows,summ=rollout(kind,'selection');pd.DataFrame(rows).to_csv(OUT/f'selection-{kind}.csv.gz',index=False,compression='gzip');baseline.extend(summ)
    selection={'candidate_runs':records,'chosen_seed':chosen['seed'],'chosen_policy':chosen['policy'],'baselines':baseline,
      'selection_rule':'highest mean 2022 episode reward; all seeds retained','test_used_for_selection':False,
      'forecast':'row-t persistence; no t+1 weather in any controller','MPC_horizon_h':3,'MPC_beam':3,
      'randomization':'per training episode parameters + emulator member; scenarios, not calibrated uncertainty',
      'initial_eval_SOC_Wh':100,'same_area_geometry_weather_initial_states':True,'product_qualified':False,
      'model_sha256':hashlib.sha256((ROOT/'results/research/E/surrogate/model.json').read_bytes()).hexdigest()}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))

def evaluate():
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    if (OUT/'test.json').exists():raise FileExistsError('Test already recorded; no adaptive retuning on held-out results')
    s=json.loads((OUT/'selection.json').read_text());policy=PPO.load(OUT/s['chosen_policy'],device='cpu');summary=[];allrows=[]
    for kind in ['fixed','rule','mpc','ppo']:
        rows,summ=rollout(kind,'test',policy);allrows.extend(rows);summary.extend(summ);print(kind,'test episodes',len(summ),flush=True)
    f=pd.DataFrame(allrows);f.to_csv(OUT/'test-hourly.csv.gz',index=False,compression='gzip')
    # Common support intersection prevents a controller gaining credit by censoring bad hours alone.
    support=f.pivot_table(index=['group','time'],columns='controller',values='water_supported',aggfunc='min').fillna(0).min(axis=1)>0
    common=set(support[support].index);eligible=f[[tuple(x) in common for x in f[['group','time']].itertuples(index=False,name=None)]]
    by={kind:{'common_supported_hours':len(z),'drain_L':float(z.drain_L.sum()),'pv_gross_Wh':float(z.pv_gross_Wh.sum()),'motor_Wh':float(z.motor_Wh.sum())} for kind,z in eligible.groupby('controller')}
    result={'policy':s['chosen_policy'],'policy_sha256':hashlib.sha256((OUT/s['chosen_policy']).read_bytes()).hexdigest(),
      'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),'episodes':summary,'common_support_totals':by,
      'all_hours_diagnostic_only':True,'annual_climate_yield':False,'independent_experiment':False,'CFD_teacher':False,
      'product_or_RL_superiority_claim_qualified':False,'test_city_year_groups':sorted(f.group.unique().tolist()),
      'inference_scope':'conditional warm episode emulator experiment; unsupported hours retained, never imputed as supported zero yield'}
    (OUT/'test.json').write_text(json.dumps(result,indent=2))

def replay_job(kind,groups):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1);s=json.loads((OUT/'selection.json').read_text());policy=PPO.load(OUT/s['chosen_policy'],device='cpu') if kind=='ppo' else None
    return kind,rollout(kind,'test',policy,groups,True)

def replay(workers=4):
    if (OUT/'teacher-replay.json').exists():raise FileExistsError('Preserve previous independent held-out solver replay')
    groups=[f'test-{city}-2023-e0' for city in ['singapore','bandar-abbas','rasht','dubai']];summ=[];rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(replay_job,k,groups) for k in ['fixed','rule','mpc','ppo']]
        for f in as_completed(fs):
            kind,(r,s)=f.result();rows.extend(r);summ.extend(s);print('teacher replay',kind,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'teacher-replay-hourly.csv.gz',index=False,compression='gzip')
    (OUT/'teacher-replay.json').write_text(json.dumps({'episodes':summ,'groups':groups,'independent_experiment':False,
      'purpose':'out-of-training multi-step controller replay in conditional numerical teacher; ranking can disagree with emulator',
      'no_tuning_after_replay':True,'product_qualified':False},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['train','test','replay']);p.add_argument('--steps',type=int,default=50000);p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27]);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.task=='train':train(a.steps,a.seeds)
    elif a.task=='test':evaluate()
    else:replay(a.workers)
if __name__=='__main__':main()
