"""Local held-out seasonal comparison; unsupported windows stay excluded."""
import argparse
import json
from pathlib import Path
import pandas as pd
from benchmark import run_case
from control import fixed_pv,causal_rule
from environment import observation
from weather import ROOT,load_weather,model_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',required=True,help='Path to policy.zip from local PPO training')
    p.add_argument('--cities',nargs='+',default=['istanbul','london','la-paz','reykjavik'])
    p.add_argument('--year',type=int,default=2025)
    a=p.parse_args()
    from stable_baselines3 import PPO
    learned=PPO.load(a.policy,device='cpu');cfg=model_config();rows=[]
    training=Path(a.policy).parent/'training.json'
    if not training.exists():raise ValueError('training.json is required beside policy.zip for provenance')
    trained=json.loads(training.read_text())['arguments']
    for city in a.cities:
        frame=load_weather(city,a.year,cfg)
        for month in (1,4,7,10):
            start=frame.index.get_loc(pd.Timestamp(f'{a.year}-{month:02d}-15T00:00Z'))
            counter={'t':start}
            def rl_policy(last_weather,physical_model):
                t=counter['t']
                obs=observation(last_weather,physical_model.state,cfg,frame.index[t],frame.attrs['site_elevation_m'],frame.attrs['city']['lat'])
                action,_=learned.predict(obs,deterministic=True)
                counter['t']+=1
                return int(action)
            for name,controller in [('fixed_pv',fixed_pv),('causal_rule',causal_rule),('ppo',rl_policy)]:
                summary,_=run_case(frame,start,controller,cfg,name)
                summary.update(city=city,month=month,year=a.year,
                               heldout_city=city not in trained['cities'],heldout_year=a.year!=trained['year'])
                rows.append(summary)
    output=Path(a.policy).parent/f'evaluation-{a.year}.csv'
    pd.DataFrame(rows).to_csv(output,index=False)
    print('Saved diagnostic comparison:',output)
    print('This comparison does not validate simulator accuracy or field performance.')


if __name__=='__main__':main()
