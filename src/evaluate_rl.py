"""Local held-out seasonal comparison; unsupported windows stay excluded."""
import argparse
import json
from pathlib import Path
import pandas as pd
from benchmark import run_case
from control import fixed_pv,causal_rule
from environment import observation
from weather import ROOT,load_weather,model_config
from assembly import MODEL_VERSION


def validate_policy_metadata(path,config):
    import hashlib
    training=Path(path).parent/'training.json'
    if not training.exists():raise ValueError('training.json required beside policy')
    meta=json.loads(training.read_text())
    expected=hashlib.sha256((ROOT/'config/model.json').read_bytes()).hexdigest()
    if meta.get('model_version')!=MODEL_VERSION or meta.get('observation_shape')!=[48]:
        raise ValueError('Legacy policy is incompatible with cassette-v2.0; retrain locally.')
    if meta.get('config_sha256')!=expected:
        raise ValueError('Policy configuration differs; retrain or explicitly design a documented transfer evaluation.')
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',required=True,help='Path to policy.zip from local PPO training')
    p.add_argument('--cities',nargs='+',default=['istanbul','london','la-paz','reykjavik'])
    p.add_argument('--year',type=int,default=2025)
    a=p.parse_args()
    cfg=model_config();meta=validate_policy_metadata(a.policy,cfg)
    from stable_baselines3 import PPO
    learned=PPO.load(a.policy,device='cpu');rows=[];trained=meta['arguments']
    for city in a.cities:
        frame=load_weather(city,a.year,cfg)
        for month in (1,4,7,10):
            start=frame.index.get_loc(pd.Timestamp(f'{a.year}-{month:02d}-15T00:00Z'))
            counter={'t':start}
            def rl_policy(last_weather,physical_model):
                t=counter['t']
                obs=observation(last_weather,physical_model.state,cfg,frame.index[t],frame.attrs['site_elevation_m'],frame.attrs['city']['lat'],physical_model.valid_actions(last_weather))
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
