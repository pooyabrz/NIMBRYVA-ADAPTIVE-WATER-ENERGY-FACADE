"""Optional local PPO training. A learned policy is not physical validation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from environment import FacadeEnv
from weather import ROOT
from assembly import MODEL_VERSION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cities',nargs='+',default=['rasht','bandar-abbas','singapore','mumbai','dubai','phoenix','seville','yazd'])
    parser.add_argument('--year',type=int,default=2024)
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--episode-hours',type=int,default=168)
    args=parser.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    env=FacadeEnv(args.cities,args.year,args.episode_hours)
    check_env(env,warn=True)
    out=ROOT/'results/rl'/f'{MODEL_VERSION}-ppo-seed{args.seed}-{args.year}'
    out.mkdir(parents=True,exist_ok=True)
    env=Monitor(env,str(out/'episodes.csv'))
    model=PPO('MlpPolicy',env,seed=args.seed,device='cpu',verbose=1,
              n_steps=1024,batch_size=64,learning_rate=3e-4,gamma=.99,
              policy_kwargs={'net_arch':[64,64]})
    model.learn(total_timesteps=args.steps)
    model.save(str(out/'policy'))
    import hashlib
    meta={'arguments':vars(args),'model_version':MODEL_VERSION,'observation_shape':[48],
          'config_sha256':hashlib.sha256((ROOT/'config/model.json').read_bytes()).hexdigest(),
          'action_protection':'Current-weather local safety veto; PPO sees causal mask features, not masked probabilities.',
          'status':'Research training in an UNVALIDATED cassette simulator.',
          'generalization':'Training reward is not held-out or real-world performance.',
          'observation':'Current and past realized interval only. No future reanalysis.',
          'versions':{p:importlib.metadata.version(p) for p in ['stable-baselines3','gymnasium','numpy','pvlib']}}
    (out/'training.json').write_text(json.dumps(meta,indent=2))
    print('Saved local research policy:',out)


if __name__=='__main__': main()
