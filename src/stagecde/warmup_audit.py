"""Replay the fully specified warm-up history without overwriting training data."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from assembly import ROOT
from weather import load_weather,model_config
from physics import PanelModel
from stageb.thermal import SurfaceScene
from stagecde.transitions import CACHE,PARAMS,RANGES

def one(path):
    meta=json.loads(path.read_text());c=model_config();c.update(meta['config']);f=load_weather(meta['city'],meta['year'],c);start=f.index.get_loc(meta['start']);part=f.iloc[start-12:start+1].copy()
    scene=SurfaceScene(initial_C=float(part.iloc[0].air_C));walls=[];grounds=[]
    for _,w in part.iterrows():
        sw=w.dni_W_m2*max(0,w.sun_z)*(w.sun_y>0)+.5*w.dhi_W_m2+.1*w.ghi_W_m2
        for _ in range(60):b=scene.step(w.air_C,sw,w.ghi_W_m2,w.lw_down_W_m2,w.wind_m_s,60)
        walls.append(b['wall_C']);grounds.append(b['ground_C'])
    part['wall_C']=walls;part['ground_C']=grounds;model=PanelModel(c);model.reset(float(part.iloc[0].air_C));rng=np.random.default_rng(meta['seed'])
    if meta['role']=='train':
        for k in PARAMS:rng.uniform(*RANGES[k])
        model.state.battery_Wh=float(rng.uniform(80,160))
    else:model.state.battery_Wh=100.
    resets=[]
    for i in range(1,13):
        try:model.step_hour(part.iloc[i],0)
        except FloatingPointError as exc:resets.append({'time':str(part.index[i]),'error':str(exc)});model.reset(float(part.iloc[i].air_C))
    # First saved teacher input must equal the replayed old state; this is a strong provenance check.
    npz=ROOT/'results/research/E/transitions'/(meta['case_group']+'.npz');x=np.load(npz)['X'][0]
    delta=float(np.max(abs(x[:8]-model.state.temperature_C)));film=float(np.max(abs(x[8:11]-model.state.film_kg)))
    return {'group':meta['case_group'],'warmup_resets':resets,'first_input_temperature_difference_K':delta,'first_input_film_difference_kg':film,'reconstruction_pass':delta<1e-8 and film<1e-8}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one,p) for p in sorted(CACHE.glob('*.json'))]):records.append(f.result())
    result={'records':records,'all_reconstructed':all(r['reconstruction_pass'] for r in records),'resets':sum(len(r['warmup_resets']) for r in records),'training_data_overwritten':False}
    (ROOT/'results/research/E/warmup-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
    if not result['all_reconstructed']:raise SystemExit(1)
if __name__=='__main__':main()
