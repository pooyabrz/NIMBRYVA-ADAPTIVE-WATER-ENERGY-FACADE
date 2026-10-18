"""Separated closure/parameter/forcing sensitivity. Ranges are scenarios, not CIs."""
import argparse,json,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from assembly import ROOT
from weather import model_config
from physics import PanelModel
from stageb.surface import SurfaceFilm
from stageb.thermal import SurfaceScene
from stagecde.control import rule
from stagecde.transitions import CACHE

OUT=ROOT/'results/research/C'
CASES=[('nominal',None,None),('external_h_low','external_h_multiplier',.7),('external_h_high','external_h_multiplier',1.3),
 ('coating_degraded','coating_condition',.5),('collector_low','collector_efficiency',.7),('collector_high','collector_efficiency',1),
 ('bridge_low','bridge_multiplier',.5),('bridge_high','bridge_multiplier',2),('motor_low','motor_running_upper_W',60),('motor_high','motor_running_upper_W',84),
 ('pinning_low','droplet_footprints_m2',1000),('pinning_high','droplet_footprints_m2',100000),
 ('Ta_minus1','Ta',-1),('Ta_plus1','Ta',1),('RH_minus3','RH',-3),('RH_plus3','RH',3),('wind_minus20pct','wind',.8),('wind_plus20pct','wind',1.2),
 ('SW_minus5pct','SW',.95),('SW_plus5pct','SW',1.05),('LW_minus15','LW',-15),('LW_plus15','LW',15),
 ('rain_minus25pct','rain',.75),('rain_plus25pct','rain',1.25),('LW_shift_minus1h','shift',-1),('LW_shift_plus1h','shift',1),('cadence_2h','cadence',2)]

def one(city,case):
    name,key,value=case;c=model_config();f=pd.read_csv(CACHE/f'train-{city}-2020-e0.csv.gz',index_col=0,parse_dates=True).iloc[:60].copy()
    if key in c:c[key]=value
    elif key=='Ta':f.air_C+=value
    elif key=='RH':f.rh_pct=np.clip(f.rh_pct+value,0,100)
    elif key=='wind':f[['wind_m_s','wind_x','wind_z']]*=value
    elif key=='SW':f[['ghi_W_m2','dni_W_m2','dhi_W_m2']]*=value
    elif key=='LW':f.lw_down_W_m2+=value
    elif key=='rain':f.rain_mm_h*=value
    elif key=='shift':f.lw_down_W_m2=f.lw_down_W_m2.shift(value)
    elif key=='cadence':
        # Means preserve interval-integrated rain and radiative energy, not moist-air enthalpy.
        for col in ['air_C','rh_pct','wind_m_s','wind_x','wind_z','rain_mm_h','ghi_W_m2','dni_W_m2','dhi_W_m2','lw_down_W_m2']:
            f[col]=f[col].groupby(np.arange(len(f))//2).transform('mean').to_numpy()
    f=f.iloc[2:-2] # same coverage after +/-1h shift; no filling of missing edges
    scene=SurfaceScene(initial_C=float(f.iloc[0].air_C));model=PanelModel(c);model.reset(float(f.iloc[0].air_C));rows=[]
    for i,(t,w) in enumerate(f.iterrows()):
        sw=w.dni_W_m2*max(0,w.sun_z)*(w.sun_y>0)+.5*w.dhi_W_m2+.1*w.ghi_W_m2
        for _ in range(60):b=scene.step(w.air_C,sw,w.ghi_W_m2,w.lw_down_W_m2,w.wind_m_s,60)
        w['wall_C']=b['wall_C'];w['ground_C']=b['ground_C']
        # Prescribed fixed actions remove policy threshold changes from physical sensitivity.
        a=2 if w.ghi_W_m2>100 else 7
        try:r=model.step_hour(w,a)
        except (FloatingPointError,ValueError) as exc:return {'city':city,'case':name,'status':'SOLVER_FAILED','error':str(exc)}
        if i>=12:rows.append(r)
    z=pd.DataFrame(rows)
    return {'city':city,'case':name,'status':'COMPLETED','hours':len(z),'water_L':float(z.drain_L.sum()),'PV_Wh':float(z.pv_gross_Wh.sum()),
      'motor_Wh':float(z.motor_Wh.sum()),'surface_dew_mean_C':float(z.surface_dew_C.mean()),'unsupported_hours':int((z.water_supported==0).sum()),
      'mass_residual_max_kg':float(z.mass_residual_kg.abs().max()),'battery_residual_max_Wh':float(z.battery_residual_Wh.abs().max())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(one,city,k) for city in ['singapore','bandar-abbas'] for k in CASES]
        for f in as_completed(fs):
            r=f.result();records.append(r);print(r['city'],r['case'],r['status'],flush=True)
    for r in records:
        base=next(x for x in records if x['city']==r['city'] and x['case']=='nominal')
        if r['status']=='COMPLETED':r['difference_from_nominal']={k:r[k]-base[k] for k in ['water_L','PV_Wh','motor_Wh','surface_dew_mean_C']}
    closures=[]
    for ta,rh in [(20,.9),(35,.95),(45,.95)]:
        for method in ['stefan','linear_mass_fraction']:
            m=SurfaceFilm(n=32,initial_C=ta)
            s=m.advance(1800,air_C=ta,rh=rh,lw_in_W_m2=300,h_W_m2_K=5,hm_m_s=.005,back_C=ta,mass_transfer_model=method)
            closures.append({'air_C':ta,'rh':rh,'closure':method,**s})
    result={'scenarios':records,'mass_transfer_closures':closures,'fixed_duration_s':1800,
      'ranges_are_probability_distributions':False,'measurement_uncertainty_calibrated':False,
      'numerical_error':'separate grid/time results, not included in these physical/forcing differences',
      'cadence_note':'2-hour averaging preserves integrated rain/irradiance; nonlinear state averaging and source timing still change results',
      'temperature_perturbation':'RH held fixed; absolute humidity changes explicitly; not a pure-elevation causal experiment',
      'wind_note':'wind components scaled consistently; CFD topology is not recomputed for every perturbation',
      'policy_schedule':'PV action2 when realized GHI>100 otherwise emitter7; prescribed diagnostic schedule, not causal controller evaluation',
      'physical_yield_qualified':False}
    (OUT/'sensitivity.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
