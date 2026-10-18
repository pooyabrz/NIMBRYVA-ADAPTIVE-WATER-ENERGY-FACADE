"""Factorial grid/time study. Steady SIMPLE iterations are not physical time."""
import argparse,json,math,time
import numpy as np
import pandas as pd
from assembly import ROOT
from physics import PanelModel
from weather import model_config
from stageb.thermal import SurfaceScene
from stageb.surface import SurfaceFilm,decode_energy
OUT=ROOT/'results/research/C'

def mixed_error(a,b,rel,absolute):
    a=np.asarray(a);b=np.asarray(b);e=np.abs(a-b);limit=np.maximum(absolute,rel*np.maximum(np.abs(a),np.abs(b)))
    return {'pass':bool(np.all(e<=limit)),'max_abs_error':float(e.max()),'max_ratio_to_allowance':float((e/limit).max())}

def forcing(hour):
    sun=max(0,math.sin(math.pi*((hour%24)-6)/12));g=800*sun
    return pd.Series({'air_C':25+3*sun,'rh_pct':85-15*sun,'pressure_Pa':101325.,'wind_m_s':2.,'wind_x':0.,'wind_z':-2.,
      'lw_down_W_m2':330+35*sun,'ghi_W_m2':g,'dni_W_m2':700 if sun>0 else 0.,'dhi_W_m2':100*sun,
      'rain_mm_h':1. if 16<=hour%24<18 else 0.,'snowfall_cm_h':0.,'sun_x':0.,'sun_y':sun,
      'sun_z':math.sqrt(max(0,1-sun*sun)) if sun>0 else 0.},name=pd.Timestamp('2024-07-15T00:00Z')+pd.Timedelta(hours=hour))

def scene_cassette(cells,dt,hours,face=1):
    c=model_config();c['integration_dt_s']=dt;scene=SurfaceScene(cells,8*cells,initial_C=25)
    model=PanelModel(c);model.reset(25,face=face);rows=[];start=time.time()
    for h in range(hours):
        w=forcing(h);n=math.ceil(3600/dt)
        for _ in range(n):s=scene.step(w.air_C,.5*w.ghi_W_m2,w.ghi_W_m2,w.lw_down_W_m2,w.wind_m_s,3600/n)
        w['wall_C']=s['wall_C'];w['ground_C']=s['ground_C'];r=model.step_hour(w,face*len(c['tilts_deg'])+2)
        rows.append({'hour':h,'wall_C':s['wall_C'],'ground_C':s['ground_C'],'surface_C':r['surface_dew_C'],
          'water_kg':r['drain_L'],'pv_Wh':r['pv_gross_Wh'],'motor_Wh':r['motor_Wh'],'mass_residual_kg':r['mass_residual_kg'],
          'thermal_residual_J':r['max_thermal_residual_J'],'battery_residual_Wh':r['battery_residual_Wh'],'physical_scope_supported':r['water_supported']})
    df=pd.DataFrame(rows);tag=f'coupled-n{cells}-dt{dt:g}'+('-pv' if face==0 else '');df.to_csv(OUT/(tag+'.csv'),index=False)
    return {'tag':tag,'grid':cells,'ground_cells':8*cells,'dt':dt,'hours':hours,'runtime_s':time.time()-start,
      'surface_C':df.surface_C.tolist(),'wall_C':df.wall_C.tolist(),'ground_C':df.ground_C.tolist(),
      'water':float(df.water_kg.sum()/c['area_m2']/hours),'energy':float(df.pv_Wh.sum()/c['area_m2']/hours),
      'mass_residual_kg_max':float(df.mass_residual_kg.abs().max()),'thermal_residual_J_max':float(df.thermal_residual_J.abs().max()),
      'battery_residual_Wh_max':float(df.battery_residual_Wh.abs().max()),'unsupported_hours':int((df.physical_scope_supported==0).sum()),
      'scope':'B1 spatial grid + cassette time step; eight-node cassette spatial model is not refined by B1 grid'}

def film_case(n,dt,case):
    m=SurfaceFilm(n=n,initial_C=-8 if case=='snow' else 20,initial_ice_kg_m2=.05 if case=='snow' else 0)
    x=(np.arange(n)+.5)/n;rain=.0004*(1+.5*np.cos(2*np.pi*x)) if case=='rain' else 0.
    s=m.advance(3600,air_C=-8 if case=='snow' else 20,rh=.9,lw_in_W_m2=220 if case=='snow' else 300,
      h_W_m2_K=5,hm_m_s=.005,rain_kg_m2_s=rain,snow_swe_kg_m2_s=2e-5 if case=='snow' else 0.,back_C=-8 if case=='snow' else 20,max_dt=dt)
    t,_,_=decode_energy(m.energy,m.mass,m.plate_C)
    return {'case':case,'grid':n,'dt':dt,'surface_C':float(t.mean()),'water':s['drain_kg']/(1.134*1.762),
      'energy':s['energy_vapour_J']/(3600*1.134*1.762),**s}

def compare(rows,grids,dts,targets):
    pairs=[]
    for fixed,varied,levels in [('dt','grid',grids),('grid','dt',dts)]:
        for value in sorted({r[fixed] for r in rows}):
            group=sorted((r for r in rows if r[fixed]==value),key=lambda r:levels.index(r[varied]))
            for a,b in zip(group,group[1:]):
                w=mixed_error(a['water'],b['water'],targets['water_relative'],targets['water_absolute_kg_m2_h'])
                e=mixed_error(a['energy'],b['energy'],targets['energy_relative'],targets['energy_absolute_Wh_m2_h'])
                t=mixed_error(a['surface_C'],b['surface_C'],0,targets['surface_temperature_absolute_K'])
                pairs.append({'fixed':{fixed:value},'varied':varied,'from':a[varied],'to':b[varied],
                    'water':w,'energy':e,'temperature':t,'all_targets_pass':w['pass'] and e['pass'] and t['pass']})
    return pairs

def main():
    p=argparse.ArgumentParser();p.add_argument('--hours',type=int,default=24);p.add_argument('--only',choices=['all','film','coupled'],default='all');p.add_argument('--face',type=int,choices=[0,1],default=1);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);targets=json.loads((ROOT/'config/research/protocol.json').read_text())['numerical_targets'];result={'targets':targets}
    if a.only in ['all','coupled']:
        rows=[]
        for n in [2,4,8]:
            for dt in [120,60,30]:
                r=scene_cassette(n,dt,a.hours,a.face);rows.append(r);print(r['tag'],flush=True)
        result['coupled']=rows;result['coupled_comparisons']=compare(rows,[2,4,8],[120,60,30],targets)
    if a.only in ['all','film']:
        result['films']={};result['film_comparisons']={}
        for case in ['dew','rain','snow']:
            rows=[film_case(n,dt,case) for n in [16,32,64] for dt in [20,10,5]];result['films'][case]=rows
            result['film_comparisons'][case]=compare(rows,[16,32,64],[20,10,5],targets);print('film',case,flush=True)
    result['CFD_grid_study']=json.loads((ROOT/'results/stage-b/release.json').read_text())['grid_quantities_of_interest_changes']
    result['steady_CFD_note']='SIMPLE iteration/deltaT is not physical transient time'
    result['whole_device_independence_established']=False
    result['why_not_whole_device']='CFD load grid dependence, eight-node spatial approximation and unresolved physical models remain distinct'
    (OUT/(f'numerics-{a.only}'+('-pv' if a.face==0 else '')+'.json')).write_text(json.dumps(result,indent=2));print('C numerical study completed',flush=True)
if __name__=='__main__':main()
