"""Auditable connections between B1 scene and the existing cassette model.

Cold phase-change references are distinct from the v2 eight-node cassette:
they do not silently enable an unvalidated cold-weather PPO environment.
"""
import argparse,json,math
import numpy as np
import pandas as pd
from assembly import ROOT
from core import SIGMA
from weather import load_weather,model_config
from physics import PanelModel
from stageb.surface import SurfaceFilm


def connect_thermal(city,month,hours=12):
    path=ROOT/f'results/stage-b/thermal/{city}-2024-{month:02d}-dt60-r1.csv.gz'
    if not path.exists():raise ValueError('Run stage-b thermal for this city/month first')
    scene=pd.read_csv(path,parse_dates=['time']).set_index('time');cfg=model_config();w=load_weather(city,2024,cfg)
    merged=w.loc[scene.index].copy();merged['wall_C']=scene.wall_C;merged['ground_C']=scene.ground_C
    merged['boundary_model']='B1-DOE-wall+dry-ground-zone-RC';merged['scored']=scene.scored
    scored=np.flatnonzero(merged.scored.to_numpy());last=scored[0]+hours;merged=merged.iloc[:last]
    base=PanelModel(cfg);with_scene=PanelModel(cfg)
    base.reset(float(merged.iloc[0].air_C),face=1);with_scene.reset(float(merged.iloc[0].air_C),face=1)
    rows=[];failure=None
    for when,row in merged.iterrows():
        if row.air_C<2 or float(row.get('snowfall_cm_h',0))>0:
            failure={'time':str(when),'status':'COLD_OR_SNOW_V2_EXCLUDED','reason':'Use independent enthalpy reference; v2 cassette cold scope remains excluded'};break
        try:
            b=base.step_hour(row.drop(labels=['wall_C','ground_C']),len(cfg['tilts_deg'])+2)
            s=with_scene.step_hour(row,len(cfg['tilts_deg'])+2)
        except FloatingPointError as exc:
            failure={'time':str(when),'status':'NUMERICAL_STEP_REJECTED','reason':str(exc)};break
        rows.append({'time':str(when),'scored':bool(row.scored),'air_C':row.air_C,'wall_C':row.wall_C,'ground_C':row.ground_C,
          'base_emitter_C':b['surface_dew_C'],'scene_emitter_C':s['surface_dew_C'],
          'emitter_delta_K':s['surface_dew_C']-b['surface_dew_C'],'base_condensed_L':b['condensed_L'],
          'scene_condensed_L':s['condensed_L'],'scene_drained_L':s['drain_L'],
          'scene_water_supported_v2':s['water_supported'],'frost_excluded_v2':s['frost_unsupported'],
          'snow_excluded_v2':s['snow_unsupported'],'max_thermal_residual_J':s['max_thermal_residual_J'],
          'boundary_from_scene':s['external_surface_model_boundary']})
    frame=pd.DataFrame(rows);out=ROOT/'results/stage-b/coupled';out.mkdir(parents=True,exist_ok=True)
    tag=f'{city}-2024-{month:02d}'
    if len(frame):frame.to_csv(out/(tag+'.csv.gz'),index=False,compression='gzip')
    selected=frame[frame.scored] if len(frame) else frame
    report={'city':city,'month':month,'hours_scored':len(selected),'cassette_warmup_hours':int((~frame.scored).sum()) if len(frame) else 0,
      'status':'COMPLETE_REFERENCE' if failure is None else failure['status'],'rejected_step':failure,
      'mean_emitter_shift_K':float(selected.emitter_delta_K.mean()) if len(selected) else None,'max_emitter_shift_K':float(selected.emitter_delta_K.abs().max()) if len(selected) else None,
      'v2_supported_scored_hours':int(selected.scene_water_supported_v2.sum()) if len(selected) else 0,
      'connection':'B1 wall and ground temperatures -> finite-view spectral radiation of v2 cassette',
      'CFD_h_hm_or_rain_closure_used':False,'two_way_CFD_CHT':False,'physical_yield_claim_supported':False,
      'reason':'Current CFD scalar wall treatment is not qualified for transfer coefficients; cold v2 exclusions remain'}
    (out/(tag+'.json')).write_text(json.dumps(report,indent=2));print(tag,report['max_emitter_shift_K'],flush=True)
    return report


def reference_join(case,hours=1,rain_result=None):
    """One operating point joins actual B2 rain with B1 and B3, as a scenario.

    CFD rain impacts -> retention scenario -> B3 film. B1 supplies wall/ground
    longwave. h/hm remain declared priors: no claim of full conjugate CFD.
    """
    from stageb.rain import load_case
    from stageb.thermal import SurfaceScene
    centres,u,extra,meta,latest=load_case(case)
    ta=meta['T_ambient_K']-273.15;p=101325.;yv=meta['Yv_inlet']
    e=yv*p/(.621945+.378055*yv)
    from stageb.surface import saturation_pa
    rh=e/float(saturation_pa(ta));wind=meta['wind10_m_s']
    qpath=__import__('pathlib').Path(case)/'quality.json';quality=json.loads(qpath.read_text())
    rain=0.;source=None;spatial=False
    if rain_result:
        rr=json.loads(__import__('pathlib').Path(rain_result).read_text())
        if rr.get('source_scheme')!='union_of_upstream_and_top_boundary_bundles_v2':raise ValueError('Superseded single-plane rain source is excluded; rerun with current rain code')
        if rr['CFD_U_sha256']!=meta['U_sha256']:raise ValueError('Rain/flow provenance mismatch')
        if 'deposited_profile_kg_s_scenario' in rr:
            rain=np.array(rr['deposited_profile_kg_s_scenario'])[0]/(meta['area_m2']/32);spatial=True
        else:rain=rr['total_retained_kg_s_scenario']/meta['area_m2']
        source=str(rain_result)
    scene=SurfaceScene(initial_C=ta);cfg=model_config();panel=PanelModel(cfg)
    panel.reset(ta,face=1);forcing={'sun_x':0,'sun_y':0,'sun_z':0,'dni_W_m2':0,'dhi_W_m2':0,'ghi_W_m2':0,'rain_mm_h':0}
    vf=panel._geometry(230,forcing)['vf'][1]
    film=SurfaceFilm(n=32,angle_deg=meta['panel_angle_deg'],initial_C=meta['T_panel_K']-273.15)
    rows=[]
    for i in range(math.ceil(hours*60)):
        s=scene.step(ta,0,0,300,wind,60)
        lw=vf[0]*300+vf[1]*SIGMA*(s['ground_C']+273.15)**4+vf[2]*SIGMA*(s['wall_C']+273.15)**4
        r=film.advance(60,air_C=ta,rh=rh,pressure=p,lw_in_W_m2=lw,h_W_m2_K=4+3*wind,
            hm_m_s=(4+3*wind)/(1.2*1006),rain_kg_m2_s=rain,back_C=ta,max_dt=5)
        rows.append({'time_s':(i+1)*60,'wall_C':s['wall_C'],'ground_C':s['ground_C'],**r})
    out=ROOT/'results/stage-b/coupled';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'joined-reference.csv.gz',index=False,compression='gzip')
    report={'case':str(case),'rain_reference':source,'CFD_cells':len(centres),'CFD_flow_gate':quality['flow_available_for_exploratory_particle_tracking'],
      'retained_rain_surface_flux_kg_m2_s_scenario':np.asarray(rain).tolist(),'spatial_rain_profile_used':spatial,
      'selected_face':'positive thickness face, emitter assumed exposed by 180 degree face switch; identical external box geometry',
      'edge_deposit_not_routed_to_selected_face_kg_s':rr['edge_deposit_kg_s_scenario'] if rain_result else 0.,
      'opposite_face_deposit_not_routed_kg_s':float(np.sum(rr['deposited_profile_kg_s_scenario'][1])) if rain_result else 0.,
      'view_factors_sky_ground_wall':vf.tolist(),
      'last_surface_state':rows[-1],'coupling':'B1 longwave -> B3, B2 rain scenario -> B3',
      'CFD_h_hm_used':False,'two_way_CFD_CHT':False,'radiation_for_this_join':'grey thin-surface reference; v2 cassette retains measured spectral-emitter curve',
      'qualification':'demonstration of conservative interfaces, not validated integrated water yield'}
    (out/'joined-reference.json').write_text(json.dumps(report,indent=2));print('joined-reference completed',flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--cities',nargs='+',default=['rasht','singapore','london','reykjavik','la-paz'])
    p.add_argument('--months',type=int,nargs='+',default=[1,7]);p.add_argument('--hours',type=int,default=12)
    p.add_argument('--case',type=str);p.add_argument('--rain-result',type=str);a=p.parse_args()
    if a.case:reference_join(a.case,a.hours,a.rain_result)
    else:
        for city in a.cities:
            for month in a.months:connect_thermal(city,month,a.hours)


if __name__=='__main__':main()
