"""Reproducible reference DOE and a queue, not CFD within each RL step."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from core import SIGMA
from assembly import ROOT
from stageb.surface import SurfaceFilm,mass_fraction_vapour

RANGES={
 'air_C':[-12,40], 'rh':[.30,.99], 'wind10_m_s':[.5,12],
 'wind_direction_deg':[-180,180], 'rain_mm_h':[0,12],
 'rain_azimuth_offset_deg':[-30,30],
 'G_plane_W_m2':[0,900], 'sky_effective_emissivity':[.45,.99],
 'angle_deg':[5,175], 'initial_offset_K':[-12,12], 'pressure_Pa':[60000,103000]}


def design(n=64,seed=42):
    if n<2:raise ValueError('At least two design points required')
    names=list(RANGES);lo,hi=np.array(list(RANGES.values())).T
    u=qmc.LatinHypercube(len(names),seed=seed).random(n)
    f=pd.DataFrame(qmc.scale(u,lo,hi),columns=names)
    f['LW_down_W_m2']=f.sky_effective_emissivity*SIGMA*(f.air_C+273.15)**4
    f['initial_C']=f.air_C+f.initial_offset_K
    f['Yv']=mass_fraction_vapour(f.air_C.to_numpy(),f.rh.to_numpy(),f.pressure_Pa.to_numpy())
    f['absolute_humidity_kg_m3']=f.rh*np.asarray(__import__('stageb.surface',fromlist=['saturation_pa']).saturation_pa(f.air_C))/(461.52*(f.air_C+273.15))
    f['bulk_Ri_building']=9.80665*11.8872*np.abs(f.initial_offset_K)/((f.air_C+273.15)*f.wind10_m_s**2)
    f['neutral_RANS_scope']=f.bulk_Ri_building<.1
    f['precipitation_phase']=np.where(f.air_C<0,'snow_SWE_input_scenario','liquid_rain')
    f['case_id']=[f'B-{i:04d}' for i in range(n)]
    return f


def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=64);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-surface',action='store_true');p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--prepare-cfd',type=int,default=0,help='Number of pending neutral-flow cases to generate, never run implicitly')
    a=p.parse_args();out=ROOT/'results/stage-b/doe';out.mkdir(parents=True,exist_ok=True)
    frame=design(a.n,a.seed);frame.to_csv(out/'design.csv',index=False)
    source_hash=hashlib.sha256((out/'design.csv').read_bytes()).hexdigest();jobs=[];summaries=[];prepared=0
    for _,w in frame.iterrows():
        neutral=bool(w.neutral_RANS_scope)
        job={'case_id':w.case_id,'status':'PENDING_NOT_EXECUTED' if neutral else 'REQUIRES_BUOYANT_SOLVER',
             'inputs':w.to_dict(),'resolution_levels':['coarse','medium','fine'],'CFD_has_latent_feedback':False,
             'design_sha256':source_hash,'independent_case_split_group':w.case_id}
        if neutral and prepared<a.prepare_cfd:
            from stageb.foam import generate
            path=out/'pending-cfd'/w.case_id
            generate(path,wind=w.wind10_m_s,angle=w.angle_deg,direction=w.wind_direction_deg,iterations=1200)
            job['path']=str(path.relative_to(ROOT));job['generated_scalar_BCs']='standard unit-response reference; not this weather thermodynamics'
            prepared+=1
        jobs.append(job)
        if a.run_surface:
            t0=float(w.initial_C);rain=float(w.rain_mm_h)/3600
            model=SurfaceFilm(n=16,angle_deg=float(w.angle_deg),initial_C=t0)
            s=model.advance(a.seconds,air_C=float(w.air_C),rh=float(w.rh),pressure=float(w.pressure_Pa),
                solar_W_m2=float(w.G_plane_W_m2),lw_in_W_m2=float(w.LW_down_W_m2),
                h_W_m2_K=4+3*float(w.wind10_m_s),hm_m_s=.005,
                rain_kg_m2_s=rain if w.air_C>=0 else 0.,snow_swe_kg_m2_s=rain if w.air_C<0 else 0.,
                back_C=float(w.air_C),max_dt=10)
            summaries.append({'case_id':w.case_id,'boundary_closure':'explicit engineering prior, not fitted CFD',**s})
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    if summaries:pd.DataFrame(summaries).to_csv(out/'surface-screening.csv',index=False)
    meta={'seed':a.seed,'n':a.n,'ranges':RANGES,'design_sha256':source_hash,
       'type':'controlled numerical experiments, not a climatological joint probability distribution',
       'LW_constraint':'epsilon_sky * sigma * T_air^4; cloud fraction not inferred',
       'altitude':'pressure varied under fixed other inputs; not a causal comparison of real cities',
       'neutral_RANS_count':int(frame.neutral_RANS_scope.sum()),'CFD_cases_executed_by_this_command':0,
       'surface_cases_executed':len(summaries),'surface_seconds':a.seconds,
       'rain_azimuth':'independent inlet direction variable for pending B2 work; integrated B3 rain input has no azimuth and does not claim to execute it',
       'RL_contract':{'no_CFD_in_environment_step':True,'train_split':'group by geometry/weather episode before fitting',
          'required_gate':['conservation','CFD convergence','grid/particle/source convergence','held-out reference error','out-of-domain rejection'],
          'surrogate_trained':False,'cold_PPO_training_authorized_by_these_references':False},
       'limitations':['supplied snow is SWE at surface; no aerodynamic snow capture prediction','weather correlations are intentionally absent in controlled DOE',
          'surface film prior h/hm cannot be passed off as qualified CFD or validation']}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
