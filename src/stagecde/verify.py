"""Workflow and causality regressions, not experimental model validation."""
import copy,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ROOT
from stagecde.numerics import mixed_error
from stagecde.convergence import richardson_three
from stagecde.plant import ResearchEnv,conserve_water,battery_step
from stagecde.experiments import rain_data,DATA
from stagecde.surrogate import Emulator,load_split
from stageb.rain import case_signature
from stageb.surface import SurfaceFilm

def main():
    checks=[]
    def check(name,value,**extra):checks.append({'name':name,'pass':bool(value),**extra})
    check('absolute criterion works near zero water',mixed_error(0,5e-5,.02,1e-4)['pass'] and not mixed_error(0,3e-4,.02,1e-4)['pass'])
    r=richardson_three(1.16,1.04,1.01);check('Richardson recovers known order two',abs(r['observed_order']-2)<1e-10 and abs(r['extrapolated']-1)<1e-10)
    check('oscillatory grid result does not invent GCI',richardson_three(1,1.1,1.05)['GCI'] is None)
    rng=np.random.default_rng(15);res=[]
    for _ in range(100):
        old=rng.random(3);raw=rng.random(8);raw[2]*=10;w=conserve_water(old,raw)
        res.append(old.sum()+w[0]+w[1]-w[2]-w[3:].sum())
    check('surrogate projection closes water for wet dry evaporation cases',np.max(abs(np.array(res)))<1e-12)
    raw=conserve_water(np.zeros(3),np.zeros(8));check('dry zero inventory remains exactly zero',np.all(raw==0))
    env=ResearchEnv('selection',False);env.reset(seed=3);c=env.c
    for soc,pv,motor in [(0,0,20),(200,200,0),(100,0,1),(100,100,10)]:
        new,b=battery_step(soc,pv,motor,c);check(f'battery ledger soc={soc} pv={pv} motor={motor}',abs(b['battery_residual_Wh'])<1e-10 and 0<=new<=200)
    obs=env._obs().copy();env.frame=env.frame.copy();env.frame.iloc[1:,env.frame.columns.get_loc('rain_mm_h')]+=100
    check('observation cannot see changed future reanalysis',np.array_equal(obs,env._obs()) and obs.shape==(50,))
    from stagecde.control import action
    a=action('rule',env);env.frame.iloc[1:,env.frame.columns.get_loc('ghi_W_m2')]+=5000
    check('causal rule sees current row only',a==action('rule',env))
    protocol=json.loads((ROOT/'config/research/protocol.json').read_text())['weather_split']
    check('train selection test years disjoint',not(set(protocol['train_years'])&set(protocol['selection_years'])) and not(set(protocol['test_years'])&set(protocol['train_years']+protocol['selection_years'])))
    check('unseen test cities excluded from training',not(set(protocol['train_cities'])&set(protocol['test_only_cities'])))
    for role in ['train','selection','test']:
        x,y,files=load_split(role);check(role+' features finite and supported label binary',np.isfinite(x).all() and np.isfinite(y).all() and x.shape[1]==49 and y.shape[1]==20 and set(np.unique(y[:,-1]))<={0.,1.})
    f,b,excluded=rain_data(DATA/'rain-event1.xls',DATA/'rain-error1.xls')
    check('nonoperating rain gauges excluded by source uncertainty table',set(excluded)=={'B4','B8'})
    check('rain amount is cumulative not summed over rows',f.Sh_mm.iloc[-1]<f.Sh_mm.sum() and (np.diff(f.Sh_mm)>=-1e-10).all())
    d=json.loads((ROOT/'results/research/D/experiments.json').read_text())
    check('dew overlapping short runs excluded before holdout',d['dew']['duplicate_prefix_detected'] and d['dew']['excluded_calibration_runs_due_to_prefix_duplication']==[8,9])
    check('independent PV data do not qualify unknown cassette BOM',len(d['PV']['modules'])==9 and d['PV']['parameters_not_transferred_to_cassette_BOM'])
    check('missing radiative cooling boundary data remains a blocked gate',d['dew']['cooling_experiment']['thermal_validation_status'].startswith('BLOCKED'))
    check('experimental component fits do not label new product validated',not d['new_product_validated'] and not d['rain']['actual_CFD_particle_solver_experimentally_validated'])
    model=Emulator();check('all model weights match frozen source hashes',all(hashlib.sha256((model.directory/n).read_bytes()).hexdigest()==sha for n,sha in model.meta['weights_sha256'].items()))
    env.reset(seed=3);state=copy.deepcopy(env.plant.state);w=env.frame.iloc[1];r=env.plant.step_hour(w,2)
    check('fast plant inventory and battery closure',abs(r['mass_residual_kg'])<1e-10 and abs(r['battery_residual_Wh'])<1e-9 and not r['physical_yield_claim_supported'])
    base=json.loads((ROOT/'results/stage-b/cfd/reference-medium/case.json').read_text());alt={**base,'RAS_model':'kOmegaSST'}
    check('turbulence model change invalidates old CFD signature',case_signature(base)!=case_signature(alt))
    film=SurfaceFilm(n=4);r=film.advance(10,mass_transfer_model='linear_mass_fraction');check('alternate gas closure conserves material and enthalpy',abs(r['mass_residual_kg'])<1e-10 and abs(r['energy_residual_J'])<1e-5)
    s=json.loads((ROOT/'results/research/E/control/selection.json').read_text());t=json.loads((ROOT/'results/research/E/control/test.json').read_text())
    check('actual three PPO seeds trained without final test selection',len(s['candidate_runs'])==3 and all(x['actual_timesteps']>=50000 for x in s['candidate_runs']) and not s['test_used_for_selection'])
    check('frozen policy and selection hashes match final test',hashlib.sha256((ROOT/'results/research/E/control'/t['policy']).read_bytes()).hexdigest()==t['policy_sha256'] and hashlib.sha256((ROOT/'results/research/E/control/selection.json').read_bytes()).hexdigest()==t['selection_sha256'])
    check('controller test includes four methods with identical episode count',set(x['controller'] for x in t['episodes'])=={'fixed','rule','mpc','ppo'} and len(t['episodes'])==64)
    result={'checks':checks,'passed':sum(x['pass'] for x in checks),'total':len(checks),'experimental_validation':False}
    (ROOT/'results/research/checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['passed']!=result['total']:raise SystemExit(1)
if __name__=='__main__':main()
