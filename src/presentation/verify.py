"""Accounting/merge regressions; does not establish experimental accuracy."""
import copy,hashlib,json,importlib.util,sys
import numpy as np
from assembly import ROOT
from weather import model_config
from physics import PanelModel
from stagecde.numerics import forcing
from presentation.accounting import AuditObserver,water_update,withdraw_mixed

def main():
    checks=[]
    def check(name,test,**kw):checks.append({'name':name,'pass':bool(test),**kw})
    stock=np.array([.008,.002,0.]);remaining,evap,collected,loss=water_update(stock,0,.002,.002,.0024,.0006,.001,.8)
    check('mixed water plus condensation-first jumping closes each origin',np.allclose(stock+[0,.002,0]-evap-collected-loss,remaining) and np.all(remaining>=0) and collected[1]>.0008)
    rng=np.random.default_rng(384)
    for i in range(40):
        x=rng.random(3);mass=rng.uniform(0,x.sum());take,left=withdraw_mixed(x,mass)
        check('mixed withdrawal conservative '+str(i),abs(take.sum()-mass)<1e-12 and np.max(abs(take+left-x))<1e-12)
    c=model_config();audit=AuditObserver(c);audit.reset(np.zeros(3),100);audit.energy_step(0,19,80)
    r=audit.finish_hour(np.zeros(3),80);check('initial battery energy never claimed as PV',r['initial_battery_Wh']==19 and r['pv_useful_in_system_Wh']==0)
    audit.reset(np.zeros(3),0);audit.energy_step(100,20,71.25);audit.energy_step(0,50,71.25-50/.95);r=audit.finish_hour(np.zeros(3),71.25-50/.95)
    check('PV usefulness excludes conversion battery losses and curtailment',abs(100-r['pv_useful_in_system_Wh']-r['conversion_loss_Wh']-r['pv_charge_loss_Wh']-r['pv_discharge_loss_Wh']-r['surplus_Wh'])<1e-10)
    a=PanelModel(c);b=PanelModel(c,AuditObserver(c));a.reset(25);b.reset(25)
    differences=[]
    for h,action in [(8,2),(9,7),(10,7),(16,2),(17,7)]:
        w=forcing(h);x=a.step_hour(w,action);y=b.step_hour(w,action)
        differences.append(max(abs(x[k]-y[k]) for k in ['drain_L','film_L','pv_gross_Wh','battery_Wh','mass_residual_kg']))
        check('tracers close within physical substeps '+str(h),y['origin_balance_max_L']<1e-9 and abs(y['drain_L']-sum(y['collected_'+s+'_L'] for s in ['rain','condensation','initial']))<1e-9)
        check('PV stock accounting across one hour '+str(h),abs(y['pv_gross_Wh']-y['conversion_loss_Wh']-y['pv_useful_in_system_Wh']-y['pv_charge_loss_Wh']-y['pv_discharge_loss_Wh']-y['surplus_Wh'])<1e-8)
    check('auditing leaves all physical results unchanged',max(differences)==0,max_difference=max(differences))
    spec=importlib.util.spec_from_file_location('_archived_CDE1_physics',ROOT/'archive_CDE1/src/physics.py');archived=importlib.util.module_from_spec(spec);sys.modules[spec.name]=archived;spec.loader.exec_module(archived)
    oldmodel=archived.PanelModel(c);newmodel=PanelModel(c,AuditObserver(c));oldmodel.reset(25);newmodel.reset(25);deltas=[]
    for h,a in [(8,2),(9,7),(16,2),(17,7)]:
        x=oldmodel.step_hour(forcing(h),a);y=newmodel.step_hour(forcing(h),a);deltas.append(max(abs(x[k]-y[k]) for k in ['drain_L','film_L','pv_gross_Wh','battery_Wh','surface_dew_C']))
    check('physical calculations match archived delivered CDE1 source',max(deltas)==0,max_difference=max(deltas))
    p=json.loads((ROOT/'config/presentation/protocol.json').read_text());old=json.loads((ROOT/'config/research/protocol.json').read_text())['weather_split']
    check('new 2025 weather is outside all previous fitting and testing years',p['test_year'] not in old['train_years']+old['selection_years']+old['test_years']+[2024])
    merge=json.loads((ROOT/'config/provenance/merge-record.json').read_text());check('attached base has no removed files',merge['base_files_removed']==[] and merge['base_files_identical']==824)
    src=ROOT/'results/research/E/control';s=json.loads((src/'selection.json').read_text());t=json.loads((src/'test.json').read_text());check('policy unchanged since CDE test',hashlib.sha256((ROOT/p['frozen_policy']).read_bytes()).hexdigest()==t['policy_sha256'] and s['chosen_policy']==p['frozen_policy'].rsplit('/',1)[1])
    actual=ROOT/'results/presentation/hourly-audited.csv.gz'
    if actual.exists():
        import pandas as pd
        f=pd.read_csv(actual);mass=abs(f.collected_rain_L+f.collected_condensation_L+f.collected_initial_L-f.drain_L).max()
        energy=abs(f.pv_gross_Wh-f.pv_useful_in_system_Wh-f.conversion_loss_Wh-f.pv_charge_loss_Wh-f.pv_discharge_loss_Wh-f.surplus_Wh).max()
        check('all 3840 realized rows conserve collected source identities',len(f)==3840 and mass<1e-9,max_error_L=float(mass))
        check('all realized PV ledgers close without initial energy credit',energy<1e-8,max_error_Wh=float(energy))
        monotonic=True
        for _,z in f.groupby(['group','controller']):monotonic &= bool(np.all(np.diff(z.sort_values('hour').common_valid_prefix.astype(int))<=0))
        check('censored trajectories never resume after unsupported state',monotonic)
        check('active cooling is explicitly absent in all realized rows',np.all(f.active_cooling_Wh==0))
    result={'checks':checks,'passed':sum(x['pass'] for x in checks),'total':len(checks),'experimental_validation':False}
    out=ROOT/'results/presentation';out.mkdir(exist_ok=True,parents=True);(out/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['passed']!=result['total']:raise SystemExit(1)
if __name__=='__main__':main()
