"""Independent published data, blocked splits, and explicit validation scope.

The small wall/rate closures below are NOT substitutions for independent
validation of 3-D CFD or of the new cassette. Missing boundary data stay missing.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from assembly import ROOT

DATA=ROOT/'data/research/experiments';OUT=ROOT/'results/research/D'
SOURCES={
 'dew':{'doi':'10.3929/ethz-b-000469811','paper_doi':'10.1126/sciadv.abf3978',
   'url':'https://www.research-collection.ethz.ch/server/api/core/bitstreams/3e254e13-807f-4887-b864-aa1a603a03e2/content',
   'attribution':'Haechler et al., 2021, Extended Data, ETH Zurich','license':'CC BY-NC 4.0; research use, no commercial license inferred'},
 'pv':{'url':'https://pvpmc.sandia.gov/app/uploads/sites/243/2022/10/Sandia_PV_Module_P-Matrix-and-TempCo-Data_2019.xlsx',
   'context':'https://pvpmc.sandia.gov/datasets/2021-blind-pv-modeling-comparison/',
   'attribution':'Sandia National Laboratories, PVPMC, 2019 measured module performance matrices'},
 'rain':{'url':'https://carmeliet.ethz.ch/research/downloads/wind-driven-rain/geometry-1.html',
   'paper_doi':'10.1016/j.buildenv.2014.04.004','reconstruction_doi':'10.1016/j.buildenv.2014.07.008',
   'attribution':'Kubilay, Derome, Blocken and Carmeliet; ETH Zurich / Empa field experiments, 2013',
   'permission':'data page permits use with acknowledgement; emailing authors appreciated, not required; no message sent'}
}

def pv(path):
    import openpyxl
    w=openpyxl.load_workbook(path,read_only=True,data_only=True);records=[];models=[]
    for sheet in w:
        rows=list(sheet.values);m=np.array([[float(v) for v in r[1:8]] for r in rows[5:32]],float)
        t,g,power=m[:,0],m[:,1],m[:,6];cal=(t>=23)&(t<=52)&(g>=400)
        design=np.c_[g/1000,g/1000*(t-25)];coef=np.linalg.lstsq(design[cal],power[cal],rcond=None)[0];pred=design@coef
        e=pred[~cal]-power[~cal];relative=e/power[~cal]
        models.append({'module':sheet.title,'calibration_n':int(cal.sum()),'validation_n':int((~cal).sum()),
          'P0_W':float(coef[0]),'gamma_per_K':float(coef[1]/coef[0]),'heldout_MAE_W':float(np.mean(abs(e))),
          'heldout_RMSE_W':float(np.sqrt(np.mean(e**2))),'heldout_MAPE_percent':float(np.mean(abs(relative))*100),
          'heldout_max_abs_percent':float(np.max(abs(relative))*100)})
        records.extend({'module':sheet.title,'temperature_C':a,'irradiance_W_m2':b,'measured_Pmp_W':p,'predicted_Pmp_W':q,
          'split':'calibration' if c else 'validation'} for a,b,p,q,c in zip(t,g,power,pred,cal))
    pd.DataFrame(records).to_csv(DATA/'pv-points.csv',index=False)
    return {'scope':'electrical Pmp(G,T) linear temperature/irradiance closure tested on nine real measured modules',
      'split':'T approximately 25/50 C and G>=400 calibration; remaining complete condition cells validation; fixed before fitting',
      'independent_measured_data':True,'outdoor_thermal_PV_validation':False,'façade_selected_module_is_identical':False,
      'measured_uncertainty_not_in_matrix':True,'parameters_not_transferred_to_cassette_BOM':True,'modules':models,
      'note':'PV weather workbook alone contains no measured Pmp; it is not used as power validation truth'}

def rain_data(path,errorpath):
    a=pd.read_excel(path,header=None).iloc[15:,:26].apply(pd.to_numeric,errors='coerce').dropna(subset=[0,1])
    names=['n','minute','U2_4','U5_4','U8_4','from_deg','Rh_mm_h','Sh_mm']+['A'+str(i) for i in range(1,10)]+['B'+str(i) for i in range(1,10)]
    a.columns=names
    if (np.diff(a.minute)<=0).any() or (np.diff(a.Sh_mm)<-1e-7).any():raise ValueError('Nonmonotonic rain record')
    error=pd.read_excel(errorpath,header=None);bounds={};excluded=[]
    for offset in [0,6]:
        for row in error.iloc[1:10].itertuples(index=False,name=None):
            name=row[offset];rel=pd.to_numeric(row[offset+5],errors='coerce')
            if not np.isfinite(rel):excluded.append(name);continue
            bounds[name]={'reported_total_mm':float(row[offset+2]),'loss_estimate_mm':float(row[offset+3]+row[offset+4]),'ETOT_percent':float(rel)}
    return a,bounds,excluded

def rain(paths):
    data=[];excluded={}
    for i,(p,e) in enumerate(paths,1):
        f,badges,mask=rain_data(p,e);data.append((f,badges));excluded[str(i)]=mask
        f.to_csv(DATA/f'rain-event{i}-normalized.csv.gz',index=False,compression='gzip')
    # West outward normal 270 degrees is documented by the original authors.
    # Reference velocity is measured at 2.4 m; no city-height pressure shortcut.
    def incident(f):
        dt=np.diff(np.r_[0.,f.minute])/60
        return f.Rh_mm_h.to_numpy()*dt*np.maximum(0,np.cos(np.radians(f.from_deg.to_numpy()-270)))*f.U2_4.to_numpy()/4.5
    base1=float(incident(data[0][0]).sum());coeff={name:float(data[0][0][name].iloc[-1]/base1) for name in data[0][1]}
    rows=[];curves=[]
    for event,(f,bounds) in enumerate(data,1):
        incoming=incident(f);qc=float(abs(np.sum(f.Rh_mm_h.to_numpy()*np.diff(np.r_[0,f.minute])/60)-f.Sh_mm.iloc[-1]))
        for gauge,b in bounds.items():
            if gauge not in coeff:continue
            prediction=np.cumsum(incoming)*coeff[gauge];obs=f[gauge].to_numpy();err=float(prediction[-1]-obs[-1])
            rows.append({'event':event,'gauge':gauge,'split':'calibration' if event==1 else 'validation','measured_total_mm':float(obs[-1]),
              'predicted_total_mm':float(prediction[-1]),'error_mm':err,'absolute_relative_percent':float(abs(err)/obs[-1]*100),
              'cumulative_curve_MAE_mm':float(np.mean(abs(prediction-obs))),**b,
              'within_reported_one_sided_adhesion_restwater_interval':bool(obs[-1]<=prediction[-1]<=obs[-1]+b['loss_estimate_mm']),
              'horizontal_Rh_integral_vs_Sh_difference_mm':qc,'total_vs_error_table_difference_mm':float(obs[-1]-b['reported_total_mm'])})
            curves.extend({'event':event,'gauge':gauge,'minute':t,'measured_cumulative_mm':o,'predicted_cumulative_mm':p} for t,o,p in zip(f.minute,obs,prediction))
    pd.DataFrame(curves).to_csv(DATA/'rain-cumulative-comparison.csv.gz',index=False,compression='gzip')
    held=[r for r in rows if r['split']=='validation']
    return {'scope':'semiempirical directional wall-flux closure, event-1 gauge shelter coefficients; no 3-D CFD validation claim',
      'model':'Rh*dt*max(cos(wind_from-270),0)*U2.4/4.5 * per-gauge coefficient',
      'calibration_events':[1],'validation_events':[2,3],'shelter_coefficients':coeff,'excluded_using_error_table_nan_ETOT':excluded,
      'error_interval_is_not_95_percent_CI':True,'measurement_loss_bounds_one_sided':True,
      'heldout_mean_abs_relative_percent':float(np.mean([r['absolute_relative_percent'] for r in held])),
      'heldout_within_loss_interval':sum(r['within_reported_one_sided_adhesion_restwater_interval'] for r in held),'heldout_comparisons':len(held),
      'results':rows,'actual_CFD_particle_solver_experimentally_validated':False,
      'geometry':{'cubes':9,'edge_m':2,'gap_m':2,'bottom_support_m':.17,'total_height_m':2.17,'array': [3,3],
        'instrumented':'west facades of cubes 2 and 8; A/B in data','outward_normal_deg_from_north':270,
        'mast_upstream_west_m':6,'wind_measurement_heights_m':[2.4,5.4,8.4],
        'BC_note':'measured U and direction every 10 min; inlet z0 differs from local asphalt z0; source paper specifies upstream sector dependence'},
      'not_applied_to_new_product':'gauge coefficients encode this nine-cube array, not the proposed building'}

def dew(path):
    import openpyxl
    w=openpyxl.load_workbook(path,read_only=True,data_only=True)
    b=np.array(list(w['Figure 4B'].values)[1:],float);raw=list(w['Figure 4C'].values)[1:];c=np.array([r[:3] for r in raw],float)
    # Figure 4B Runs 8/9 are downsampled prefixes of the two Figure 4C runs.
    idx=np.searchsorted(c[:,0],b[:,0]);duplicate=bool(np.allclose(b[:,8:10],c[idx,1:3],rtol=1e-9,atol=1e-8))
    if not duplicate:raise ValueError('Dew split overlap assumption changed; inspect source version before fitting')
    t=b[:,0];cal=b[:,1:8]
    def f(time,p):return p[0]+p[1]*np.exp(-time/p[2])
    fit=least_squares(lambda p:(f(t,p)[:,None]-cal).ravel(),[28,8,600],bounds=([0,-100,30],[100,100,3600]))
    pred=f(c[:,0],fit.x);tests=[]
    for j in [1,2]:
        e=pred-c[:,j];obs=float(np.trapezoid(c[:,j],c[:,0])/3600);pm=float(np.trapezoid(pred,c[:,0])/3600)
        tests.append({'run':'Figure4C-'+str(j),'duration_h':float((c[-1,0]-c[0,0])/3600),'rate_MAE_g_m2_h':float(np.mean(abs(e))),
          'rate_RMSE_g_m2_h':float(np.sqrt(np.mean(e**2))),'measured_integrated_g_m2':obs,'predicted_integrated_g_m2':pm,
          'integrated_relative_error_percent':float(100*(pm-obs)/obs)})
    pd.DataFrame({'time_s':c[:,0],'measured_run1_g_m2_h':c[:,1],'measured_run2_g_m2_h':c[:,2],'predicted_g_m2_h':pred}).to_csv(DATA/'dew-heldout-curves.csv.gz',index=False,compression='gzip')
    pd.DataFrame(b,columns=['time_s']+[f'run{i}_g_m2_h' for i in range(1,10)]).to_csv(DATA/'dew-short-runs.csv.gz',index=False,compression='gzip')
    # Run resampling, never resampling strongly correlated second-by-second values.
    rng=np.random.default_rng(71);means=cal[t>=1200].mean(0);boot=np.mean(rng.choice(means,(4000,len(means)),replace=True),1)
    return {'scope':'phenomenological collected-flux repeatability/relaxation benchmark only; not coupled heat/mass physical validation',
      'calibration_runs':['Figure4B-'+str(i) for i in range(1,8)],'validation_runs':['Figure4C-1','Figure4C-2'],
      'excluded_calibration_runs_due_to_prefix_duplication':[8,9],'duplicate_prefix_detected':duplicate,
      'model':'q=qinf+A exp(-t/tau), common parameters fitted to seven calibration runs; no test-run initial flux used',
      'fit':{'qinf_g_m2_h':float(fit.x[0]),'A_g_m2_h':float(fit.x[1]),'tau_s':float(fit.x[2])},
      'run_bootstrap_calibration_steady_mean_95_interval_g_m2_h':np.quantile(boot,[.025,.975]).tolist(),
      'interval_scope':'uncertainty in mean of seven calibration runs; not prediction interval or total model uncertainty',
      'heldout':tests,'geometry':{'copper_disc_diameter_m':.1016,'copper_disc_thickness_m':.0005,'Peltier_voltage_V':2.4,
        'Peltier_power_known':False,'fan_during_measurement':'off','run_specific_plate_temperature_series_available':False},
      'cooling_experiment':{'roof':'ETH Zurich, August 13-14 2019','lat':47.377747,'lon':8.547607,
        'shield_half_angles_deg':[30,90],'bottom_insulated':True,'solar_RH_Tair_measured':True,
        'Figure2B_FigureS5_same_experiment':True,'independent_cooling_runs_available':False,
        'longwave_and_wind_BCs_available':False,'thermal_validation_status':'BLOCKED_MISSING_LW_WIND_AND_INDEPENDENT_RUN',
        'not_inferred_from_observed_surface_temperature':True},
      'whole_cassette_or_CNF_lifetime_validated':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,help='folder containing downloaded pv-matrix.xlsx and rain-event files');p.add_argument('--dew-source',type=Path);a=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    mapping={'pv-matrix.xlsx':'pv-matrix.xlsx','rain-event1.xls':'rain-event1-ua.xls','rain-event2.xls':'rain-event2-ua.xls','rain-event3.xls':'rain-event3.xls',
      **{f'rain-error{i}.xls':f'rain-error{i}.xls' for i in range(1,4)}}
    if a.source_dir:
        for dest,src in mapping.items():shutil.copy2(a.source_dir/src,DATA/dest)
    if a.dew_source:shutil.copy2(a.dew_source,DATA/'dew-data.xlsx')
    required=list(mapping)+['dew-data.xlsx']
    for name in required:
        if not (DATA/name).exists():raise FileNotFoundError('Missing independently measured source '+name)
    result={'PV':pv(DATA/'pv-matrix.xlsx'),'rain':rain([(DATA/f'rain-event{i}.xls',DATA/f'rain-error{i}.xls') for i in range(1,4)]),
      'dew':dew(DATA/'dew-data.xlsx'),'sources':SOURCES,'source_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in required},
      'new_product_validated':False,'scope':'independent component/closure tests; missing physical BC and CFD validation remain explicit'}
    (OUT/'experiments.json').write_text(json.dumps(result,indent=2));(DATA/'SOURCES.json').write_text(json.dumps({'sources':SOURCES,'checksums':result['source_sha256']},indent=2))
    print(json.dumps({'PV_modules':len(result['PV']['modules']),'rain_heldout_mean_abs_percent':result['rain']['heldout_mean_abs_relative_percent'],
      'dew_heldout_integrated_percent':[r['integrated_relative_error_percent'] for r in result['dew']['heldout']], 'product_validated':False},indent=2))
if __name__=='__main__':main()
