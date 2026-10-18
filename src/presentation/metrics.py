"""Inspectable denominators, strict continuity and non-additive energy ratios."""
import json
import numpy as np
import pandas as pd
from assembly import ROOT
from weather import model_config
from actuator import load_envelope,allowable_wind
from assembly import compile_assembly

OUT=ROOT/'results/presentation'
SUMS=['rain_input_L','condensed_L','evaporated_L','uncollected_departure_L','drain_L','geometric_incident_rain_L','rain_entry_loss_L',
      'collected_rain_L','collected_condensation_L','collected_initial_L','evaporated_rain_L','evaporated_condensation_L','gutter_loss_rain_L','gutter_loss_condensation_L',
      'pv_gross_Wh','dc_available_Wh','pv_useful_in_system_Wh','surplus_Wh','conversion_loss_Wh','charge_loss_Wh','discharge_loss_Wh','pv_direct_Wh','pv_battery_Wh','initial_battery_Wh',
      'motor_Wh','idle_demand_Wh','active_cooling_Wh','unserved_Wh','changed','motion_seconds','movement_angle_deg',
      'causal_wind_interlock','causal_energy_interlock','action_blocked','wet_emitter_dry_optics_approximation']

def summarize(frame):
    c=model_config();r={'hours':len(frame),'area_m2':c['area_m2']}
    for k in SUMS:r[k]=float(frame[k].sum()) if len(frame) else 0.
    n=frame.group.nunique() if len(frame) else 0;r['independent_startups']=int(n);r['initial_battery_stock_Wh']=100*n
    ends=frame.sort_values('hour').groupby('group').tail(1)
    for source in ['rain','condensation','initial']:
        col='stored_'+source+'_L';r[col]=float(ends[col].sum()) if len(frame) else 0.
    r['final_battery_stock_Wh']=float(ends.battery_Wh.sum()) if len(frame) else 0.
    r['requested_control_Wh']=r['motor_Wh']+r['idle_demand_Wh'];r['modeled_control_Wh']=max(0,r['requested_control_Wh']-r['unserved_Wh']);r['modeled_control_and_cooling_Wh']=r['modeled_control_Wh']+r['active_cooling_Wh']
    r['liquid_scope_unsupported_hours']=int((frame.water_supported==0).sum());r['control_power_unsupported_hours']=int((frame.unserved_Wh>1e-8).sum())
    r['outside_model_hours']=int((~frame.presentation_supported.astype(bool)).sum());r['outside_model_percent']=100*r['outside_model_hours']/len(frame) if len(frame) else None
    r['collected_condensation_L_m2_h']=r['collected_condensation_L']/(c['area_m2']*len(frame)) if len(frame) else None
    r['collected_rain_L_m2_h']=r['collected_rain_L']/(c['area_m2']*len(frame)) if len(frame) else None
    for name in ['drive_utilization','hold_utilization','tilting_utilization','radial_utilization']:
        r[name+'_max']=float(frame[name].max()) if len(frame) else None
    r['source_mass_residual_max_L']=float(frame.origin_balance_max_L.abs().max()) if len(frame) else None
    r['battery_residual_max_Wh']=float(frame.battery_residual_Wh.abs().max()) if len(frame) else None
    r['water_balance_residual_L']=r['rain_input_L']+r['condensed_L']-r['evaporated_L']-r['drain_L']-r['uncollected_departure_L']-sum(r['stored_'+s+'_L'] for s in ['rain','condensation','initial'])
    floor=1e-4*c['area_m2']*len(frame);r['water_ratio_resolution_floor_L']=floor;r['energy_water_ratios']={}
    for origin in ['rain','condensation']:
        volume=r['collected_'+origin+'_L'];resolved=volume>max(floor,1e-12)
        r['energy_water_ratios'][origin]={'collected_L':volume,'denominator_resolved_at_design_floor':resolved,
          'whole_system_control_burden_Wh_per_L':r['modeled_control_Wh']/volume if resolved else None,
          'nominal_arithmetic_control_burden_Wh_per_L':r['modeled_control_Wh']/volume if volume>1e-12 else None,
          'active_cooling_Wh_per_L':r['active_cooling_Wh']/volume if resolved else None,
          'interpretation':'shared system burden; numerators overlap across rain/condensation; not causal process-specific SEC'}
    return r

def calculate():
    f=pd.read_csv(OUT/'hourly.csv.gz');f['presentation_supported']=(f.water_supported==1)&(f.unserved_Wh<=1e-8)
    p=json.loads((ROOT/'config/presentation/protocol.json').read_text());kinds=p['controllers'];h=p['episode_hours'];prefix={};complete=[]
    for tag,z in f.groupby('group'):
        pivot=z.pivot_table(index='hour',columns='controller',values='presentation_supported',aggfunc='min').reindex(index=np.arange(1,h+1),columns=kinds).fillna(0)
        common=(pivot.min(axis=1)>0).to_numpy();bad=np.flatnonzero(~common);prefix[tag]=int(bad[0]) if len(bad) else h
        if prefix[tag]==h:complete.append(tag)
    f['common_valid_prefix']=[row.hour<=prefix[row.group] for row in f.itertuples()];f['whole_episode_jointly_supported']=f.group.isin(complete)
    f.to_csv(OUT/'hourly-audited.csv.gz',index=False,compression='gzip')
    datasets={'all_hours_diagnostic':f,'common_prefix':f[f.common_valid_prefix],'complete_supported_episodes':f[f.whole_episode_jointly_supported]}
    summary={name:{kind:summarize(z[z.controller==kind]) for kind in kinds} for name,z in datasets.items()};city=[]
    for city_id,z in f.groupby('city'):
        for kind in kinds:
            raw=z[z.controller==kind];eligible=raw[raw.common_valid_prefix];s=summarize(eligible)
            city.append({'city':city_id,'controller':kind,'scored_prefix_hours':len(eligible),'window_hours':len(raw),
              'common_prefix_excluded_percent':100*(1-len(eligible)/len(raw)),'raw_outside_model_percent':100*float((~raw.presentation_supported).mean()),**s})
    # Resample city clusters, paired policy differences. This is not physical uncertainty.
    means=pd.DataFrame(city);intervals={};rng=np.random.default_rng(956)
    for origin in ['rain','condensation']:
        col='collected_'+origin+'_L_m2_h';pivot=means.pivot(index='city',columns='controller',values=col).dropna()
        for baseline in ['fixed','emitter','rule','mpc']:
            values=(pivot.ppo-pivot[baseline]).to_numpy()
            if len(values):
                samples=rng.choice(values,(5000,len(values)),replace=True).mean(1);intervals[f'ppo_minus_{baseline}_{origin}']={'mean_difference_L_m2_h':float(values.mean()),'city_cluster_resampling_95_interval_L_m2_h':np.quantile(samples,[.025,.975]).tolist(),'cities':len(values),'product_improvement_claim':False}
    scenarios=json.loads((OUT/'scenarios.json').read_text());ranges=[]
    for cid in sorted({r['city'] for r in scenarios['runs']}):
        z=[r for r in scenarios['runs'] if r['city']==cid];ranges.append({'city':cid,'controller':'rule','hours_per_run':z[0]['hours'],
          **{k:[min(r[k] for r in z),max(r[k] for r in z)] for k in ['collected_rain_L','collected_condensation_L','pv_gross_Wh','motor_Wh']},'runs_with_unsupported_hours':sum(r['unsupported_hours']>0 for r in z),'type':'scenario envelope, not confidence interval'})
    cfg=model_config();assembly=compile_assembly(cfg);wind=[]
    for film in [0,.5,5.]:
        for speed in [0,5,10,12,15,20,25]:wind.append({'wind_mean_m_s':speed,'film_kg':film,**load_envelope(cfg,assembly,speed,film_mass=film)})
    pd.DataFrame(wind).to_csv(OUT/'wind-screening.csv',index=False)
    eps=json.loads((OUT/'episodes.json').read_text())['episodes'];coverage=[]
    for cid in p['cities']:
        z=[e for e in eps if e['city']==cid];coverage.append({'city':cid,'annual_weather_hours':z[0]['annual_weather_hours'],'annual_weather_outside_preselection_percent':z[0]['annual_weather_outside_preselection_percent'],
          'simulated_hours':sum(e['hours'] for e in z),'scored_common_prefix_hours':sum(prefix[e['group']] for e in z),'windows':[{k:e[k] for k in ['start','end','hours','start_offset_from_target_hours']} for e in z]})
    result={'version':'presentation-F1','summaries':summary,'by_city':city,'prefix_hours_by_group':prefix,'complete_jointly_supported_groups':complete,
      'weather_coverage':coverage,'conditional_city_cluster_intervals':intervals,'scenario_envelopes':ranges,
      'wind_mean_limit_dry_screening_m_s':allowable_wind(cfg,assembly),'active_cooling_added':False,'online_compute_energy_measured':False,
      'total_physical_uncertainty_interval':None,'total_physical_uncertainty_reason':'unidentified materials, wet optics, turbulent rain/transfer and physical validation remain unresolved; scenario ranges are not calibrated priors',
      'annual_yield_or_climate_ranking_claim':False,'RL_improvement_claim_allowed':False,'independent_numerical_weather_holdout':True,'independent_physical_validation':False,
      'useful_PV_definition':'new PV delivered to modeled controls plus net newly-PV-origin battery stock; excludes starting battery energy, conversion/battery losses and unused surplus',
      'energy_to_water_note':'reported shared control burdens use the same numerator; do not sum origin-specific burdens or interpret rain ratio as atmospheric condensation SEC',
      'domain_note':'known liquid-water computational scope only; physical yield remains unqualified, even on common supported prefixes'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2));pd.DataFrame(city).drop(columns=['energy_water_ratios']).to_csv(OUT/'city-metrics.csv',index=False)
    ratios=[]
    for r in city:
        for origin,v in r['energy_water_ratios'].items():ratios.append({'city':r['city'],'controller':r['controller'],'origin':origin,'hours':r['hours'],'modeled_control_Wh':r['modeled_control_Wh'],**v})
    pd.DataFrame(ratios).to_csv(OUT/'energy-water-by-city.csv',index=False)
    return result
