"""Deterministic one-at-a-time screening, NOT confidence intervals/Sobol."""
import argparse
import pandas as pd
from weather import ROOT,load_weather,model_config
from benchmark import run_case
from control import causal_rule

SCENARIOS={
 'reference':{},'emitter_IR_tail_0':{'dew_unknown_tail_epsilon':0.},
 'emitter_IR_tail_1':{'dew_unknown_tail_epsilon':1.},
 'sky_min_absorption':{'sky_spectral_policy':'low'},'sky_max_absorption':{'sky_spectral_policy':'high'},
 'coating_condition_0':{'coating_condition':0.},'jump_fraction_0':{'jump_fraction':0.},
 'collector_0':{'collector_efficiency':0.},'port_Cp_0p02':{'port_delta_Cp':.02},
 'port_Cp_0p6':{'port_delta_Cp':.6},'bridge_x10':{'bridge_multiplier':10.},
 'convection_half':{'external_h_multiplier':.5},'convection_twice':{'external_h_multiplier':2.},
 'motor_catalogue_30W':{'motor_energy_policy':'catalogue_nominal'},
 'gust_factor_2':{'wind_gust_factor':2.},'PV_absorptance_0p8':{'pv_effective_absorptance':.8},
 'PV_absorptance_0p96':{'pv_effective_absorptance':.96},'PV_capacity_half':{'pv_capacity_multiplier':.5},
 'PV_capacity_1p5':{'pv_capacity_multiplier':1.5}}
SCENARIOS.update({'footprints_1e3':{'droplet_footprints_m2':1e3},'footprints_1e8':{'droplet_footprints_m2':1e8}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--cities',nargs='+',default=['rasht','singapore','dubai','la-paz'])
    p.add_argument('--year',type=int,default=2024);p.add_argument('--month',type=int,default=7)
    p.add_argument('--hours',type=int,default=48);p.add_argument('--warmup',type=int,default=24)
    p.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS));a=p.parse_args()
    if not 0<=a.warmup<a.hours:raise ValueError('Invalid scoring window')
    rows=[]
    for city in a.cities:
        for name in a.scenarios:
            c=model_config();c.update(SCENARIOS[name]);f=load_weather(city,a.year,c)
            start=f.index.get_loc(pd.Timestamp(f'{a.year}-{a.month:02d}-15T00:00Z'))
            summary,_=run_case(f,start,causal_rule,c,'causal_rule',a.hours,a.warmup)
            rows.append({'city':city,'year':a.year,'month':a.month,'scenario':name,**summary})
            print('Conditional sensitivity:',city,name,flush=True)
    out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f'parameter-scenarios-{a.year}-{a.month:02d}-h{a.hours}.csv',index=False)


if __name__=='__main__':main()
