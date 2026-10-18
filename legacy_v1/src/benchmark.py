"""Seasonal diagnostics only: 4 x (24 h warm-up + 72 h scored), no extrapolation."""
import argparse
import json
import numpy as np
import pandas as pd
from control import fixed_pv,causal_rule
from physics import PanelModel
from weather import ROOT,cities,load_weather,model_config


def run_case(frame,start,policy,config=None,policy_name='rule'):
    model=PanelModel(config);model.reset(frame.iloc[start].air_C)
    records=[]
    for offset in range(1,97):
        # This controller sees row t; the solver advances through interval t+1.
        action=policy(frame.iloc[start+offset-1],model)
        row=model.step_hour(frame.iloc[start+offset],action)
        records.append({'time':str(frame.index[start+offset]),'scored':offset>24,**row})
    r=pd.DataFrame(records);score=r[r.scored]
    invalid=bool((r.water_supported==0).any())
    summary={'policy':policy_name,'scored_hours':len(score),'warmup_hours':24,
             'unsupported_hours_full_window':int((r.water_supported==0).sum()),
             'window_water_supported':not invalid,'status':'UNVALIDATED_REDUCED_ORDER_DIAGNOSTIC',
             # A cold hour invalidates the entire window because subsequent film
             # and thermal states inherit omitted freezing physics.
             'drain_L':None if invalid else float(score.drain_L.sum()),
             'pv_gross_Wh':None if invalid else float(score.pv_gross_Wh.sum()),
             'unserved_Wh':None if invalid else float(score.unserved_Wh.sum()),
             'motor_Wh':float(score.motor_Wh.sum()),'end_soc_Wh':float(score.battery_Wh.iloc[-1]),
             'max_mass_residual_kg':float(r.mass_residual_kg.abs().max()),
             'max_battery_residual_Wh':float(r.battery_residual_Wh.abs().max()),
             'max_thermal_residual_J':float(r.max_thermal_residual_J.max())}
    return summary,r


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cities',nargs='+',default=[c['id'] for c in cities()]);p.add_argument('--year',type=int,default=2024)
    p.add_argument('--dt',type=float,default=60)
    p.add_argument('--lw-shift',type=int,default=0,choices=[-1,0,1])
    a=p.parse_args();cfg=model_config();cfg['integration_dt_s']=a.dt
    rows=[];dest=ROOT/'results/seasonal';dest.mkdir(exist_ok=True)
    for city in a.cities:
        frame=load_weather(city,a.year,cfg,a.lw_shift)
        for month in (1,4,7,10):
            start=frame.index.get_loc(pd.Timestamp(f'{a.year}-{month:02d}-15T00:00Z'))
            for name,policy in [('fixed_pv',fixed_pv),('causal_rule',causal_rule)]:
                summary,trace=run_case(frame,start,policy,cfg,name)
                summary.update(city=city,month=month,year=a.year,dt_s=a.dt,lw_shift_h=a.lw_shift)
                rows.append(summary)
                trace.to_csv(dest/f'{city}-{month:02d}-{name}-dt{a.dt:g}-lw{a.lw_shift}.csv.gz',index=False,compression='gzip')
        print('Seasonal diagnostics:',city,flush=True)
    out=ROOT/'results'/f'seasonal-summary-dt{a.dt:g}-lw{a.lw_shift}.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Saved',out)


if __name__=='__main__': main()
