"""Temporal discretization and mixed-source LW-label sensitivity diagnostics.

This is NOT a complete uncertainty analysis. No extrapolation to annual yield.
"""
import argparse
import pandas as pd
from benchmark import run_case
from control import causal_rule
from weather import ROOT,model_config,load_weather


def main():
    p=argparse.ArgumentParser();p.add_argument('--cities',nargs='+',default=['rasht','singapore','mumbai','dubai','reykjavik','la-paz'])
    p.add_argument('--year',type=int,default=2024);p.add_argument('--month',type=int,default=7)
    a=p.parse_args();rows=[]
    for city in a.cities:
        for shift,dt in [(0,60),(0,30),(0,15),(-1,30),(1,30)]:
            cfg=model_config();cfg['integration_dt_s']=dt
            frame=load_weather(city,a.year,cfg,shift)
            start=frame.index.get_loc(pd.Timestamp(f'{a.year}-{a.month:02d}-15T00:00Z'))
            summary,_=run_case(frame,start,causal_rule,cfg,'causal_rule')
            rows.append({**summary,'city':city,'year':a.year,'month':a.month,'dt_s':dt,'lw_shift_h':shift})
        print('Sensitivity:',city,flush=True)
    output=ROOT/'results'/f'sensitivity-{a.year}-{a.month:02d}.csv'
    pd.DataFrame(rows).to_csv(output,index=False)
    print('Saved',output)


if __name__=='__main__':main()
