"""Local adaptive particle/source study with explicit finite compute budgets."""
import argparse,json,math
from pathlib import Path
from assembly import ROOT
from stageb.rain import simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=ROOT/'results/stage-b/cfd/reference-fine')
    p.add_argument('--target',type=float,default=.05,help='Relative sampling 95%% halfwidth and separate bundle tolerance')
    p.add_argument('--initial-samples',type=int,default=8192);p.add_argument('--max-samples',type=int,default=131072)
    p.add_argument('--bundles',nargs='+',type=float,default=[6,9,12]);p.add_argument('--dt',type=float,default=.025);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    if not 0<a.target<1 or len(a.bundles)<2 or a.initial_samples>a.max_samples:raise ValueError('Invalid convergence budget')
    rows=[]
    for bundle in a.bundles:
        n=a.initial_samples
        while True:
            r=simulate(a.case,n,5.,a.seed,a.dt,bundle,.5)
            half=r['sampling_relative95_halfwidth'];enough=half is not None and half<=a.target and min(x['panel_hits'] for x in r['bins'])>=10
            if enough or n>=a.max_samples:break
            factor=2 if half is None else max(2,(half/a.target)**2)
            n=min(a.max_samples,2**math.ceil(math.log2(n*factor)))
        rows.append({'bundle_m':bundle,'samples_per_size':n,'specific_catch_ratio':r['total_specific_catch_ratio'],
            'sampling_standard_error':r['sampling_only_standard_error_catch_ratio'],'sampling_relative95_halfwidth':half,'sampling_gate':enough})
    diffs=[]
    for left,right in zip(rows,rows[1:]):
        denom=max(abs(right['specific_catch_ratio']),1e-20)
        change=abs(left['specific_catch_ratio']-right['specific_catch_ratio'])/denom
        margin=1.96*math.hypot(left['sampling_standard_error'],right['sampling_standard_error'])/denom
        diffs.append({'relative_change':change,'conservative_MC_margin':margin,'gate':change+margin<=a.target})
    result={'case':str(a.case),'target':a.target,'budget_max_samples_per_size':a.max_samples,'runs':rows,'source_comparisons':diffs,
       'sampling_and_source_gate':all(x['sampling_gate'] for x in rows) and all(x['gate'] for x in diffs),
       'remaining_gates':['CFD grid convergence','physical splash/retention and DSD uncertainty','independent experimental validation'],
       'annual_yield_or_RL_training_authorized':False}
    path=ROOT/'results/stage-b/rain/adaptive-study.json';path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
