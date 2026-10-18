"""Conservative Richardson diagnostics: do not assign GCI to oscillating data."""
import math
import numpy as np

def richardson_three(coarse,medium,fine,r=2.,absolute_floor=1e-12):
    a,b,c=map(float,[coarse,medium,fine]);d1=a-b;d2=b-c
    if r<=1:raise ValueError('Refinement ratio must exceed one')
    if max(abs(d1),abs(d2))<absolute_floor:return {'status':'below_absolute_resolution','absolute_floor':absolute_floor,'GCI':None}
    if d1*d2<=0 or abs(d2)<absolute_floor:return {'status':'oscillatory_or_degenerate','GCI':None}
    p=math.log(abs(d1/d2))/math.log(r)
    if p<=0:return {'status':'not_converging','observed_order':p,'GCI':None}
    correction=(c-b)/(r**p-1);band=1.25*abs(c-b)/(r**p-1)
    return {'status':'monotone_three_level_estimate_asymptotic_not_proven','observed_order':p,'extrapolated':c+correction,
      'GCI_absolute':band,'GCI_relative':band/abs(c) if abs(c)>absolute_floor else None,'safety_factor':1.25,
      'limitation':'three levels alone do not prove asymptotic range; refinement scheme and iterative convergence must also hold'}
