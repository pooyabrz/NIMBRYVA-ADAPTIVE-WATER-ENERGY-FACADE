"""B3: conservative resolved film transport with water/ice enthalpy.

One-dimensional surface control volumes; equilibrium pure-water phase change.
Snow enters as WATER-EQUIVALENT mass, never as depth without a density model.
This smooth-film reference does not identify CNF droplet nucleation/ageing.
"""
import argparse,json,math
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erf
from assembly import ROOT
from core import SIGMA

LF=333500.;CI=2100.;CW=4180.;CV=1860.;LV0=2501000.;RHO_W=1000.;RHO_I=917.


def saturation_pa(t):
    """Vector ASHRAE ice/water saturation; same coefficients as core.py."""
    t=np.asarray(t,dtype=float);k=t+273.15
    ice=-5674.5359/k+6.3925247-.009677843*k+6.2215701e-7*k*k+2.0747825e-9*k**3-9.484024e-13*k**4+4.1635019*np.log(k)
    water=-5800.2206/k+1.3914993-.048640239*k+4.1764768e-5*k*k-1.4452093e-8*k**3+6.5459673*np.log(k)
    return np.exp(np.where(t<=.01,ice,water))


def mass_fraction_vapour(t,rh,p):
    e=rh*saturation_pa(t)
    if np.any(e>=p) or np.any(e<0):raise ValueError('Invalid moist-air partial pressure')
    return .621945*e/(p-.378055*e)


def decode_energy(energy,mass,plate_C=850.3):
    energy=np.asarray(energy);mass=np.asarray(mass)
    negative=energy<0;positive=energy>mass*LF
    t=np.where(negative,energy/(plate_C+mass*CI),np.where(positive,(energy-mass*LF)/(plate_C+mass*CW),0.))
    liquid=np.where(negative,0.,np.where(positive,mass,np.maximum(0,energy)/LF))
    return t,np.minimum(mass,liquid),np.maximum(0,mass-liquid)


def encode_energy(t,mass,plate_C=850.3,liquid_fraction=None):
    t=np.asarray(t);mass=np.asarray(mass)
    if liquid_fraction is not None:return plate_C*t+mass*(CI*np.minimum(t,0)+LF*liquid_fraction+CW*np.maximum(t,0))
    return plate_C*t+mass*np.where(t<0,CI*t,LF+CW*t)


class SurfaceFilm:
    def __init__(self,n=32,length=1.762,width=1.134,angle_deg=50.,initial_C=20.,initial_liquid_kg_m2=0.,initial_ice_kg_m2=0.,plate_C=850.3):
        if n<2 or length<=0 or width<=0:raise ValueError('Invalid surface mesh')
        self.n=n;self.length=length;self.width=width;self.dx=length/n;self.area_cell=self.dx*width;self.angle=angle_deg;self.plate_C=plate_C
        self.mass=np.full(n,initial_liquid_kg_m2+initial_ice_kg_m2,dtype=float)
        if initial_C<0 and initial_liquid_kg_m2:raise ValueError('Specify initial ice below freezing')
        if initial_C>0 and initial_ice_kg_m2:raise ValueError('Specify initial liquid above freezing')
        if initial_C==0:self.energy=np.full(n,initial_liquid_kg_m2*LF)
        else:self.energy=encode_energy(np.full(n,initial_C),self.mass,plate_C)
        self.snow=np.zeros(n);self.elapsed=0.;self.budget={k:0. for k in ['rain_kg','snow_swe_kg','vapour_in_kg','vapour_out_kg','drain_kg','energy_environment_J','energy_material_in_J','energy_vapour_J','drain_energy_J']}
        self.mass0=float(self.mass.sum()*self.area_cell);self.energy0=float(self.energy.sum()*self.area_cell)
        self.max_dry_equivalent_thickness=0.;self.substeps=0

    def advance(self,duration_s,air_C=20.,rh=.8,pressure=101325.,solar_W_m2=0.,lw_in_W_m2=350.,h_W_m2_K=5.,hm_m_s=.005,
                rain_kg_m2_s=0.,snow_swe_kg_m2_s=0.,rain_C=None,snow_C=None,back_C=20.,back_G_W_m2_K=.5,alpha=.04,epsilon=.93,
                shear_Pa=0.,max_dt=10.,snow_density=200.,mass_transfer_model='stefan'):
        rain_kg_m2_s=np.broadcast_to(np.asarray(rain_kg_m2_s,dtype=float),(self.n,))
        snow_swe_kg_m2_s=np.broadcast_to(np.asarray(snow_swe_kg_m2_s,dtype=float),(self.n,))
        if not 0<=rh<=1 or np.any(rain_kg_m2_s<0) or np.any(snow_swe_kg_m2_s<0) or min(h_W_m2_K,hm_m_s)<0:raise ValueError('Invalid forcing')
        if snow_density<=0:raise ValueError('Snow density must be positive')
        if mass_transfer_model not in ['stefan','linear_mass_fraction']:raise ValueError('Unknown vapour transfer model')
        rain_C=air_C if rain_C is None else rain_C;snow_C=min(0,air_C) if snow_C is None else snow_C
        if rain_C<0 and np.any(rain_kg_m2_s>0):raise ValueError('Supercooled rain/freezing-rain kinetics are not this equilibrium reference')
        ya=float(mass_fraction_vapour(air_C,rh,pressure));rho=pressure/(287.05*(air_C+273.15))
        total_area=self.n*self.area_cell;elapsed=0.
        while elapsed<duration_s-1e-9:
            t,liquid,ice=decode_energy(self.energy,self.mass,self.plate_C)
            delta=liquid/RHO_W
            qmass=RHO_W**2*9.80665*math.sin(math.radians(self.angle))*delta**3/(3*.001)+RHO_W*shear_Pa*delta**2/(2*.001)
            stable=np.divide(.4*self.dx*liquid,np.abs(qmass),out=np.full(self.n,np.inf),where=np.abs(qmass)>1e-20)
            dt=min(max_dt,duration_s-elapsed,float(stable.min()))
            if dt<1e-6:raise FloatingPointError('Film transport CFL is too restrictive; refine/reference regime')
            faces=np.zeros(self.n+1)
            if np.all(qmass>=0):faces[1:]=qmass
            elif np.all(qmass<=0):faces[:-1]=qmass
            else:
                faces[1:-1]=np.maximum(qmass[:-1],0)+np.minimum(qmass[1:],0)
                faces[0]=min(qmass[0],0);faces[-1]=max(qmass[-1],0)
            ef=np.zeros_like(faces);hliq=LF+CW*t
            ef[1:-1]=np.maximum(qmass[:-1],0)*hliq[:-1]+np.minimum(qmass[1:],0)*hliq[1:]
            ef[0]=faces[0]*hliq[0];ef[-1]=faces[-1]*hliq[-1]
            self.mass-=dt*np.diff(faces)/self.dx;self.energy-=dt*np.diff(ef)/self.dx
            drain=(faces[-1]-faces[0])*dt*self.width;de=(ef[-1]-ef[0])*dt*self.width
            self.budget['drain_kg']+=float(drain);self.budget['drain_energy_J']+=float(de)
            if self.mass.min()<-1e-10:raise FloatingPointError('Negative liquid inventory from transport')
            self.mass=np.maximum(0,self.mass)
            m0=self.mass.copy();e0=self.energy.copy();incoming=(rain_kg_m2_s+snow_swe_kg_m2_s)*dt
            material=rain_kg_m2_s*(LF+CW*rain_C)+snow_swe_kg_m2_s*CI*snow_C
            def evaluate(temp):
                es=saturation_pa(temp);ys=.621945*es/(pressure-.378055*es)
                if np.any(ys>=1):raise ValueError('Reference above dilute moist-air range')
                # Exact stagnant gas-film Stefan correction in mass fractions.
                j=rho*hm_m_s*(np.log((1-ys)/(1-ya)) if mass_transfer_model=='stefan' else ya-ys)
                j=np.maximum(j,-(m0+incoming)/dt)
                mass=m0+incoming+j*dt
                hv=LF+LV0+CV*np.where(j>=0,air_C,temp)
                q=alpha*solar_W_m2+h_W_m2_K*(air_C-temp)+epsilon*(lw_in_W_m2-SIGMA*(temp+273.15)**4)+back_G_W_m2_K*(back_C-temp)
                rhs=e0+dt*(q+material+j*hv)
                return mass,rhs,j,q,hv
            zeros=np.zeros(self.n);mz,ez,*_=evaluate(zeros)
            # Enthalpy discontinuity at T=0 is represented exactly: no invented
            # mushy heat capacity or automatic melting at the freezing point.
            freeze=ez<0;melt=ez>mz*LF;active=freeze|melt
            lo=np.where(freeze,-80.,0.);hi=np.where(freeze,0.,100.)
            for _ in range(42):
                mid=(lo+hi)/2;mass,rhs,*_=evaluate(mid)
                target=encode_energy(mid,mass,self.plate_C)
                residual=target-rhs
                hi=np.where(active&(residual>0),mid,hi);lo=np.where(active&(residual<=0),mid,lo)
            temp=np.where(active,(lo+hi)/2,0.)
            mass,energy,j,q,hv=evaluate(temp)
            self.mass=mass;self.energy=energy
            self.snow+=snow_swe_kg_m2_s*dt
            _,liq,solid=decode_energy(energy,mass,self.plate_C);self.snow=np.minimum(self.snow,solid)
            self.max_dry_equivalent_thickness=max(self.max_dry_equivalent_thickness,float(np.max(self.snow/snow_density+(solid-self.snow)/RHO_I+liq/RHO_W)))
            for key,value in [('rain_kg',rain_kg_m2_s.sum()*dt*self.area_cell),('snow_swe_kg',snow_swe_kg_m2_s.sum()*dt*self.area_cell),
                ('vapour_in_kg',np.maximum(j,0).sum()*dt*self.area_cell),('vapour_out_kg',np.maximum(-j,0).sum()*dt*self.area_cell),
                ('energy_environment_J',q.sum()*dt*self.area_cell),('energy_material_in_J',material.sum()*dt*self.area_cell),('energy_vapour_J',(j*hv).sum()*dt*self.area_cell)]:
                self.budget[key]+=float(value)
            elapsed+=dt;self.elapsed+=dt;self.substeps+=1
        return self.summary()

    def summary(self):
        t,liq,ice=decode_energy(self.energy,self.mass,self.plate_C);b=self.budget
        mass_res=self.mass0+b['rain_kg']+b['snow_swe_kg']+b['vapour_in_kg']-b['vapour_out_kg']-b['drain_kg']-self.mass.sum()*self.area_cell
        energy_res=self.energy0+b['energy_environment_J']+b['energy_material_in_J']+b['energy_vapour_J']-b['drain_energy_J']-self.energy.sum()*self.area_cell
        return {'elapsed_s':self.elapsed,'cells':self.n,'temperature_mean_C':float(t.mean()),'temperature_min_C':float(t.min()),
          'liquid_kg':float(liq.sum()*self.area_cell),'ice_plus_snow_kg':float(ice.sum()*self.area_cell),
          'unmelted_snow_tracer_kg':float(self.snow.sum()*self.area_cell),'mass_residual_kg':float(mass_res),'energy_residual_J':float(energy_res),
          'maximum_layer_thickness_m':self.max_dry_equivalent_thickness,'substeps':self.substeps,**b,
          'thin_equilibrium_layer_approximation':True,'thick_snow_thermal_gradient_unresolved':self.max_dry_equivalent_thickness>.001,
          'snow_aerodynamic_capture_and_adhesion_validated':False,'CNF_droplet_physics_validated':False}


def stefan_reference(n=100,dt=1.,end=600.,mushy_K=.1):
    """1D enthalpy conduction vs the Neumann one-phase analytic solution.

    Artificial verification material uses equal constant properties in phases;
    it is not claimed to be a measured ice slab. Mushy interval [0,dT].
    """
    rho=917.;cp=2100.;k=2.2;length=.1;hot=10.;dx=length/n;g=k/dx**2;t=np.zeros(n)
    def enthalpy(t):return rho*(cp*t+LF*np.clip(t/mushy_K,0,1))
    def capacity(t):return rho*(cp+LF/mushy_K*((t>0)&(t<mushy_K)))
    boundary=np.zeros(n);boundary[0]=2*g*hot
    diag=np.full(n,2*g);diag[[0,-1]]=3*g
    energy_in=0.;initial=0.;elapsed=0.
    while elapsed<end-1e-10:
        dtime=min(dt,end-elapsed);old=t.copy();hold=enthalpy(old)
        def residual(x):
            diff=diag*x-boundary;diff[1:]-=g*x[:-1];diff[:-1]-=g*x[1:]
            return (enthalpy(x)-hold)/dtime+diff
        for iteration in range(100):
            res=residual(t);norm=np.max(np.abs(res))
            if norm<1e-5:break
            band=np.zeros((3,n));band[0,1:]=-g;band[2,:-1]=-g;band[1]=capacity(t)/dtime+diag
            delta=solve_banded((1,1),band,-res)
            step=1.
            while step>1e-8 and np.max(np.abs(residual(t+step*delta)))>=norm:step*=.5
            t+=step*delta
        else:raise FloatingPointError('Stefan enthalpy Newton failed')
        energy_in+=(2*k/dx*(hot-t[0])+2*k/dx*(0-t[-1]))*dtime
        elapsed+=dtime
    fraction=np.clip(t/mushy_K,0,1);front=float(fraction.sum()*dx)
    ste=cp*hot/LF;lam=brentq(lambda z:z*math.exp(z*z)*erf(z)-ste/math.sqrt(math.pi),1e-8,2)
    exact=2*lam*math.sqrt(k/(rho*cp)*end)
    return {'n':n,'dt_s':dt,'mushy_K':mushy_K,'front_m':front,'analytic_front_m':exact,
            'relative_front_error':abs(front-exact)/exact,'energy_residual_J_m2':energy_in-float(enthalpy(t).sum()*dx),
            'scope':'one-phase Neumann benchmark, equal phase properties, temperature-smoothed enthalpy'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stefan',action='store_true');p.add_argument('--hours',type=float,default=2);a=p.parse_args()
    out=ROOT/'results/stage-b/surface';out.mkdir(parents=True,exist_ok=True)
    if a.stefan:
        rows=[stefan_reference(n,dt) for n,dt in [(50,2),(100,1),(200,.5)]]
        (out/'stefan-convergence.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2));return
    cases=[]
    for name,temp,rh,solar,lw,snow in [('humid-night',20,.9,0,300,0),('cold-snow',-8,.8,0,220,.00002),('melting',8,.6,600,340,0)]:
        model=SurfaceFilm(n=32,initial_C=min(temp,0) if name!='humid-night' else 20,initial_ice_kg_m2=.05 if name!='humid-night' else 0)
        row=model.advance(a.hours*3600,air_C=temp,rh=rh,solar_W_m2=solar,lw_in_W_m2=lw,snow_swe_kg_m2_s=snow,back_C=temp,max_dt=10)
        cases.append({'case':name,**row});print(name,row['mass_residual_kg'],row['energy_residual_J'],flush=True)
    (out/'surface-reference-cases.json').write_text(json.dumps(cases,indent=2))


if __name__=='__main__':main()
