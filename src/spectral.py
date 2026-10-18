"""Wavelength-resolved emitter, reference solar shape, explicit sky uncertainty.

Raw ETH Figure 1C contains 13 absorptances >1 above 23 um. They are retained
for audit, never silently clipped. Default trusted range .25..20 um.
Solar ASTM G173 shape and scaled Planck sky are SCREENING assumptions.
Broadband forcing cannot identify a real atmospheric spectrum.
"""
from functools import lru_cache
import json
import numpy as np
from assembly import ROOT
from core import SIGMA


def planck_exitance_um(wavelength_um,temp_C):
    lam=np.asarray(wavelength_um,dtype=float)*1e-6;t=np.asarray(temp_C)+273.15
    if np.any(lam<=0) or np.any(t<=0):raise ValueError('Invalid Planck arguments')
    x=6.62607015e-34*299792458/(lam*1.380649e-23*t)
    return (2*np.pi*6.62607015e-34*299792458**2/lam**5)/np.expm1(np.minimum(x,700))*1e-6


def trapezoid_weights(x):
    x=np.asarray(x);d=np.diff(x)
    if len(x)<3 or np.any(d<=0):raise ValueError('Need increasing spectral grid')
    return np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]


def raw_optics():
    return np.genfromtxt(ROOT/'data/materials/haechler-fig1c-raw.csv',delimiter=',',names=True)


def validate_optics(lam,alpha):
    if np.any(~np.isfinite(alpha)) or np.any((alpha<0)|(alpha>1)):
        raise ValueError('Unphysical raw optical data: preserve/flag, do not silently clip')
    if np.any(np.diff(lam)<=0):raise ValueError('Nonmonotone optical wavelengths')


class SpectralRadiation:
    def __init__(self,cfg):
        self.cfg=cfg;r=raw_optics();ok=r['trusted_0p25_20um'].astype(bool)
        self.raw=r;self.x=r['wavelength_um'][ok];self.a=r['absorptance_raw'][ok]
        validate_optics(self.x,self.a)
        # Explicit left/right limits avoid grid-dependent smearing at the
        # transition from measured data to the unknown spectral tail.
        edge=self.x[-1]
        self.lam=np.unique(np.r_[np.geomspace(2.5,1000,cfg['spectral_points']),self.x[self.x>=2.5],np.nextafter(edge,np.inf)])
        tail=cfg['dew_unknown_tail_epsilon']
        self.epsilon=np.interp(self.lam,self.x,self.a,left=tail,right=tail)*cfg['dew_angular_multiplier']
        if np.any((self.epsilon<0)|(self.epsilon>1)):raise ValueError('Unphysical emissivity scenario')
        self.weights=trapezoid_weights(self.lam)
        self.temps=np.linspace(-60,140,2001)
        m=planck_exitance_um(self.lam[None,:],self.temps[:,None])
        self.emission=m@(self.weights*self.epsilon)
        # Unresolved tail above 1000 um is reported and is not declared zero
        # in the strict evidence gate. It is tiny at ambient temperatures.
        g=np.genfromtxt(ROOT/'data/materials/ASTMG173.csv',delimiter=',',skip_header=2)
        self.sw_lam=g[:,0]/1000;self.sw_global=g[:,2]*1000
        sw_alpha=np.interp(self.sw_lam,self.x,self.a)
        self.solar_alpha=float(np.trapezoid(sw_alpha*self.sw_global,self.sw_lam)/np.trapezoid(self.sw_global,self.sw_lam))
        uv=(self.sw_lam<=.4)
        self.uv_fraction=float(np.trapezoid(self.sw_global[uv],self.sw_lam[uv])/np.trapezoid(self.sw_global,self.sw_lam))

    def exitance(self,t):
        if not -60<=t<=140:raise ValueError('Emission lookup outside verified temperature grid')
        return float(np.interp(t,self.temps,self.emission))

    def derivative(self,t):return (self.exitance(t+.01)-self.exitance(t-.01))/.02

    def sky(self,ta,lw):
        """Min/max absorption for 0<=E_lambda<=M_lambda(Tair+cap_offset).

        Greedy linear allocation is the exact discrete linear-program bound
        for the supplied broadband integral and assumed blackbody ceiling.
        This is an identifiability envelope, NOT atmospheric reconstruction.
        """
        if not np.isfinite(lw) or lw<0:raise ValueError('Invalid broadband longwave')
        cap=planck_exitance_um(self.lam,ta+self.cfg['sky_temperature_cap_offset_K'])*self.weights
        if lw>cap.sum()+1e-6:
            raise ValueError('LW exceeds spectral-envelope blackbody ceiling; adjust declared boundary scenario')
        def fill(order):
            remaining=lw;absorbed=0.
            for i in order:
                take=min(remaining,cap[i]);absorbed+=take*self.epsilon[i];remaining-=take
                if remaining<=1e-10:break
            return absorbed
        order=np.argsort(self.epsilon)
        lo=fill(order);hi=fill(order[::-1])
        nominal=lw*self.exitance(ta)/float(np.dot(planck_exitance_um(self.lam,ta),self.weights))
        return {'nominal':nominal,'low':lo,'high':hi,'shape_identified':False}

    def audit(self):
        t=25.;total=SIGMA*(t+273.15)**4
        full=float(np.dot(planck_exitance_um(self.lam,t),self.weights))
        trusted=self.lam<=20
        known=float(np.dot(planck_exitance_um(self.lam[trusted],t),trapezoid_weights(self.lam[trusted])))
        return {'raw_samples':len(self.raw),'raw_above_one':int((self.raw['absorptance_raw']>1).sum()),
                'trusted_range_um':[float(self.x[0]),float(self.x[-1])],
                'thermal_grid_range_um':[2.5,1000.], 'spectral_grid_points':len(self.lam),
                'thermal_integral_vs_sigma_relative_error':abs(full-total)/total,
                'blackbody_fraction_above_trusted_20um':1-known/total,
                'reference_G173_dew_alpha':self.solar_alpha,'reference_G173_UV_fraction':self.uv_fraction,
                'angular_emissivity_measured':False,'real_city_hourly_spectrum_available':False,
                'optical_tail_is_scenario':True}


class SpectralForcing:
    """Optional external angular-isotropic spectral forcing, no network calls.

    NPZ keys: times_utc (exact hourly labels), sw_um, dni_W_m2_um,
    dhi_W_m2_um; lw_um, sky_W_m2_um. Arrays have shape (time,wavelength).
    A JSON sidecar with source/model/profile hashes and units is mandatory.
    Hemisphere-only sky still assumes isotropic radiance for a tilted facade.
    No extrapolation in wavelength or time, no assumed meteorological profile.
    """
    def __init__(self,path):
        from pathlib import Path
        import pandas as pd
        self.path=Path(path);self.meta=json.loads(self.path.with_suffix('.json').read_text())
        for key in ['source','units','profile_sha256','model_version','purpose']:
            if not self.meta.get(key):raise ValueError('Missing spectrum metadata '+key)
        if self.meta['units']!={'wavelength':'um','spectral_irradiance':'W m-2 um-1'}:
            raise ValueError('Explicit spectral units must be um and W m-2 um-1')
        digest=self.meta['profile_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Need an actual SHA256 of the supplied atmospheric input/profile')
        self.z=np.load(self.path,allow_pickle=False)
        self.times=pd.DatetimeIndex(pd.to_datetime(self.z['times_utc'],utc=True))
        if not self.times.is_unique:raise ValueError('Duplicate spectral timestamps')
        self.index={str(t):i for i,t in enumerate(self.times)}
        for grid,cols in [('sw_um',['dni_W_m2_um','dhi_W_m2_um']),('lw_um',['sky_W_m2_um'])]:
            x=self.z[grid];trapezoid_weights(x)
            for col in cols:
                v=self.z[col]
                if v.shape!=(len(self.times),len(x)) or not np.isfinite(v).all() or (v<0).any():
                    raise ValueError('Invalid spectral array '+col)
        if self.z['sw_um'][0]>.28 or self.z['sw_um'][-1]<4 or self.z['lw_um'][0]>2.5 or self.z['lw_um'][-1]<100:
            raise ValueError('Incomplete supplied spectral range')

    def boundary(self,time,weather,radiation):
        import pandas as pd
        i=self.index.get(str(pd.Timestamp(time)))
        if i is None:raise ValueError('No external spectrum for '+str(time))
        sw=self.z['sw_um'];lw=self.z['lw_um']
        direct=self.z['dni_W_m2_um'][i];diffuse=self.z['dhi_W_m2_um'][i];sky=self.z['sky_W_m2_um'][i]
        # Prevent pairing an unrelated spectral atmosphere with city weather.
        for array,grid,key in [(direct,sw,'dni_W_m2'),(diffuse,sw,'dhi_W_m2'),(sky,lw,'lw_down_W_m2')]:
            value=float(np.trapezoid(array,grid));target=float(weather[key])
            if abs(value-target)>max(5,.05*target):raise ValueError('Spectral/broadband disagreement: '+key)
        a=np.interp(sw,radiation.x,radiation.a)
        eps=np.interp(lw,radiation.lam,radiation.epsilon)
        return {'dew_abs_dni':float(np.trapezoid(a*direct,sw)),
                'dew_abs_dhi':float(np.trapezoid(a*diffuse,sw)),
                'dew_abs_sky_lw':float(np.trapezoid(eps*sky,lw)),
                'status':'external_hemispherical_spectrum_isotropic_angle_assumption',
                'validation_only':self.meta['purpose']=='verification_only'}
