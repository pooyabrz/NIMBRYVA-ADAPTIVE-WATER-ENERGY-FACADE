"""Two-port pressure network and quasi-steady cavity humidity/enthalpy.

Network equations follow Bernoulli/stack pressure balance, not resolved CFD.
The port pressure coefficient is an epistemic input. No automatic conversion
of reanalysis wind into a measured facade pressure distribution is made.
"""
import math
from scipy.optimize import brentq
from core import saturation_pressure_pa


def mixing_ratio(t,rh,p):
    e=rh*saturation_pressure_pa(t)
    if not 0<=e<p:raise ValueError('Invalid vapour partial pressure')
    return .621945*e/(p-e)


def air_properties(t,p,w=0):
    tk=t+273.15
    if tk<=0 or p<=0:raise ValueError('Invalid air state')
    e=p*w/(.621945+w)
    rho=(p-e)/(287.05*tk)+e/(461.52*tk)
    mu=1.716e-5*(tk/273.15)**1.5*(273.15+111)/(tk+111)
    k=.0241*(tk/273.15)**.9
    return rho,mu,k


def channel_flow(cfg,angle_deg,wind,ta,tc,p,w):
    rho,mu,k=air_properties((ta+tc)/2,p,w)
    width=cfg['panel_width_m'];gap=cfg['cavity_gap_m'];length=cfg['rotation_chord_m']
    area=width*gap;dh=2*width*gap/(width+gap)
    dpw=.5*rho*wind**2*cfg['port_delta_Cp']
    dps=rho*9.80665*length*math.sin(math.radians(angle_deg))*(tc-ta)/(ta+273.15)
    dp=dpw+dps
    ai=cfg['port_inlet_m2'];ao=cfg['port_outlet_m2'];cd=cfg['port_discharge_coefficient']
    if ai==0 or ao==0 or abs(dp)<1e-12:
        return {'volume_m3_s':0.,'dry_air_kg_s':0.,'velocity_m_s':0.,'Re':0.,'dp_Pa':dp,'h_W_m2_K':k/gap}
    if min(ai,ao,cd)<=0:raise ValueError('Invalid port geometry')
    resistance=1/(cd*ai)**2+1/(cd*ao)**2
    def pressure(q):
        re=rho*(q/area)*dh/mu
        # Parallel-plate laminar Darcy f*Re=96. Rectangular finite-width and
        # developing-flow effects remain part of the uncertainty envelope.
        viscous=48*mu*length*q/(dh**2*area)
        if re>2300:
            f=.3164/re**.25
            viscous=.5*rho*(q/area)**2*f*length/dh
        return .5*rho*q*q*resistance+viscous
    upper=math.sqrt(2*abs(dp)/(rho*resistance))
    q=brentq(lambda x:pressure(x)-abs(dp),0,upper,xtol=1e-12)
    vel=q/area;re=rho*vel*dh/mu
    pr=1006*mu/k
    # Nu=7.54 is the fully developed parallel-plate isothermal limit; both
    # walls of the real cavity are not isothermal. This is a screening closure.
    nu=7.54 if re<=2300 else .023*re**.8*pr**.4
    return {'volume_m3_s':math.copysign(q,dp),'dry_air_kg_s':rho*q/(1+w),
            'velocity_m_s':vel,'Re':re,'dp_Pa':dp,'h_W_m2_K':cfg['channel_h_multiplier']*nu*k/dh}


def cavity_balance(mda,h,area,ta,tins,tdew,wa,p,film_available,dt,lewis=.86):
    """Algebraic well-mixed cavity, negligible air storage.

    Moist-air enthalpy reference h=1006*T+w*(2501000+1860*T), J/kg dry.
    A surface flux is bounded by both mass transfer and vapour replenishment.
    No condensation on other internal surfaces is included: flag if needed.
    """
    ws=mixing_ratio(tdew,1,p) if tdew>=0 else mixing_ratio(0,1,p)
    conductance=h*area
    # Lewis heat/mass analogy in humidity-ratio coordinates.
    gm=conductance/1006*lewis**(-2/3)
    wc=(mda*wa+gm*ws)/(mda+gm) if mda+gm else wa
    flux=mda*(wa-wc)
    if tdew<0:flux=0.;wc=wa
    if flux<0:
        flux=max(flux,-film_available/dt)
        wc=wa-flux/mda if mda else wa
    # Sensible cooling of the captured condensate is included so that total
    # enthalpy removed from air equals liquid enthalpy + latent heat to solid.
    den=mda*(1006+1860*wc)+2*conductance+flux*1860
    num=mda*(1006*ta+wa*(2501000+1860*ta)-wc*2501000)+conductance*(tins+tdew)-flux*2501000
    tc=num/den if den>0 else .5*(tins+tdew)
    lv=2501000+(1860-4180)*tdew
    q_lat=flux*lv
    q_cond_sensible=flux*1860*(tc-tdew)
    # Numerically inspect total moist enthalpy balance, including liquid drain.
    hin=1006*ta+wa*(2501000+1860*ta)
    hout=1006*tc+wc*(2501000+1860*tc)
    residual=mda*(hin-hout)-conductance*(tc-tins)-conductance*(tc-tdew)-q_lat-flux*4180*tdew-q_cond_sensible
    return {'temperature_C':tc,'mixing_ratio':wc,'mass_kg_s':flux,'latent_W':q_lat+q_cond_sensible,
            'q_insulation_W':conductance*(tc-tins),'q_dew_sensible_W':conductance*(tc-tdew),
            'humidity_balance_kg_s':mda*(wa-wc)-flux,'enthalpy_residual_W':residual,
            'other_wall_condensation_unsupported':int(wc>mixing_ratio(max(0,tins),1,p)+1e-10)}


def external_h(t,ta,p,wind,length,natural_floor):
    rho,mu,k=air_properties((t+ta)/2,p)
    re=rho*max(wind,0)*length/mu;pr=1006*mu/k
    nu=(.664*re**.5 if re<5e5 else .037*re**.8-871)*pr**(1/3)
    # A declared natural-convection floor, not an all-angle correlation.
    return (max(nu*k/length,0)**3+natural_floor**3)**(1/3)


def coupled_cavity(cfg,angle,wind,ta,tins,tdew,wa,p,film_available,dt):
    """Bracket stack/temperature coupling, including opposing wind and stack.

    Direct fixed-point iteration may oscillate near zero net port pressure.
    Sensible/enthalpy algebra keeps Tc between its three boundary temperatures.
    """
    def evaluate(tc):
        f=channel_flow(cfg,angle,wind,ta,tc,p,wa)
        b=cavity_balance(f['dry_air_kg_s'],f['h_W_m2_K'],cfg['dew_area_m2'],ta,tins,tdew,wa,p,film_available,dt,cfg['lewis_number'])
        return f,b
    lo=min(ta,tins,tdew);hi=max(ta,tins,tdew)
    if hi-lo<1e-10:return evaluate((lo+hi)/2)
    tc=brentq(lambda t:evaluate(t)[1]['temperature_C']-t,lo,hi,xtol=1e-9)
    return evaluate(tc)
