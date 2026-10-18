"""Mass-conserving water reservoirs; collection remains an explicit scenario.

CNF jumping fraction, deposited-drop footprint density and gutter capture are
NOT identified by advancing/receding angles or ETH's steady collected flux.
No fitted 30 s drainage constant or outdoor lifetime is asserted.
"""
import math


def surface_tension(t):
    tau=1-(max(0,t)+273.15)/647.096
    return .2358*tau**1.256*(1-.625*tau)


def depinning_mass(area,angle,t,adv,rec,footprints_per_m2,shape=1):
    """Furmidge retention force vs gravity, monodisperse spherical droplets.

    This diagnostic needs an assumed footprint population; it does not model
    drop nucleation, coalescence kinetics, pendent drops or a wetting lifetime.
    """
    sb=abs(math.sin(math.radians(angle)))
    if sb<1e-8:return math.inf
    dc=max(0,math.cos(math.radians(rec))-math.cos(math.radians(adv)))
    r=math.sqrt(3*shape*surface_tension(t)*dc/(2*math.pi*1000*9.80665*sb))
    return area*footprints_per_m2*(4*math.pi*r**3/3)*1000


def drain(c,film,cond_new,area,normal_up,angle,t,dt,cnf=False):
    if t<0:return film,0.,0.,{'jumping_kg':0.,'sliding_kg':0.}
    quality=c['coating_condition']
    # Capturable jumping departure only when condenser faces downward.
    jump=min(film,max(0,cond_new)*c['jump_fraction']*quality) if cnf and normal_up<0 else 0.
    film-=jump
    adv=161.2 if cnf else c['outer_adv_deg'];rec=159.4 if cnf else c['outer_rec_deg']
    # Degraded wetting is a stress case at fixed condition, not a time law.
    if cnf:adv=110+(adv-110)*quality;rec=70+(rec-70)*quality
    pinned=depinning_mass(area,angle,t,adv,rec,c['droplet_footprints_m2'])
    mobile=max(0,film-pinned)
    delta=mobile/(1000*area)
    mu=c['water_viscosity_Pa_s'] # declared near-20 C proxy
    # Smooth laminar gravity film limit after depinning. Inverted shedding,
    # capillary rivulets and splash are not solved, hence no delivery claim.
    flow=1000**2*9.80665*abs(math.sin(math.radians(angle)))*c['panel_width_m']*delta**3/(3*mu)
    slide=min(mobile,flow*dt)
    film-=slide;departed=jump+slide
    collected=departed*c['collector_efficiency']
    loss=departed-collected
    return film,collected,loss,{'jumping_kg':jump,'sliding_kg':slide}
