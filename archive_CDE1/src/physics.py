"""Cassette screening solver. Numerical verification is not field validation.

Eight solid temperatures, algebraic two-port moist cavity, three water
reservoirs, spectral emitter and finite rotation. Backward Euler is solved
nonlinearly each substep. All uncertain closures remain exposed in config.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.optimize import root
from core import SIGMA,saturation_pressure_pa,R_VAPOR
from assembly import validate_config,MODEL_VERSION
from geometry import panel_frame,cassette_view_table,ray_category_from
from flow import mixing_ratio,air_properties,coupled_cavity,external_h
from spectral import SpectralRadiation,SpectralForcing
from drainage import drain
from actuator import pose,transition,motion_angle,load_envelope
from weather import model_config


@lru_cache(maxsize=12)
def radiation_cached(points,tail,angular,cap):
    return SpectralRadiation({'spectral_points':points,'dew_unknown_tail_epsilon':tail,
        'dew_angular_multiplier':angular,'sky_temperature_cap_offset_K':cap})


@dataclass
class State:
    temperature_C:np.ndarray
    film_kg:np.ndarray
    battery_Wh:float
    face:int=0
    tilt_index:int=2
    angle_deg:float=50.
    cavity_temperature_C:float=25.
    cavity_mixing_ratio:float=0.
    wet_hours:float=0.
    uv_dose_MJ_m2:float=0.
    model_version:str=MODEL_VERSION


class PanelModel:
    def __init__(self,config=None):
        self.c=dict(config or model_config());c=self.c
        self.assembly=validate_config(c);self.capacity=self.assembly['capacity_J_K'];self.lap=self.assembly['laplacian_W_K']
        self.rad=radiation_cached(c['spectral_points'],c['dew_unknown_tail_epsilon'],c['dew_angular_multiplier'],c['sky_temperature_cap_offset_K'])
        self.angles,self.vf=cassette_view_table(c['panel_width_m'],c['rotation_chord_m'],c['mounting_height_m'],c['wall_standoff_m'],c['building_height_m'],c['building_width_m'],self.assembly['assembly_depth_m'],c['geometry_angle_step_deg'],c['geometry_rays'],c['geometry_area_samples'])
        self.spectra=SpectralForcing(c['spectral_forcing_path']) if c.get('spectral_forcing_path') else None
        self.state=None

    @property
    def n_actions(self):return 2*len(self.c['tilts_deg'])

    def reset(self,air_C,battery_fraction=.5,face=0,tilt_index=2):
        if not 0<=battery_fraction<=1:raise ValueError('Invalid initial SOC')
        self.state=State(np.full(8,float(air_C)),np.zeros(3),self.c['battery_capacity_Wh']*battery_fraction,
                         face,tilt_index,pose(face,self.c['tilts_deg'][tilt_index]),float(air_C))
        return self.state

    def valid_actions(self,weather):
        s=self.state;c=self.c;nt=len(c['tilts_deg'])
        rho,_,_=air_properties(float(weather['air_C']),float(weather['pressure_Pa']))
        load=load_envelope(c,self.assembly,float(weather['wind_m_s']),rho,film_mass=float(s.film_kg.sum()))
        mask=[]
        for action in range(self.n_actions):
            f,i=divmod(action,nt);move=transition(c,s.angle_deg,pose(f,c['tilts_deg'][i]))
            power=c['motor_running_upper_W'] if c['motor_energy_policy']=='upper_running' else c['motor_running_W']
            mask.append(move['duration_s']==0 or (load['screen_pass'] and move['duration_s']*power/3600<=s.battery_Wh*c['discharge_efficiency']))
        return np.asarray(mask,dtype=bool)

    def _geometry(self,angle,weather):
        c=self.c;n=panel_frame(angle)[2];normals=np.array([n,-n])
        vf=np.array([[np.interp(angle%360,self.angles,self.vf[:,f,k]) for k in range(3)] for f in (0,1)])
        sun=np.array([weather.get('sun_x',0),weather.get('sun_y',1),weather.get('sun_z',0)],dtype=float)
        direct=np.zeros(2)
        if sun[1]>0:
            u,v,n=panel_frame(angle);sample=(np.arange(c['geometry_area_samples'])+.5)/c['geometry_area_samples']-.5
            for f,sign in enumerate((1,-1)):
                visible=0
                for x in sample:
                    for y in sample:
                        origin=np.array([0.,c['mounting_height_m'],0.])+x*c['panel_width_m']*u+y*c['rotation_chord_m']*v+sign*self.assembly['assembly_depth_m']/2*n
                        visible+=int(ray_category_from(sun,origin,c['wall_standoff_m'],c['building_height_m'],c['building_width_m'])[0]==0)
                direct[f]=max(0,float(sun@normals[f]))*visible/len(sample)**2
        poa=direct*float(weather['dni_W_m2'])+vf[:,0]*float(weather['dhi_W_m2'])+vf[:,1]*c['ground_shortwave_albedo']*float(weather['ghi_W_m2'])
        drop=np.array([weather.get('wind_x',0),-c['drop_fall_speed_m_s'],weather.get('wind_z',0)])
        rain=np.maximum(0,-normals@drop)/c['drop_fall_speed_m_s']*float(weather['rain_mm_h'])/3600*c['rain_interception_efficiency']
        return {'normals':normals,'vf':vf,'direct':direct,'poa':np.maximum(0,poa),'rain_kg_m2_s':rain}

    def step_hour(self,weather,action):
        c=self.c;s=self.state
        if s is None:raise RuntimeError('Call reset first')
        if int(action)!=action or not 0<=int(action)<self.n_actions:raise ValueError('Invalid action')
        keys=['air_C','pressure_Pa','rh_pct','wind_m_s','lw_down_W_m2','dni_W_m2','dhi_W_m2','ghi_W_m2','rain_mm_h']
        if not all(math.isfinite(float(weather[x])) for x in keys):raise ValueError('Nonfinite forcing')
        if any(float(weather[x])<0 for x in ['wind_m_s','lw_down_W_m2','dni_W_m2','dhi_W_m2','ghi_W_m2','rain_mm_h']):
            raise ValueError('Negative physical forcing')
        ta=float(weather['air_C']);p=float(weather['pressure_Pa']);rh=float(weather['rh_pct'])/100
        supersaturated=rh>1
        if rh<0 or p<=0:raise ValueError('Invalid thermodynamic forcing')
        wa=mixing_ratio(ta,min(rh,1),p);rho,_,_=air_properties(ta,p,wa)
        wind=max(0,float(weather['wind_m_s']));e=p*wa/(.621945+wa)
        tw=float(weather.get('wall_C',ta+c['wall_temperature_offset_K']));tg=float(weather.get('ground_C',ta))
        if not all(math.isfinite(x) and -100<x<200 for x in [tw,tg]):raise ValueError('Invalid surface-model boundary temperature')
        sky=self.rad.sky(ta,float(weather['lw_down_W_m2']));dew_sky=sky[c['sky_spectral_policy']]
        boundary=None
        if self.spectra:
            boundary=self.spectra.boundary(weather.name,weather,self.rad);dew_sky=boundary['dew_abs_sky_lw']
        f,i=divmod(int(action),len(c['tilts_deg']));target=pose(f,c['tilts_deg'][i])
        move=transition(c,s.angle_deg,target);mask=self.valid_actions(weather);blocked=not bool(mask[int(action)])
        outside_motor_temperature=not c['motor_temperature_min_C']<=ta<=c['motor_temperature_max_C']
        if blocked or outside_motor_temperature:
            move=transition(c,s.angle_deg,s.angle_deg);f=s.face;i=s.tilt_index;blocked=True
        changed=move['duration_s']>0;soc0=s.battery_Wh;film0=float(s.film_kg.sum());elapsed=0.
        keys=['rain','cond','evap','drain','lost','pv','motor','surplus','charge_loss','discharge_loss','unserved','volume','jump','slide']
        total={k:0. for k in keys}
        max_heat=max_humidity=max_cavity_energy=0.;max_iterations=0;min_t=float(s.temperature_C.min())
        min_dt=c['integration_dt_s'];frost=min_t<0;other_wall=False;wet_optics=False;motion_overload=False;last={}
        load=load_envelope(c,self.assembly,wind,rho,film_mass=float(s.film_kg.sum()))
        power=c['motor_running_upper_W'] if c['motor_energy_policy']=='upper_running' else c['motor_running_W']
        while elapsed<3600-1e-7:
            dt=min(c['integration_dt_s'],3600-elapsed)
            if elapsed<move['duration_s']<elapsed+dt:dt=move['duration_s']-elapsed
            old=s.temperature_C.copy();oldfilm=s.film_kg.copy()
            while True:
                angle=motion_angle(move,elapsed+dt/2);geo=self._geometry(angle,weather)
                ad=c['dew_area_m2'];area=c['area_m2'];areas=np.array([area,ad,ad]);indices=(0,6,6)
                rr=np.r_[geo['rain_kg_m2_s']*areas[:2],0.]*dt;available=oldfilm+rr
                pv_poa=geo['poa'][0];dew_poa=geo['poa'][1]
                if boundary:
                    dew_solar=boundary['dew_abs_dni']*geo['direct'][1]+boundary['dew_abs_dhi']*geo['vf'][1,0]+self.rad.solar_alpha*c['ground_shortwave_albedo']*float(weather['ghi_W_m2'])*geo['vf'][1,1]
                else:dew_solar=self.rad.solar_alpha*dew_poa
                solar_pv=c['pv_effective_absorptance']*pv_poa*area
                def terms(temp):
                    q=np.zeros(8);m=np.zeros(3)
                    for surf,node,a in [(0,0,area),(1,6,ad)]:
                        ts=float(temp[node]);ht=c['external_h_multiplier']*external_h(ts,ta,p,wind,c['rotation_chord_m'],c['natural_convection_floor_W_m2_K'])
                        hm=ht/(rho*1006)*c['lewis_number']**(-2/3)
                        flux=hm*(e-saturation_pressure_pa(ts))/(R_VAPOR*(273.15+(ts+ta)/2))*a
                        m[surf]=max(flux,-available[surf]/dt) if ts>=0 else 0
                        q[node]+=ht*a*(ta-ts)+m[surf]*(2501000-2320*ts)+rr[surf]/dt*4180*(ta-ts)
                    eff=max(0,c['pv_efficiency_at25']*(1+c['pv_gamma_per_K']*(temp[2]-25)))
                    eff=min(eff,c['pv_effective_absorptance']);pv=area*pv_poa*eff
                    q[0]+=solar_pv*(1-c['pv_solar_heat_cell_fraction'])
                    q[2]+=solar_pv*c['pv_solar_heat_cell_fraction']-pv;q[6]+=ad*dew_solar
                    fs,fg,fw=geo['vf'][0]
                    q[0]+=area*c['pv_longwave_emissivity']*(fs*float(weather['lw_down_W_m2'])+fg*SIGMA*(tg+273.15)**4+fw*SIGMA*(tw+273.15)**4-SIGMA*(temp[0]+273.15)**4)
                    fs,fg,fw=geo['vf'][1]
                    q[6]+=ad*(fs*dew_sky+fg*self.rad.exitance(tg)+fw*self.rad.exitance(tw)-self.rad.exitance(float(temp[6])))
                    flow,cavity=coupled_cavity(c,angle,wind,ta,temp[5],temp[6],wa,p,available[2],dt)
                    q[5]+=cavity['q_insulation_W'];q[6]+=cavity['q_dew_sensible_W']+cavity['latent_W'];m[2]=cavity['mass_kg_s']
                    qr=c['cavity_radiation_effective_epsilon']*SIGMA*ad*((temp[5]+273.15)**4-(temp[6]+273.15)**4)
                    q[5]-=qr;q[6]+=qr;q[7]+=c['frame_ambient_W_K']*(ta-temp[7])
                    return q,m,pv,flow,cavity
                def residual(temp):
                    q,*_=terms(temp)
                    return (self.capacity*(temp-old)/dt+self.lap@temp-q)/np.maximum(self.capacity/dt,1)
                sol=root(residual,old,method='hybr',options={'xtol':1e-9})
                err=float(np.max(np.abs(residual(sol.x))))
                if np.isfinite(sol.x).all() and err<c['nonlinear_tolerance_K']:break
                dt/=2
                if dt<.1:raise FloatingPointError(f'Nonlinear balance not converged: {err}')
            temp=sol.x;q,m,pv,flow,cavity=terms(temp);s.temperature_C=temp
            positive=np.maximum(0,m*dt);negative=np.maximum(0,-m*dt);film=available+positive-negative
            if film.min()<-1e-10:raise FloatingPointError('Negative film mass')
            film=np.maximum(film,0);collected=loss=jump=slide=0.
            for j in range(3):
                up=geo['normals'][j,1] if j<2 else -geo['normals'][1,1]
                film[j],dd,ll,detail=drain(c,film[j],positive[j],areas[j],up,angle,temp[indices[j]],dt,cnf=j==2)
                collected+=dd;loss+=ll;jump+=detail['jumping_kg'];slide+=detail['sliding_kg']
            s.film_kg=film
            wet_load=load_envelope(c,self.assembly,wind,rho,film_mass=float(film.sum()))
            for key in load:
                if key!='screen_pass':load[key]=max(load[key],wet_load[key])
            motion_overload|=bool(elapsed<move['duration_s'] and wet_load['drive_utilization']>1)
            heat_res=float(np.sum(self.capacity*(temp-old))-dt*np.sum(q));max_heat=max(max_heat,abs(heat_res))
            max_humidity=max(max_humidity,abs(cavity['humidity_balance_kg_s']));max_cavity_energy=max(max_cavity_energy,abs(cavity['enthalpy_residual_W']))
            max_iterations=max(max_iterations,sol.nfev);min_dt=min(min_dt,dt);min_t=min(min_t,float(temp.min()));frost|=min_t<0
            other_wall|=bool(cavity['other_wall_condensation_unsupported']);wet_optics|=film[1]>1e-8
            motor=power*dt/3600 if elapsed<move['duration_s']-1e-8 else 0.
            net=pv*dt/3600*c['dc_efficiency']-c['idle_W']*dt/3600-motor
            if net>=0:
                incoming=min(net,(c['battery_capacity_Wh']-s.battery_Wh)/c['charge_efficiency'])
                s.battery_Wh+=incoming*c['charge_efficiency'];total['charge_loss']+=incoming*(1-c['charge_efficiency']);total['surplus']+=net-incoming
            else:
                supplied=min(-net,s.battery_Wh*c['discharge_efficiency']);s.battery_Wh-=supplied/c['discharge_efficiency']
                total['discharge_loss']+=supplied*(1/c['discharge_efficiency']-1);total['unserved']+=-net-supplied
            for key,value in [('rain',rr.sum()),('cond',positive.sum()),('evap',negative.sum()),('drain',collected),('lost',loss),('pv',pv*dt/3600),('motor',motor),('volume',abs(flow['volume_m3_s'])*dt),('jump',jump),('slide',slide)]:total[key]+=float(value)
            if film[2]>0:s.wet_hours+=dt/3600
            s.uv_dose_MJ_m2+=dew_poa*self.rad.uv_fraction*dt/1e6
            elapsed+=dt;last={'flow':flow,'cavity':cavity,'geo':geo}
        s.face=f;s.tilt_index=i;s.angle_deg=move['end'];s.cavity_temperature_C=last['cavity']['temperature_C'];s.cavity_mixing_ratio=last['cavity']['mixing_ratio']
        mass_res=film0+total['rain']+total['cond']-total['evap']-total['drain']-total['lost']-s.film_kg.sum()
        battery_res=soc0+total['pv']*c['dc_efficiency']-(c['idle_W']+total['motor']-total['unserved'])-total['surplus']-total['charge_loss']-total['discharge_loss']-s.battery_Wh
        snow=float(weather.get('snowfall_cm_h',0))>0;mech_fail=load['hold_utilization']>1 or load['tilt_utilization']>1 or load['radial_utilization']>1
        supported=not(frost or snow or other_wall or supersaturated or mech_fail or motion_overload or outside_motor_temperature)
        return {'model_version':MODEL_VERSION,'drain_L':total['drain'],'rain_input_L':total['rain'],
          'condensed_L':total['cond'],'evaporated_L':total['evap'],'uncollected_departure_L':total['lost'],
          'film_L':float(s.film_kg.sum()),'pv_gross_Wh':total['pv'],'dc_available_Wh':total['pv']*c['dc_efficiency'],
          'surplus_Wh':total['surplus'],'motor_Wh':total['motor'],'motor_nominal_reference_Wh':move['nominal_Wh'],
          'motion_seconds':move['duration_s'],'idle_demand_Wh':c['idle_W'],'unserved_Wh':total['unserved'],
          'battery_Wh':s.battery_Wh,'soc_delta_Wh':s.battery_Wh-soc0,
          'charge_loss_Wh':total['charge_loss'],'discharge_loss_Wh':total['discharge_loss'],
          'mass_residual_kg':float(mass_res),'battery_residual_Wh':battery_res,
          'max_thermal_residual_J':max_heat,'max_cavity_enthalpy_residual_W':max_cavity_energy,
          'max_cavity_humidity_residual_kg_s':max_humidity,'nonlinear_max_evaluations':max_iterations,
          'frost_unsupported':int(frost),'snow_unsupported':int(snow),'other_wall_condensation_unsupported':int(other_wall),
          'mechanical_hold_unsupported':int(mech_fail),'supersaturation_unsupported':int(supersaturated),
          'motor_temperature_unsupported':int(outside_motor_temperature),'wet_emitter_dry_optics_approximation':int(wet_optics),
          'motion_overload_unsupported':int(motion_overload),
          'min_surface_C':min_t,'surface_pv_C':float(s.temperature_C[0]),'cell_C':float(s.temperature_C[2]),
          'surface_dew_C':float(s.temperature_C[6]),'insulation_inner_C':float(s.temperature_C[5]),
          'cavity_C':s.cavity_temperature_C,'cavity_mixing_ratio':s.cavity_mixing_ratio,
          'cavity_flow_mean_m3_s':total['volume']/3600,'cavity_Re_final':last['flow']['Re'],
          'jumping_departure_L':total['jump'],'sliding_departure_L':total['slide'],
          'coating_wet_hours':s.wet_hours,'emitter_UV_MJ_m2':s.uv_dose_MJ_m2,
          'face':f,'tilt_deg':c['tilts_deg'][i],'angle_deg':s.angle_deg,
          'poa_W_m2':float(last['geo']['poa'][0]),'changed':int(changed),'action_blocked':int(blocked),
          'effective_dt_s':min_dt,'water_supported':int(supported),'drive_utilization':load['drive_utilization'],
          'hold_utilization':load['hold_utilization'],'tilting_utilization':load['tilt_utilization'],
          'sky_dew_absorption_lower_W_m2':float(sky['low']),'sky_dew_absorption_upper_W_m2':float(sky['high']),
          'external_spectral_boundary':bool(boundary),'wall_boundary_C':tw,'ground_boundary_C':tg,
          'external_surface_model_boundary':bool('wall_C' in weather and 'ground_C' in weather),
          'physical_yield_claim_supported':False,'evidence_status':c['status']}
