"""Uncalibrated two-node heat/film/battery reference, NOT a high-fidelity solver.

No active refrigeration, ice, fog impaction, CFD, runoff contamination or potable
water treatment. Invalid freezing states stop defensible liquid-water accounting.
"""
from dataclasses import dataclass
import math
import numpy as np
from core import SIGMA, R_VAPOR, saturation_pressure_pa
from geometry import panel_frame, view_factors
from weather import model_config


@dataclass
class State:
    temperature_C: np.ndarray
    film_kg: np.ndarray
    battery_Wh: float
    face: int = 0
    tilt_index: int = 2


class PanelModel:
    def __init__(self, config=None):
        self.c = dict(config or model_config())
        c = self.c
        self.factors = [view_factors(t,c['mounting_height_m'],c['wall_standoff_m'],c['building_height_m'],c['building_width_m']) for t in c['tilts_deg']]
        self.normals = [panel_frame(t)[2] for t in c['tilts_deg']]
        self.state = None

    @property
    def n_actions(self):
        return 2*len(self.c['tilts_deg'])

    def reset(self, air_C, battery_fraction=.5, face=0, tilt_index=2):
        self.state = State(np.full(2,float(air_C)),np.zeros(2),self.c['battery_capacity_Wh']*battery_fraction,face,tilt_index)
        return self.state

    def step_hour(self, weather, action):
        c,s = self.c,self.state
        if s is None: raise RuntimeError('Call reset before stepping')
        if not 0<=int(action)<self.n_actions: raise ValueError('Invalid action')
        nt = len(c['tilts_deg']); requested_face,requested_tilt = divmod(int(action),nt)
        changed = requested_face!=s.face or requested_tilt!=s.tilt_index
        requested_cost = (c['face_switch_Wh']*(requested_face!=s.face)
                          + c['tilt_Wh_per_deg']*abs(c['tilts_deg'][requested_tilt]-c['tilts_deg'][s.tilt_index]))
        soc_initial = s.battery_Wh
        motor = 0.; action_blocked = 0
        if requested_cost/c['discharge_efficiency'] <= s.battery_Wh+1e-12:
            s.battery_Wh -= requested_cost/c['discharge_efficiency']
            motor = requested_cost
            s.face,s.tilt_index = requested_face,requested_tilt
        else:
            changed = False; action_blocked = 1
        f = s.face; b = 1-f; area = c['area_m2']
        tilt = c['tilts_deg'][s.tilt_index]
        normal = self.normals[s.tilt_index]
        fs,fg,fw = self.factors[s.tilt_index]
        ta = float(weather['air_C']); rh = float(weather['rh_pct'])/100
        p = float(weather['pressure_Pa']); wind = max(0.,float(weather['wind_m_s']))
        lw = float(weather['lw_down_W_m2'])
        if not math.isfinite(lw): raise ValueError('Missing LW boundary condition')
        poa = max(0.,float(weather[f'poa_{s.tilt_index}']))
        tw = ta+c['wall_temperature_offset_K']; tg = ta
        vapor_p = rh*saturation_pressure_pa(ta)
        # Ideal-gas moist density, local pressure rather than standard sea level.
        rho = (p-vapor_p)/(287.05*(ta+273.15))+vapor_p/(R_VAPOR*(ta+273.15))
        h = c['front_convection_intercept_W_m2_K']+c['front_convection_wind_slope']*wind
        hm = h/(rho*1006.)*c['lewis_number']**(-2/3)
        drop = np.array([float(weather['wind_x']),-c['drop_fall_speed_m_s'],float(weather['wind_z'])])
        rain_rate = max(0.,float(weather['rain_mm_h']))/3600*max(0.,-float(drop@normal))/c['drop_fall_speed_m_s']*c['rain_interception_efficiency']
        # Heavy rain and latent transfer can make a 60 s explicit step unstable.
        # Use a conservative local heat-transfer slope to cap the requested dt.
        # This is a numerical safeguard, not an error-controlled CFD integrator.
        bound_C=min(120.,max(ta,float(s.temperature_C[f]))+10.)
        dpdt=(saturation_pressure_pa(bound_C+.05)-saturation_pressure_pa(bound_C-.05))/.1
        latent_slope=2.5e6*hm*dpdt/(R_VAPOR*(273.15+(ta+bound_C)/2))
        front_slope=h+4*c['longwave_emissivity'][f]*SIGMA*(bound_C+273.15)**3+4180*rain_rate+latent_slope+c['inter_face_conductance_W_m2_K']
        back_slope=c['back_conductance_W_m2_K']+c['inter_face_conductance_W_m2_K']
        dt_limit=.4*min(c['heat_capacity_J_m2_K'][f]/max(front_slope,1e-9),c['heat_capacity_J_m2_K'][b]/max(back_slope,1e-9))
        nsteps = math.ceil(3600/min(c['integration_dt_s'],dt_limit)); dt = 3600/nsteps
        capacity = np.array(c['heat_capacity_J_m2_K'],dtype=float)*area
        alpha = c['solar_absorptance'][f]; eps = c['longwave_emissivity'][f]
        film0 = float(s.film_kg.sum())
        rain=cond=evap=drain=pv_Wh=surplus=charge_loss=discharge_loss=unserved=0.
        max_thermal_residual = 0.; frost = False; ts_min = float(s.temperature_C.min())
        for _ in range(nsteps):
            temp = s.temperature_C.copy(); ts = float(temp[f])
            if min(temp)<0: frost=True
            # A dry heat calculation continues for diagnostics only below 0 C.
            # Its water result is flagged unsupported and excluded from scoring.
            rr = rain_rate*area*dt if ts>=0 else 0.
            rain += rr; s.film_kg[f] += rr
            net_mass = 0.
            if ts>=0:
                film_K = 273.15+(ta+ts)/2
                potential = hm*(vapor_p-saturation_pressure_pa(ts))/(R_VAPOR*film_K)*area*dt
                if potential>=0:
                    cond += potential; s.film_kg[f] += potential; net_mass = potential
                else:
                    loss = min(s.film_kg[f],-potential)
                    evap += loss; s.film_kg[f] -= loss; net_mass = -loss
            for j in (0,1):
                if temp[j]>=0:
                    flow = max(0.,s.film_kg[j]-c['film_hold_up_kg_m2']*area)*(1-math.exp(-dt/c['drain_time_s']))
                    s.film_kg[j] -= flow; drain += flow
            eff = min(alpha,max(0.,c['pv_efficiency_at25']*(1+c['pv_gamma_per_K']*(ts-25)))) if f==0 else 0.
            pv_w = area*poa*eff
            latent_J_kg = 2.501e6-2369*ts
            radiation = eps*(fs*lw+fg*SIGMA*(tg+273.15)**4+fw*SIGMA*(tw+273.15)**4-SIGMA*(ts+273.15)**4)
            front = area*(alpha*poa+h*(ta-ts)+radiation)+latent_J_kg*net_mass/dt+4180*rr*(ta-ts)/dt-pv_w
            coupling = area*c['inter_face_conductance_W_m2_K']*(temp[b]-ts)
            backside = area*c['back_conductance_W_m2_K']*(tw-temp[b])
            q = np.zeros(2);q[f]=front+coupling;q[b]=backside-coupling
            s.temperature_C += dt*q/capacity
            ts_min = min(ts_min,float(s.temperature_C.min()))
            frost |= bool((s.temperature_C<0).any())
            if not np.isfinite(s.temperature_C).all() or np.max(np.abs(s.temperature_C))>150:
                raise FloatingPointError('Thermal integrator diverged; lower time step/check boundaries')
            residual = float(np.sum(capacity*(s.temperature_C-temp))-dt*(front+backside))
            max_thermal_residual = max(max_thermal_residual,abs(residual))
            generated = pv_w*dt/3600*c['dc_efficiency']
            pv_Wh += pv_w*dt/3600
            load = c['idle_W']*dt/3600
            net = generated-load
            if net>=0:
                charge = min(net,(c['battery_capacity_Wh']-s.battery_Wh)/c['charge_efficiency'])
                s.battery_Wh += charge*c['charge_efficiency']
                charge_loss += charge*(1-c['charge_efficiency'])
                surplus += net-charge
            else:
                supplied = min(-net,s.battery_Wh*c['discharge_efficiency'])
                s.battery_Wh -= supplied/c['discharge_efficiency']
                discharge_loss += supplied*(1/c['discharge_efficiency']-1)
                unserved += -net-supplied
        discharge_loss += motor*(1/c['discharge_efficiency']-1)
        mass_residual = film0+rain+cond-evap-drain-float(s.film_kg.sum())
        dc_energy = pv_Wh*c['dc_efficiency']
        battery_residual = soc_initial+dc_energy-(c['idle_W']-unserved)-motor-surplus-charge_loss-discharge_loss-s.battery_Wh
        return {'drain_L':drain,'rain_input_L':rain,'condensed_L':cond,'evaporated_L':evap,
                'film_L':float(s.film_kg.sum()),'pv_gross_Wh':pv_Wh,'dc_available_Wh':dc_energy,
                'surplus_Wh':surplus,'motor_Wh':motor,'idle_demand_Wh':c['idle_W'],
                'unserved_Wh':unserved,'battery_Wh':s.battery_Wh,'soc_delta_Wh':s.battery_Wh-soc_initial,
                'charge_loss_Wh':charge_loss,'discharge_loss_Wh':discharge_loss,
                'mass_residual_kg':mass_residual,'battery_residual_Wh':battery_residual,
                'max_thermal_residual_J':max_thermal_residual,
                'frost_unsupported':int(frost),'snow_unsupported':int(float(weather.get('snowfall_cm_h',0))>0),
                'min_surface_C':ts_min,'surface_pv_C':float(s.temperature_C[0]),'surface_dew_C':float(s.temperature_C[1]),
                'face':s.face,'tilt_deg':tilt,'poa_W_m2':poa,'changed':int(changed),
                'action_blocked':action_blocked,'effective_dt_s':dt,
                'water_supported':int(not frost and float(weather.get('snowfall_cm_h',0))==0)}
