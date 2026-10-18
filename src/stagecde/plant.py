"""Fast research plant with exact inventory bookkeeping and causal observations.

Network temperatures/fluxes remain unqualified approximations of a conditional
screening solver. Conservation projection cannot make them experimentally true.
"""
import copy,json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from assembly import ROOT
from physics import PanelModel
from weather import model_config
from environment import observation
from actuator import pose,transition
from stagecde.transitions import features,PARAMS,RANGES,CACHE
from stagecde.surrogate import Emulator

def conserve_water(previous,raw):
    raw=np.maximum(0,np.asarray(raw,dtype=float));available=float(np.sum(previous)+raw[0]+raw[1])
    raw[2]=min(raw[2],available);left=available-raw[2];sink=float(raw[3:].sum())
    if sink>1e-15:raw[3:]*=left/sink
    else:raw[5:]=0;raw[5]=left
    return raw

def battery_step(soc,pv,motor,c):
    net=pv*c['dc_efficiency']-motor-c['idle_W'];surplus=cl=dl=unserved=0.
    if net>=0:
        incoming=min(net,(c['battery_capacity_Wh']-soc)/c['charge_efficiency']);new=soc+incoming*c['charge_efficiency'];cl=incoming*(1-c['charge_efficiency']);surplus=net-incoming
    else:
        supplied=min(-net,soc*c['discharge_efficiency']);new=soc-supplied/c['discharge_efficiency'];dl=supplied*(1/c['discharge_efficiency']-1);unserved=-net-supplied
    residual=soc+pv*c['dc_efficiency']-motor-c['idle_W']+unserved-surplus-cl-dl-new
    return new,{'surplus_Wh':surplus,'charge_loss_Wh':cl,'discharge_loss_Wh':dl,'unserved_Wh':unserved,'battery_residual_Wh':residual}

class FastPlant:
    def __init__(self,emulator=None,config=None,member=None):
        self.net=emulator or Emulator();self.c=dict(config or model_config());self.proxy=PanelModel(self.c);self.member=member
    @property
    def state(self):return self.proxy.state
    @state.setter
    def state(self,s):self.proxy.state=s
    def reset(self,ta):return self.proxy.reset(ta)
    def valid_actions(self,w):return self.proxy.valid_actions(w)
    def step_hour(self,w,action):
        s=self.state;c=self.c;mask=self.valid_actions(w);a=int(action);blocked=not mask[a]
        if blocked:a=s.face*5+s.tilt_index
        f,i=divmod(a,5);move=transition(c,s.angle_deg,pose(f,c['tilts_deg'][i]));motor=move['duration_s']*c['motor_running_upper_W']/3600
        x=features(s,w,a,c);y=self.net.predict(x,self.member)[0];ta=float(w.air_C)
        raw=.01*np.expm1(np.clip(y[8:16],0,12));old=float(s.film_kg.sum());raw=conserve_water(s.film_kg,raw)
        rain,cond,evap,drain,loss=raw[:5];s.film_kg=raw[5:].copy();s.temperature_C=np.clip(y[:8]+ta,-50,120)
        pv=float(np.clip(y[16],0,c['area_m2']*(float(w.dni_W_m2)+float(w.dhi_W_m2)+.3*float(w.ghi_W_m2))))
        s.battery_Wh,ledger=battery_step(s.battery_Wh,pv,motor,c)
        s.face=f;s.tilt_index=i;s.angle_deg=move['end'];s.cavity_temperature_C=float(np.clip(y[17]+ta,-50,100));s.cavity_mixing_ratio=float(np.clip(y[18],0,.1))
        s.wet_hours+=float(s.film_kg[2]>1e-8);s.uv_dose_MJ_m2+=max(0,float(w.ghi_W_m2))*.04*.0036
        ood=self.net.out_of_distribution(x);supported=bool(y[19]>=.9 and s.temperature_C.min()>=0 and not ood and float(w.get('snowfall_cm_h',0))==0)
        return {'rain_input_L':float(rain),'condensed_L':float(cond),'evaporated_L':float(evap),'drain_L':float(drain),'uncollected_departure_L':float(loss),
          'pv_gross_Wh':pv,'motor_Wh':motor,'idle_demand_Wh':c['idle_W'],'battery_Wh':s.battery_Wh,'film_L':float(s.film_kg.sum()),
          'mass_residual_kg':old+rain+cond-evap-drain-loss-s.film_kg.sum(),'water_supported':int(supported),'support_score':float(y[19]),'OOD':ood,
          'surface_pv_C':float(s.temperature_C[0]),'surface_dew_C':float(s.temperature_C[6]),'changed':int(move['duration_s']>0),'action_blocked':int(blocked),
          'resolved_action':a,'physical_yield_claim_supported':False,**ledger}

def reward(r,c):
    return c['reward_water_per_L']*r['drain_L']-c['reward_unserved_per_Wh']*r['unserved_Wh']-c['reward_switch_penalty']*r['changed']-(0 if r['water_supported'] else 10)

class ResearchEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,role='train',randomize=True,emulator=None):
        self.role=role;self.randomize=randomize;self.net=emulator or Emulator();self.groups=sorted(CACHE.glob(role+'-*.csv.gz'))
        if not self.groups:raise FileNotFoundError('Missing research episode cache')
        self.frames={p.name.removesuffix('.csv.gz'):pd.read_csv(p,index_col=0,parse_dates=True) for p in self.groups}
        self.metas={tag:json.loads((CACHE/(tag+'.json')).read_text()) for tag in self.frames}
        self.cities={x['id']:x for x in json.loads((ROOT/'config/cities.json').read_text())}
        self.action_space=spaces.Discrete(10);self.observation_space=spaces.Box(-1000,1000,shape=(50,),dtype=np.float32)
        self.c=model_config();self.plant=FastPlant(self.net,self.c);self.t=0
    def _obs(self):
        row=self.frame.iloc[self.t];city=self.cities[self.meta['city']]
        # Elevation is not inferred from air pressure: preserve geographic vs thermodynamic distinctions.
        mp=ROOT/'data/normalized'/f"{self.meta['city']}-{self.meta['year']}.metadata.json"
        if not hasattr(self,'elevations'):self.elevations={}
        if str(mp) not in self.elevations:self.elevations[str(mp)]=json.loads(mp.read_text())['era5_elevation_m']
        elev=self.elevations[str(mp)]
        obs=observation(row,self.plant.state,self.c,row.name,elev,city['lat'],self.plant.valid_actions(row))
        return np.r_[obs,row.wall_C/60,row.ground_C/60].astype(np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};tag=options.get('group',self.np_random.choice(list(self.frames)))
        self.meta=self.metas[tag];self.frame=self.frames[tag];self.tag=tag;self.t=0;self.c=model_config()
        if self.randomize:self.c.update({k:float(self.np_random.uniform(*RANGES[k])) for k in PARAMS})
        member=int(self.np_random.integers(len(self.net.members))) if self.randomize else None
        self.plant=FastPlant(self.net,self.c,member);self.plant.reset(float(self.frame.iloc[0].air_C));self.soc0=100.
        return self._obs(),{'group':tag}
    def step(self,action):
        # All policies observe row t. Only the plant reads realized row t+1.
        self.t+=1;r=self.plant.step_hour(self.frame.iloc[self.t],int(action));done=self.t>=len(self.frame)-1
        value=reward(r,self.c)
        if done:value-=self.c['reward_final_SOC_deficit_per_Wh']*max(0,self.soc0-r['battery_Wh'])
        return self._obs(),float(value),False,done,{**r,'group':self.tag,'time':str(self.frame.index[self.t])}
