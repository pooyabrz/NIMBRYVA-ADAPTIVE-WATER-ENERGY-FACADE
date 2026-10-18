"""Gymnasium PPO environment. A step acts on an UNKNOWN next interval.

Observations use row t and the current physical state; step physics uses t+1.
No future reanalysis enters the observation. This is a persistence-information
baseline, not a weather forecasting model. Cold/frozen operation is unsupported.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from weather import load_weather, model_config
from physics import PanelModel


def observation(row,state,config,time,site_elevation_m,latitude):
    hour=time.hour;day=time.dayofyear;s=state
    values=[row.air_C/40,row.rh_pct/100,row.dew_C/40,row.pressure_Pa/101325,
            row.wind_m_s/15,row.wind_x/15,row.wind_z/15,row.rain_mm_h/20,
            row.ghi_W_m2/1000,row.lw_down_W_m2/500,row.cloud_pct/100,
            *list(s.temperature_C/60),*list(s.film_kg/config['area_m2']),
            s.battery_Wh/config['battery_capacity_Wh'],float(s.face),
            config['tilts_deg'][s.tilt_index]/90,
            np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*day/366),np.cos(2*np.pi*day/366),
            site_elevation_m/4000,row.sun_x,row.sun_y,row.sun_z,latitude/90]
    return np.asarray(values,dtype=np.float32)


class FacadeEnv(gym.Env):
    metadata = {'render_modes':[]}

    def __init__(self, city_ids=('rasht','bandar-abbas','singapore','mumbai','dubai','phoenix','seville','yazd'),
                 year=2024, episode_hours=168, config=None):
        super().__init__()
        self.cfg = config or model_config()
        self.model = PanelModel(self.cfg)
        self.action_space = spaces.Discrete(self.model.n_actions)
        # Broad finite bounds; physical forcing has stricter source QC limits.
        self.observation_space = spaces.Box(-1000.,1000.,shape=(27,),dtype=np.float32)
        self.episode_hours = int(episode_hours)
        self.frames = {};self.starts = {}
        for city in city_ids:
            frame = load_weather(city,year,self.cfg)
            # Select warm, snow-free engineering episodes. This selection is not
            # a general climate evaluation and cannot support cold-city claims.
            supported = (frame.air_C>=8)&(frame.snowfall_cm_h==0)&frame.lw_down_W_m2.notna()
            bad = np.concatenate(([0],np.cumsum(~supported.to_numpy())))
            starts = np.arange(len(frame)-self.episode_hours)
            valid = starts[(bad[starts+self.episode_hours+1]-bad[starts])==0]
            if len(valid):
                self.frames[city]=frame;self.starts[city]=valid
        if not self.frames: raise ValueError('No warm supported episodes for this city/year selection')
        self.t=0;self.elapsed=0

    def _observation(self):
        attr = self.frame.attrs
        return observation(self.frame.iloc[self.t],self.model.state,self.cfg,self.frame.index[self.t],
                           attr['site_elevation_m'],attr['city']['lat'])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.city_id = options.get('city',self.np_random.choice(list(self.frames)))
        self.frame = self.frames[self.city_id]
        self.t = int(options.get('start',self.np_random.choice(self.starts[self.city_id])))
        if self.t not in self.starts[self.city_id]:
            raise ValueError('Start is outside the warm supported episode set')
        self.start = self.t;self.elapsed=0
        self.model.reset(self.frame.iloc[self.t].air_C)
        self.soc0 = self.model.state.battery_Wh
        return self._observation(),{'city':self.city_id,'time':str(self.frame.index[self.t]),'status':self.cfg['status']}

    def step(self, action):
        # The action is chosen before realized weather at t+1 is visible.
        self.t += 1;self.elapsed += 1
        result = self.model.step_hour(self.frame.iloc[self.t],int(action))
        terminated = not bool(result['water_supported'])
        truncated = self.elapsed>=self.episode_hours
        reward = (self.cfg['reward_water_per_L']*result['drain_L']
                  -self.cfg['reward_unserved_per_Wh']*result['unserved_Wh']
                  -self.cfg['reward_switch_penalty']*result['changed'])
        if terminated: reward -= 10.  # unsupported domain, never a positive result
        if terminated or truncated:
            reward -= self.cfg['reward_final_SOC_deficit_per_Wh']*max(0.,self.soc0-result['battery_Wh'])
        info = {**result,'city':self.city_id,'time':str(self.frame.index[self.t]),
                'status':self.cfg['status'],'unsupported_terminal':terminated}
        return self._observation(),float(reward),terminated,truncated,info
