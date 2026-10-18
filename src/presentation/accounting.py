"""Passive tracers. They observe the solver; they never alter physical states.

Water tags identify the mechanism of entry: rain, surface condensation, initial
inventory. Evaporation/sliding use perfect mixing within each resolved reservoir;
CNF jumping departure is taken from newly condensed water first. This is not an
isotopic or droplet-resolved gas-phase recirculation model.

Battery tags allocate delivered energy between initial charge and new PV.
This bookkeeping convention prevents attributing initial battery energy to PV.
"""
import numpy as np

ORIGINS=['rain','condensation','initial'];ENERGY_ORIGINS=['initial','pv']

def withdraw_mixed(stock,mass):
    stock=np.asarray(stock,dtype=float);amount=float(mass);available=float(stock.sum())
    if amount < -1e-12 or amount>available+1e-9:raise ValueError('Tracer withdrawal exceeds inventory')
    if available<=1e-20:return np.zeros_like(stock),stock.copy()
    taken=stock*min(1.,max(0.,amount/available));return taken,np.maximum(0,stock-taken)

def water_update(stock,rain,condensation,evaporation,collected,loss,jumping,efficiency):
    x=np.asarray(stock,dtype=float).copy();x[0]+=rain;x[1]+=condensation
    evaporated,x=withdraw_mixed(x,evaporation)
    # The physical CNF closure defines jumping as a fraction of new condensation.
    jump=np.zeros(3);jump[1]=min(jumping,x[1]);remaining=jumping-jump[1];x-=jump
    if remaining>1e-12:
        rest,x=withdraw_mixed(x,remaining);jump+=rest
    sliding=max(0,collected+loss-jumping);slide,x=withdraw_mixed(x,sliding)
    departed=jump+slide;out=departed*efficiency;lost=departed-out
    if abs(out.sum()-collected)>1e-9 or abs(lost.sum()-loss)>1e-9:raise ValueError('Tracer/physical drainage mismatch')
    return x,evaporated,out,lost

class AuditObserver:
    def __init__(self,config):self.c=config
    def reset(self,film,battery):
        self.water=np.zeros((3,3));self.water[:,2]=film;self.battery=np.array([float(battery),0.]);self.soc_initial=float(battery)
        self.cumulative={k:np.zeros(3) for k in ['input','evap','collected','loss']};self.incident_rain=0.;self.entry_loss=0.
        self.energy={k:0. for k in ['pv_direct','pv_battery','initial_battery','conversion_loss','pv_charge_loss','pv_discharge_loss','initial_discharge_loss','surplus']}
        self.begin_hour(film,battery)
    def begin_hour(self,film,battery):
        if not np.allclose(self.water.sum(1),film,atol=1e-9,rtol=1e-10) or abs(self.battery.sum()-battery)>1e-8:raise ValueError('Audit state diverged from physical state')
        self.water_start=self.water.copy();self.battery_start=self.battery.copy();self.cumulative_start={k:v.copy() for k,v in self.cumulative.items()};self.energy_start=dict(self.energy)
        self.incident_start=self.incident_rain;self.entry_start=self.entry_loss
    def water_step(self,rain,cond,evap,collected,loss,jump,final_film,incident):
        eta=self.c['collector_efficiency']
        for j in range(3):
            self.water[j],ev,dd,ll=water_update(self.water[j],rain[j],cond[j],evap[j],collected[j],loss[j],jump[j],eta)
            self.cumulative['input'][:2]+=[rain[j],cond[j]];self.cumulative['evap']+=ev;self.cumulative['collected']+=dd;self.cumulative['loss']+=ll
        self.incident_rain+=float(np.sum(incident));self.entry_loss+=float(np.sum(incident)-np.sum(rain))
        if not np.allclose(self.water.sum(1),final_film,atol=1e-9,rtol=1e-10):raise ValueError('Water tag closure failed')
    def energy_step(self,pv_gross,demand,battery_after):
        c=self.c;dc=pv_gross*c['dc_efficiency'];direct=min(dc,demand);self.energy['pv_direct']+=direct;self.energy['conversion_loss']+=pv_gross-dc;net=dc-demand
        if net>=0:
            incoming=min(net,(c['battery_capacity_Wh']-self.battery.sum())/c['charge_efficiency'])
            self.battery[1]+=incoming*c['charge_efficiency'];self.energy['pv_charge_loss']+=incoming*(1-c['charge_efficiency']);self.energy['surplus']+=net-incoming
        else:
            supplied=min(-net,float(self.battery.sum())*c['discharge_efficiency']);removed,self.battery=withdraw_mixed(self.battery,supplied/c['discharge_efficiency'])
            delivered=removed*c['discharge_efficiency'];self.energy['initial_battery']+=delivered[0];self.energy['pv_battery']+=delivered[1]
            self.energy['initial_discharge_loss']+=removed[0]-delivered[0];self.energy['pv_discharge_loss']+=removed[1]-delivered[1]
        if abs(self.battery.sum()-battery_after)>1e-8:raise ValueError('Battery tracer closure failed')
    def finish_hour(self,film,battery):
        result={};inputs=self.cumulative['input']-self.cumulative_start['input'];evap=self.cumulative['evap']-self.cumulative_start['evap'];dd=self.cumulative['collected']-self.cumulative_start['collected'];loss=self.cumulative['loss']-self.cumulative_start['loss']
        delta=self.water.sum(0)-self.water_start.sum(0);balance=inputs-evap-dd-loss-delta
        for i,name in enumerate(ORIGINS):
            for key,a in [('input',inputs),('evaporated',evap),('collected',dd),('gutter_loss',loss),('stored',self.water.sum(0)),('storage_delta',delta),('source_balance',balance)]:result[f'{key}_{name}_L']=float(a[i])
        for k,v in self.energy.items():result[k+'_Wh']=float(v-self.energy_start[k])
        result['geometric_incident_rain_L']=self.incident_rain-self.incident_start;result['rain_entry_loss_L']=self.entry_loss-self.entry_start
        result['pv_stored_delta_Wh']=float(self.battery[1]-self.battery_start[1]);result['initial_stored_delta_Wh']=float(self.battery[0]-self.battery_start[0])
        result['pv_stored_end_Wh']=float(self.battery[1]);result['initial_stored_end_Wh']=float(self.battery[0]);result['pv_useful_in_system_Wh']=result['pv_direct_Wh']+result['pv_battery_Wh']+result['pv_stored_delta_Wh']
        result['origin_balance_max_L']=float(np.max(abs(balance)));result['passive_tracer']=True
        return result
