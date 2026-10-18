"""B1: finite-volume DOE wall, dry ground column and a coupled zone balance.

Massless IDF insulation remains a pure thermal resistance. This is a stated
reference building/surface model, not an execution of EnergyPlus or a calibrated
site soil model. Temperatures are no longer set equal to ambient air.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from assembly import ROOT
from core import SIGMA
from weather import model_config,load_weather


def wall_grid(layers,cells_each=4):
    """Cell-centred mass layers plus zero-storage physical boundary faces."""
    cap=[0.];half=[];between=[];pending=0.;labels=['outside face']
    previous_half=None
    for layer in layers:
        if 'R_m2_K_W' in layer:pending+=layer['R_m2_K_W'];continue
        d=layer['thickness_m']/cells_each;k=layer['k_W_m_K']
        for _ in range(cells_each):
            hr=d/(2*k)
            between.append((previous_half or 0)+pending+hr);pending=0.;previous_half=hr
            cap.append(d*layer['rho_kg_m3']*layer['cp_J_kg_K']);labels.append(layer['name'])
    between.append((previous_half or 0)+pending);cap.append(0.);labels.append('inside face')
    n=len(cap);lap=np.zeros((n,n))
    for i,r in enumerate(between):
        if r<=0:raise ValueError('Invalid conduction path')
        g=1/r;lap[i,i]+=g;lap[i+1,i+1]+=g;lap[i,i+1]-=g;lap[i+1,i]-=g
    return np.array(cap),lap,labels


class SurfaceScene:
    def __init__(self,wall_cells_each=4,ground_cells=30,initial_C=20.,deep_ground_C=15.):
        spec=json.loads((ROOT/'config/stage-b/doe-wall.json').read_text());self.wall_spec=spec
        cw,lw,_=wall_grid(spec['layers'],wall_cells_each)
        # Dry homogeneous soil is a property scenario, not inferred geotechnics.
        ground=[{'name':'dry-ground reference','thickness_m':3.,'k_W_m_K':1.5,'rho_kg_m3':1600.,'cp_J_kg_K':800.}]
        cg,lg,_=wall_grid(ground,ground_cells)
        self.nw=len(cw);self.ng=len(cg);self.n=self.nw+self.ng+1
        self.area=49.911*11.8872;self.zone_capacity=49.911*33.2738*3*120000.
        self.cap=np.r_[cw*self.area,cg,self.zone_capacity]
        self.lap=np.zeros((self.n,self.n));self.lap[:self.nw,:self.nw]=lw*self.area;self.lap[self.nw:-1,self.nw:-1]=lg
        hin=7.7*self.area;self.lap[self.nw-1,self.nw-1]+=hin;self.lap[-1,-1]+=hin
        self.lap[self.nw-1,-1]-=hin;self.lap[-1,self.nw-1]-=hin
        self.t=np.full(self.n,float(initial_C));self.t[-1]=21.;self.deep=deep_ground_C
        self.eps=spec['layers'][0]['epsilon_LW'];self.alpha=spec['layers'][0]['alpha_SW']
        self.last=None

    def step(self,ta,solar_wall,ghi,lw,wind,dt=60,h_wall=None):
        old=self.t.copy();n=self.n;wi=0;gi=self.nw;bottom=n-2;zi=n-1
        # Explicit reference closures, replaced by supplied CFD coefficients
        # only when their temperature/flow validity envelope matches the case.
        hw=float(h_wall) if h_wall is not None else 4+3*max(wind,0)
        hg=4+3*max(wind,0)
        def terms(t):
            q=np.zeros(n);der=np.zeros((n,n))
            tw,tg,tz=t[wi],t[gi],t[zi]
            q[wi]=self.area*(self.alpha*solar_wall+hw*(ta-tw)+self.eps*(.5*lw+.5*SIGMA*(tg+273.15)**4-SIGMA*(tw+273.15)**4))
            der[wi,wi]=self.area*(-hw-4*self.eps*SIGMA*(tw+273.15)**3)
            der[wi,gi]=self.area*2*self.eps*SIGMA*(tg+273.15)**3
            q[gi]=.8*ghi+hg*(ta-tg)+.95*(lw-SIGMA*(tg+273.15)**4)
            der[gi,gi]=-hg-4*.95*SIGMA*(tg+273.15)**3
            # Deep-ground Dirichlet face, exact algebraic constraint below.
            hvac=np.clip(2000*(21-tz),-100000,100000)
            q[zi]=3000+hvac+500*(ta-tz)
            der[zi,zi]=-500-(2000 if abs(2000*(21-tz))<100000 else 0)
            return q,der
        t=old.copy()
        for iteration in range(20):
            q,der=terms(t);res=self.cap*(t-old)/dt+self.lap@t-q
            jac=np.diag(self.cap/dt)+self.lap-der
            res[bottom]=t[bottom]-self.deep;jac[bottom,:]=0;jac[bottom,bottom]=1
            delta=solve(jac,-res,assume_a='gen');t+=delta
            if np.max(np.abs(delta))<1e-8:break
        else:raise FloatingPointError('Surface/zone balance failed to converge')
        q,_=terms(t);conduction=self.lap@t
        # The imposed deep reservoir supplies the remaining bottom-face flux.
        q[bottom]=conduction[bottom]
        residual=float(np.sum(self.cap*(t-old))-dt*np.sum(q))
        self.t=t;self.last={'wall_C':float(t[wi]),'ground_C':float(t[gi]),'zone_C':float(t[zi]),
              'thermal_balance_residual_J':residual,'wall_h_W_m2_K':hw,'ground_h_W_m2_K':hg,'newton_iterations':iteration+1}
        return self.last


def simulate(city='rasht',year=2024,month=7,hours=48,warmup=48,dt=60,refinement=1,deep=15):
    cfg=model_config();f=load_weather(city,year,cfg)
    start=f.index.get_loc(pd.Timestamp(f'{year}-{month:02d}-15T00:00Z'));begin=start-warmup
    if begin<0 or start+hours>len(f):raise ValueError('Insufficient warm-up/weather')
    model=SurfaceScene(4*refinement,30*refinement,initial_C=float(f.iloc[begin].air_C),deep_ground_C=deep)
    out=[];max_res=0
    for i in range(begin,start+hours):
        w=f.iloc[i];sun=max(0,w.sun_z) if w.sun_y>0 else 0.
        poa=float(w.dni_W_m2)*sun+.5*float(w.dhi_W_m2)+.1*float(w.ghi_W_m2)
        steps=math.ceil(3600/dt)
        for _ in range(steps):
            r=model.step(float(w.air_C),poa,float(w.ghi_W_m2),float(w.lw_down_W_m2),float(w.wind_m_s),3600/steps)
            max_res=max(max_res,abs(r['thermal_balance_residual_J']))
        out.append({'time':str(f.index[i]),'scored':i>=start,'air_C':w.air_C,'ghi_W_m2':w.ghi_W_m2,**r})
    df=pd.DataFrame(out);path=ROOT/'results/stage-b/thermal';path.mkdir(parents=True,exist_ok=True)
    name=f'{city}-{year}-{month:02d}-dt{dt:g}-r{refinement}'
    if warmup!=48:name+=f'-wu{warmup}'
    if deep!=15:name+=f'-deep{deep:g}'
    df.to_csv(path/(name+'.csv.gz'),index=False,compression='gzip')
    meta={'city':city,'year':year,'month':month,'hours':hours,'warmup_hours':warmup,'dt_s':dt,
          'wall_cells_each_mass_layer':4*refinement,'ground_cells':30*refinement,'deep_ground_C':deep,
          'max_balance_residual_J':max_res,'wall_air_max_difference_K':float((df.wall_C-df.air_C).abs().max()),
          'ground_air_max_difference_K':float((df.ground_C-df.air_C).abs().max()),
          'source_wall':model.wall_spec,'CFD_convection_used':False,
          'limitations':['dry ground property/deep-temperature scenario','no ground evapotranspiration or urban canopy','zone RC/HVAC parameters are design scenarios, not a full EnergyPlus run','warm-up sufficiency not assumed']}
    (path/(name+'.json')).write_text(json.dumps(meta,indent=2));print(name,flush=True);return df,meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--cities',nargs='+',default=['rasht','singapore','reykjavik','london','la-paz'])
    p.add_argument('--year',type=int,default=2024);p.add_argument('--months',type=int,nargs='+',default=[1,7])
    p.add_argument('--hours',type=int,default=48);p.add_argument('--warmup',type=int,default=48)
    p.add_argument('--dt',type=float,default=60);p.add_argument('--refinement',type=int,default=1);p.add_argument('--deep-ground',type=float,default=15)
    a=p.parse_args()
    for city in a.cities:
        for month in a.months:simulate(city,a.year,month,a.hours,a.warmup,a.dt,a.refinement,a.deep_ground)


if __name__=='__main__':main()
