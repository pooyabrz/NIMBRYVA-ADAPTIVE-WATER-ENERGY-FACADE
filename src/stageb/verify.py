"""Independent analytic checks for Stage B; no claim of field validation."""
import argparse,json,math,tempfile
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp
from assembly import ROOT
from core import saturation_pressure_pa
from stageb.thermal import wall_grid,SurfaceScene
from stageb.surface import SurfaceFilm,LF,decode_energy,encode_energy,saturation_pa
from stageb.rain import terminal_speed,drag_factor,distribution,intersections,triangles,VelocityField,source_regions,load_case,advance_particles
from stageb.design import design


def slab(n,dt,end=600.):
    """Independent exact Fourier eigenmode for a homogeneous 0.1 m slab."""
    length=.1;k=1.5;rho=1600.;cp=800.;dx=length/n
    x=(np.arange(n)+.5)*dx;t=np.sin(np.pi*x/length);g=k/dx
    lap=np.diag(np.r_[3*g,np.full(n-2,2*g),3*g])+np.diag(np.full(n-1,-g),1)+np.diag(np.full(n-1,-g),-1)
    cap=rho*cp*dx;elapsed=0.
    while elapsed<end-1e-10:
        d=min(dt,end-elapsed);t=solve(lap+np.eye(n)*cap/d,t*cap/d,assume_a='pos');elapsed+=d
    exact=np.sin(np.pi*x/length)*math.exp(-k/(rho*cp)*(math.pi/length)**2*end)
    return float(np.sqrt(np.mean((t-exact)**2)))


def run():
    checks=[]
    def check(name,condition,**data):
        checks.append({'name':name,'pass':bool(condition),**data})
    layers=json.loads((ROOT/'config/stage-b/doe-wall.json').read_text())['layers']
    resist=sum(x.get('R_m2_K_W',x.get('thickness_m',0)/x.get('k_W_m_K',1)) for x in layers)
    cap,lap,_=wall_grid(layers,7);bc=lap.copy();rhs=np.zeros(len(cap))
    bc[0,:]=0;bc[0,0]=1;rhs[0]=30;bc[-1,:]=0;bc[-1,-1]=1;rhs[-1]=10
    t=solve(bc,rhs);flux=float((lap@t)[0]);check('layered steady conduction including massless R',abs(flux-20/resist)<1e-9,flux_W_m2=flux,analytic_W_m2=20/resist)
    check('massless resistance has no invented heat capacity',len(cap)==3*7+2 and cap[0]==cap[-1]==0)
    check('conduction matrix conserves heat',np.max(np.abs(lap.sum(1)))<1e-10)
    spatial=[slab(n,.25) for n in [10,20,40]]
    temporal=[slab(80,d) for d in [20,10,5]]
    check('slab Fourier mode spatial convergence',spatial[2]<spatial[1]<spatial[0],rms_K=spatial)
    check('slab Fourier mode backward Euler temporal convergence',temporal[2]<temporal[1]<temporal[0],rms_K=temporal)
    scene=SurfaceScene();s=scene.step(20,600,800,350,3,dt=60)
    check('wall and ground temperatures solved independently',abs(s['wall_C']-20)>1 and abs(s['ground_C']-20)>1,wall_C=s['wall_C'],ground_C=s['ground_C'])
    check('scene energy includes deep-ground boundary',abs(s['thermal_balance_residual_J'])<.01,residual_J=s['thermal_balance_residual_J'])
    ts=np.array([-30,-10,0,10,40.]);sat=np.array([saturation_pressure_pa(x) for x in ts])
    check('ice/water saturation matches independent project API',np.allclose(saturation_pa(ts),sat,rtol=1e-8))
    energies=np.array([-2000.,0.,.25*LF,.9*LF,LF+4180*3]);temps,liq,ice=decode_energy(energies,np.ones(5),0)
    check('enthalpy inverse preserves melting plateau',np.all(temps[1:4]==0) and np.allclose(liq[2:4],[.25,.9]) and np.allclose(liq+ice,1))
    check('latent heat ledger exact',np.isclose(encode_energy(np.array([0]),np.array([1.]),0,liquid_fraction=.4)[0],.4*LF))
    for melting in [False,True]:
        phase=SurfaceFilm(n=8,angle_deg=0,initial_C=0,initial_liquid_kg_m2=0 if melting else 1,initial_ice_kg_m2=1 if melting else 0)
        state=phase.advance(100,air_C=0,rh=0,hm_m_s=0,h_W_m2_K=0,epsilon=0,alpha=0,back_C=1 if melting else -1,back_G_W_m2_K=1000)
        expected=(100000/LF if melting else 1-100000/LF)*1.762*1.134
        check('analytic constant-flux '+('melting' if melting else 'freezing')+' plateau',abs(state['liquid_kg']-expected)<1e-10 and state['temperature_mean_C']==0)
    film=SurfaceFilm(n=8,angle_deg=50,initial_C=20,initial_liquid_kg_m2=.02)
    state=film.advance(.1,air_C=20,rh=0,hm_m_s=0,h_W_m2_K=0,epsilon=0,alpha=0,back_G_W_m2_K=0,max_dt=.1)
    exact=1000**2*9.80665*math.sin(math.radians(50))*(.02/1000)**3/(3*.001)*1.134*.1
    check('Nusselt smooth-film initial drainage flux analytic',abs(state['drain_kg']-exact)<1e-14,computed_kg=state['drain_kg'],analytic_kg=exact)
    dry=SurfaceFilm(n=8,initial_C=30);d=dry.advance(60,air_C=30,rh=0,hm_m_s=.02,lw_in_W_m2=500,back_C=30)
    check('evaporation cannot consume missing water',d['vapour_out_kg']==0 and abs(d['mass_residual_kg'])<1e-10)
    wet=SurfaceFilm(n=16,initial_C=20,initial_liquid_kg_m2=.1);w=wet.advance(120,air_C=20,rh=.2,hm_m_s=.01,back_C=20)
    check('wet film drainage and re-evaporation coexist',w['drain_kg']>0 and w['vapour_out_kg']>0)
    check('film transport conserves mass and enthalpy',abs(w['mass_residual_kg'])<1e-9 and abs(w['energy_residual_J'])<1e-5,energy_residual_J=w['energy_residual_J'])
    spatial=SurfaceFilm(n=16);forcing=np.linspace(0,.001,16)
    result=spatial.advance(20,air_C=20,rh=.7,rain_kg_m2_s=forcing)
    check('spatial rain handoff preserves integrated mass',abs(result['rain_kg']-forcing.mean()*20*1.762*1.134)<1e-12 and abs(result['mass_residual_kg'])<1e-9)
    snow=SurfaceFilm(n=8,initial_C=-10);s=snow.advance(60,air_C=-10,rh=.7,snow_swe_kg_m2_s=1e-5,back_C=-10)
    check('snow forcing is water-equivalent mass',abs(s['snow_swe_kg']-1e-5*60*1.762*1.134)<1e-12)
    check('cold surface retains solid inventory and closes balances',s['ice_plus_snow_kg']>0 and abs(s['mass_residual_kg'])<1e-9 and abs(s['energy_residual_J'])<1e-5)
    sub=SurfaceFilm(n=8,initial_C=-10,initial_ice_kg_m2=.01);s=sub.advance(30,air_C=-10,rh=.01,back_C=-10)
    check('sublimation is present',s['vapour_out_kg']>0 and s['energy_vapour_J']<0)
    rejected=False
    try:SurfaceFilm(n=8).advance(10,air_C=-5,rain_kg_m2_s=.001)
    except ValueError:rejected=True
    check('supercooled rain rejected explicitly',rejected)
    for diameter in [.00025,.001,.004]:
        v=terminal_speed(diameter);tau=1000*diameter**2/(18*1.81e-5)
        res=v*float(drag_factor(1.2*v*diameter/1.81e-5))/tau-9.80665*(1-1.2/1000)
        check(f'terminal drag balance D={diameter:g}m',abs(res)<1e-8,terminal_m_s=v)
    f,v,m=distribution(5,[.25,.5,.75,1,1.5,2,3,4]);check('DSD integrates to specified horizontal rain',abs(sum(f)-1)<1e-14 and np.all(f>0))
    diameter=.001;air=np.array([5.,0,0]);tau=1000*diameter**2/(18*1.81e-5)
    def rhs(_,state):
        slip=air-state[3:];re=1.2*np.linalg.norm(slip)*diameter/1.81e-5
        return np.r_[state[3:],slip*float(drag_factor(re))/tau+[0,0,-9.80665*(1-1.2/1000)]]
    reference=solve_ivp(rhs,[0,2],np.zeros(6),method='DOP853',rtol=1e-11,atol=1e-12).y[:,-1];errors=[]
    for dt in [.05,.025,.0125]:
        x=np.zeros((1,3));v=np.zeros((1,3))
        for i in range(round(2/dt)):x,v=advance_particles(x,v,np.array([air]),diameter,dt)
        errors.append(float(np.linalg.norm(x[0]-reference[:3])/np.linalg.norm(reference[:3])))
    check('nonlinear particle trajectory converges to independent DOP853 ODE',errors[2]<errors[1]<errors[0] and errors[2]<.005,relative_position_errors=errors)
    # Unit square plane at X=0, independent of CFD interpolation.
    tri=np.array([[0.,-1,-1],[0,1,-1],[0,0,1]])
    obj,distance,normal=intersections(np.array([[-1.,0,0],[-1,4,0]]),np.array([[1.,0,0],[1,4,0]]),[(1,tri),(2,tri+np.array([.5,0,0]))])
    check('thin surface first impact without tunnelling',obj.tolist()==[1,-1] and abs(distance[0]-.5)<1e-12)
    pts=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    vf=VelocityField(pts,np.tile([5.,0.,0.],(4,1)))
    check('uniform flow reproduced by interpolation',np.allclose(vf(np.array([[.2,.3,.4],[2,3,4]])),[5,0,0]))
    meta={'bounds_m':[[-60,-100,0],[200,100,80]],'z0_m':.03,'wind10_m_s':5.}
    def primitive(z):return (z+.03)*(math.log((z+.03)/.03)-1)
    fall=5/math.log(10.03/.03)*(primitive(80)-primitive(8))/60
    regions,_,_=source_regions(meta,np.array([0.,0,8]),fall,6)
    check('corner rain source retains upstream AND roof patches',{r[0] for r in regions}=={'upstream','top'})
    with tempfile.TemporaryDirectory(dir=ROOT/'results/stage-b') as td:
        path=Path(td);(path/'case.json').write_text('{}');rejected=False
        try:load_case(path)
        except ValueError as exc:rejected='execution' in str(exc).lower()
        check('CFD cannot be invented when no execution record exists',rejected)
        np.savez_compressed(path/'reference-field.npz',centres=np.zeros((4,3)),U=np.zeros((4,3)))
        (path/'compact-export.json').write_text(json.dumps({'NPZ_sha256':'wrong'}));rejected=False
        try:load_case(path,require_execution=False)
        except ValueError as exc:rejected='checksum' in str(exc).lower()
        check('compact CFD export corruption rejected',rejected)
    frame=design(32,42);check('DOE seed is reproducible',frame.equals(design(32,42)))
    check('DOE atmospheric constraints physical',np.all(frame.LW_down_W_m2>0) and np.all(frame.Yv>0) and np.all(frame.Yv<1))
    check('DOE flags buoyancy outside neutral-flow model',frame.neutral_RANS_scope.any() and (~frame.neutral_RANS_scope).any())
    stefan=ROOT/'results/stage-b/surface/stefan-convergence.json'
    if stefan.exists():
        data=json.loads(stefan.read_text());err=[r['relative_front_error'] for r in data]
        check('executed Stefan front converges to analytic Neumann solution',len(err)>=3 and err[-1]<.001 and all(x>y for x,y in zip(err,err[1:])),relative_errors=err)
    out=ROOT/'results/stage-b/verification.json';out.parent.mkdir(parents=True,exist_ok=True)
    result={'checks':checks,'passed':sum(x['pass'] for x in checks),'total':len(checks),'experimental_validation':False}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['passed']!=result['total']:raise SystemExit(1)
    return result


def main():run()
if __name__=='__main__':main()
