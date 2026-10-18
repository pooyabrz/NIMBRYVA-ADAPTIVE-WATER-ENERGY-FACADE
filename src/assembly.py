"""Finite-volume laminate and explicit parallel bridge paths (SI).

The commercial PV internal BOM is not published. Layers marked proxy remain
epistemic inputs, not measured properties of the Trina module.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'cassette-v2.0'


def reference():
    return json.loads((ROOT/'config/evidence/reference.json').read_text())


def parameters():
    return {r['id']:r['nominal'] for r in reference()['parameters']}


def bridge_conductance(paths):
    """Parallel paths; segments in series, contact resistivity on each path.

    This is a 1D path approximation: no 3D spreading/constriction solution.
    Contact resistivity is K m2/W, not K/W.
    """
    out=[]
    for path in paths:
        n=path['count'];a=path['area_each_m2']
        if n<0 or a<=0: raise ValueError('Invalid bridge count/area')
        resistance=sum(s['length_m']/(s['k_W_m_K']*a) for s in path['segments'])
        resistance+=path.get('contact_R_m2_K_W',0)/a
        if resistance<=0:raise ValueError('Invalid bridge resistance')
        out.append({'name':path['name'],'G_W_K':n/resistance})
    return sum(x['G_W_K'] for x in out),out


def compile_assembly(cfg):
    p=parameters();a=cfg['area_m2'];ad=cfg['dew_area_m2']
    layers=[
        ('PV front glass',p['pv_glass_front_m'],p['pv_glass_rho'],p['pv_glass_cp'],p['pv_glass_k'],'datasheet thickness / generic glass proxy'),
        ('PV front encapsulant',p['pv_eva_each_m'],p['pv_eva_rho'],p['pv_eva_cp'],p['pv_eva_k'],'EVA proxy; actual EVA/POE grade unknown'),
        ('PV Si cell',p['pv_silicon_m'],p['pv_silicon_rho'],p['pv_silicon_cp'],p['pv_silicon_k'],'unknown wafer thickness / literature proxy'),
        ('PV back encapsulant',p['pv_eva_each_m'],p['pv_eva_rho'],p['pv_eva_cp'],p['pv_eva_k'],'EVA proxy'),
        ('PV back glass',p['pv_glass_back_m'],p['pv_glass_rho'],p['pv_glass_cp'],p['pv_glass_k'],'datasheet thickness / generic glass proxy'),
        ('XPS',p['insulation_m'],p['xps_rho'],p['xps_cp'],p['xps_k'],'material property proxy at proposed thickness')]
    for i,l in enumerate(layers):
        name,d,rho,cp,k,status=l
        layers[i]={'name':name,'d_m':d,'rho_kg_m3':rho,'cp_J_kg_K':cp,'k_W_m_K':k,
                   'area_m2':a,'C_J_K':a*d*rho*cp*(cfg.get('pv_capacity_multiplier',1) if i<5 else 1),
                   'R_K_W':d/(k*a),'evidence':status}
    # Very thin emitter is a single isothermal node. Through-layer resistance
    # is reported; air-side resistance dominates at the resolved time scale.
    cd=ad*(p['pdms_m']*p['pdms_rho']*p['pdms_cp']+p['silica_m']*p['silica_rho']*p['silica_cp'])
    layers.append({'name':'Dew emitter/condenser','C_J_K':cd,'area_m2':ad,
                   'R_K_W':(p['pdms_m']/p['pdms_k']+p['silica_m']/p['silica_k']+p['cnf_ptfe_m']/p['cnf_ptfe_k'])/ad,
                   'evidence':'Haechler geometry; thermal-property proxies; nanolayer storage neglected'})
    # Account for otherwise unallocated module mass; generic aluminium cp is
    # a declared proxy, not a measured transient module heat capacity.
    proxy_mass=sum(l['d_m']*l['rho_kg_m3']*a for l in layers[:5])
    frame_mass=max(0,p['pv_module_mass_kg']-proxy_mass)+p['extra_support_mass_kg']
    layers.append({'name':'Frame/support','C_J_K':frame_mass*cfg['frame_cp_J_kg_K'],
                   'area_m2':0,'R_K_W':0,'evidence':'unallocated mass with aluminium heat-capacity proxy'})
    cap=np.array([l['C_J_K'] for l in layers]);lap=np.zeros((8,8))
    def edge(i,j,g):
        lap[i,i]+=g;lap[j,j]+=g;lap[i,j]-=g;lap[j,i]-=g
    for i in range(5):
        contact=cfg.get('laminate_contact_R_m2_K_W',0)/a if i<4 else cfg['pv_xps_contact_R_m2_K_W']/a
        edge(i,i+1,1/(.5*layers[i]['R_K_W']+.5*layers[i+1]['R_K_W']+contact))
    edge(4,7,cfg['frame_pv_contact_W_K'])
    gb,paths=bridge_conductance(cfg['bridge_paths'])
    gb*=cfg.get('bridge_multiplier',1)
    edge(7,6,gb)
    mass=p['pv_module_mass_kg']+ad*(p['silica_m']*p['silica_rho']+p['pdms_m']*p['pdms_rho'])+a*p['insulation_m']*p['xps_rho']+p['extra_support_mass_kg']
    return {'layers':layers,'capacity_J_K':cap,'laplacian_W_K':lap,'bridge_W_K':gb,
            'bridge_paths':paths,'moving_mass_kg':mass,'PV_proxy_mass_kg':proxy_mass,
            'unallocated_module_mass_kg':p['pv_module_mass_kg']-proxy_mass,
            'assembly_depth_m':p['pv_frame_thickness_m']+p['insulation_m']+p['vent_gap_m']+p['silica_m']+p['pdms_m']+p['cnf_ptfe_m']+p['silver_m']+p['chromium_adhesion_m']+p['chromium_protection_m']}


def validate_config(c):
    if c.get('model_version')!=MODEL_VERSION:
        raise ValueError('Old two-node configuration cannot run the cassette solver; use legacy_v1.')
    if c['area_m2']<=0 or not 0<c['dew_area_m2']<=c['area_m2']:
        raise ValueError('Invalid PV/dew areas')
    if not math.isclose(c['area_m2'],c['panel_width_m']*c['rotation_chord_m'],rel_tol=1e-9):
        raise ValueError('Area and dimensions disagree')
    if min(c['integration_dt_s'],c['battery_capacity_Wh'])<=0:raise ValueError('Invalid dt/battery')
    a=compile_assembly(c)
    if c['wall_standoff_m']-math.hypot(c['rotation_chord_m']/2,a['assembly_depth_m']/2)<c['clearance_allowance_m']:
        raise ValueError('Full rotation sweep intersects required wall clearance')
    if c['mounting_height_m']-math.hypot(c['rotation_chord_m']/2,a['assembly_depth_m']/2)<0:
        raise ValueError('Sweep intersects ground')
    for key in ['rain_interception_efficiency','jump_fraction','collector_efficiency','coating_condition','charge_efficiency','discharge_efficiency','dc_efficiency']:
        if not 0<=c[key]<=1:raise ValueError('Invalid fraction '+key)
    return a
