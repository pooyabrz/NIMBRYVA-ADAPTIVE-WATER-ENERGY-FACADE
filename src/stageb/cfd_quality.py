"""Geometry, residual, scalar-bound and empty-ABL gates beyond checkMesh."""
import argparse,json,math,re
from pathlib import Path
import numpy as np


def payload(path):
    text=Path(path).read_text()
    if re.search(r'format\s+binary',text):raise ValueError('ASCII OpenFOAM output required')
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    text=text.split('}',1)[1]
    m=re.fullmatch(r'\s*(\d+)\s*\((.*)\)\s*;?\s*',text,re.S)
    if not m:raise ValueError('Unsupported OpenFOAM mesh list: '+str(path))
    return int(m.group(1)),m.group(2)


def mesh(case):
    poly=Path(case)/'constant/polyMesh'
    npnt,s=payload(poly/'points');points=np.fromstring(s.replace('(',' ').replace(')',' '),sep=' ').reshape(-1,3)
    if len(points)!=npnt:raise ValueError('Point count mismatch')
    nf,s=payload(poly/'faces');faces=[np.fromstring(x,sep=' ',dtype=int) for _,x in re.findall(r'(\d+)\s*\(([^()]*)\)',s)]
    if len(faces)!=nf:raise ValueError('Face count mismatch')
    no,s=payload(poly/'owner');owner=np.fromstring(s,sep=' ',dtype=int)
    n,s=payload(poly/'boundary');patch={}
    for name,body in re.findall(r'(\w+)\s*\{([^{}]*)\}',s):
        start=re.search(r'startFace\s+(\d+)',body);num=re.search(r'nFaces\s+(\d+)',body)
        if start and num:patch[name]={'start':int(start.group(1)),'n':int(num.group(1))}
    return points,faces,owner,patch


def face_geometry(points,faces):
    centres=[];vectors=[]
    for ids in faces:
        p=points[ids];centres.append(p.mean(0));vectors.append(sum((np.cross(p[i]-p[0],p[i+1]-p[0])/2 for i in range(1,len(p)-1)),start=np.zeros(3)))
    return np.array(centres),np.array(vectors)


def boundary_values(path,name,n):
    text=Path(path).read_text();part=text.split('boundaryField',1)[1]
    m=re.search(r'\b'+re.escape(name)+r'\s*\{([^{}]*)\}',part,re.S)
    if not m:raise ValueError('Boundary field absent: '+name)
    v=re.search(r'value\s+uniform\s+([\deE+.-]+)\s*;',m.group(1))
    if v:return np.full(n,float(v.group(1)))
    v=re.search(r'value\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;',m.group(1),re.S)
    if not v:raise ValueError('Unsupported scalar boundary value')
    out=np.fromstring(v.group(1),sep=' ')
    if len(out)!=n:raise ValueError('Boundary count mismatch')
    return out


def quantities(case,centres,u,extra,meta,latest):
    from stageb.rain import VelocityField
    target=np.array(meta['panel_vertices_m']).mean(0)
    probes=target+np.array([[-5,0,0],[-2,0,0],[-.5,0,0],[0,0,2],[0,0,-2]])
    vf=VelocityField(centres,u)
    out={'probe_points_m':probes.tolist(),'probe_U_m_s':vf(probes).tolist()}
    if meta['empty_domain']:return out
    pts,faces,owner,patch=mesh(case);p=patch['cassette'];ids=np.arange(p['start'],p['start']+p['n'])
    fc,sf=face_geometry(pts,[faces[i] for i in ids]);area=np.linalg.norm(sf,axis=1);norm=sf/area[:,None]
    cell=owner[ids];dn=np.abs(np.sum((centres[cell]-fc)*norm,axis=1));dn=np.maximum(dn,1e-8)
    nut=boundary_values(latest/'nut','cassette',p['n'])
    ht=(1.5e-5/meta['Pr']+nut/meta['Prt'])*(extra['T'][cell]-meta['T_panel_K'])/dn*1.2*1006/(meta['T_ambient_K']-meta['T_panel_K'])
    hm=(1.5e-5/meta['Sc_molecular']+nut/meta['Sct'])*(extra['Yv'][cell]-meta['Yv_panel'])/dn/(meta['Yv_inlet']-meta['Yv_panel'])
    pressure_forces=1.2*extra['p'][cell,None]*sf
    force=pressure_forces.sum(0);moment=np.cross(fc-target,pressure_forces).sum(0)
    out.update(pressure_only_force_N=force.tolist(),pressure_only_moment_Nm=moment.tolist(),pressure_only_rotation_axis_moment_Nm=float(moment[1]),
       diagnostic_h_W_m2_K=float(ht@area/area.sum()),diagnostic_hm_m_s=float(hm@area/area.sum()),
       diagnostic_h_not_qualified=True,flux_gradient='owner-to-face normal distance; omits nonorthogonal correction',
       pressure_only_load_not_actuator_rating=True,load_omissions=['viscous traction','gust/transient load','anchor/support loads'],
       scalar_wall_limitation='no thermal/mass wall functions and no resolved prism boundary layers')
    return out


def export_compact(case,centres,u,extra,meta,latest):
    import hashlib
    from stageb.rain import case_signature
    case=Path(case);archive=case/'reference-field.npz'
    np.savez_compressed(archive,centres=centres,U=u,**extra)
    provenance={'schema':'stage-b-compact-cfd/v1','source_time':latest.name,'cells':len(centres),
       'U_sha256':hashlib.sha256((latest/'U').read_bytes()).hexdigest(),
       'case_definition_sha256':case_signature(meta),
       'NPZ_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
       'coordinates':'OpenFOAM X downwind, Y lateral, Z up, m','T':'K','Yv':'water-vapour mass fraction, kg/kg moist air',
       'p':'kinematic pressure m2/s2, gauge','contains':'actual internal cell fields, lossless float64 values parsed from solver ASCII output',
       'rebuild_mesh':'rerun stage-b cfd generate/run with archived case.json and system/constant inputs'}
    (case/'compact-export.json').write_text(json.dumps(provenance,indent=2))


def geometry_gate(case):
    case=Path(case);meta=json.loads((case/'case.json').read_text())
    points,faces,owner,patch=mesh(case)
    out={'patch_faces':{k:v['n'] for k,v in patch.items()},'passed':True}
    if not meta['empty_domain']:
        if any(p not in patch or patch[p]['n']==0 for p in ['building','cassette']):
            return {**out,'passed':False,'reason':'Required solid patch absent/empty: thin geometry was lost'}
        p=patch['cassette'];fc,sf=face_geometry(points,faces[p['start']:p['start']+p['n']]);area=np.linalg.norm(sf,axis=1).sum()
        v=np.array(meta['panel_vertices_m']);lengths=[np.linalg.norm(v[i]-v[0]) for i in [1,3,4]]
        exact=2*(lengths[0]*lengths[1]+lengths[1]*lengths[2]+lengths[2]*lengths[0])
        err=abs(area/exact-1);bounderr=float(np.max(np.abs(np.r_[fc.min(0)-v.min(0),fc.max(0)-v.max(0)])))
        out.update(cassette_area_m2=float(area),physical_closed_box_area_m2=float(exact),relative_area_error=float(err),face_centre_bounds_error_m=bounderr,
                   geometry_tolerance_for_screening=.15,geometry_qualified_to_two_percent=bool(err<.02))
        out['passed']=bool(err<.15 and bounderr<.15)
        if not out['passed']:out['reason']='Cassette geometry area/bounds differ excessively from physical STL'
    return out


def analyze(case):
    from stageb.rain import load_case
    case=Path(case);centres,u,extra,meta,latest=load_case(case,require_execution=False)
    geometry=geometry_gate(case);text=(case/'log.simpleFoam').read_text();last=re.split(r'^Time =',text,flags=re.M)[-1]
    residual={}
    for name,value in re.findall(r'Solving for (\w+), Initial residual = ([\deE+.-]+)',last):residual[name]=float(value)
    continuity=re.findall(r'time step continuity errors : sum local = ([\deE+.-]+)',text)
    finite=bool(np.isfinite(u).all() and all(np.isfinite(x).all() for x in extra.values()))
    t=extra.get('T',np.array([np.nan]));y=extra.get('Yv',np.array([np.nan]))
    tbound=bool(np.min(t)>=min(meta['T_panel_K'],meta['T_ambient_K'])-1e-3 and np.max(t)<=max(meta['T_wall_K'],meta['T_ambient_K'])+1e-3)
    ybound=bool(np.min(y)>=min(meta['Yv_panel'],meta['Yv_inlet'])-1e-7 and np.max(y)<=max(meta['Yv_panel'],meta['Yv_inlet'])+1e-7)
    dissipation='omega' if meta.get('RAS_model')=='kOmegaSST' else 'epsilon'
    flow_ok=all(residual.get(n,np.inf)<=1e-5 for n in ['p','Ux','Uy','Uz','k',dissipation])
    scalar_ok=all(residual.get(n,np.inf)<=1e-6 for n in ['T','Yv']) and tbound and ybound
    constant_preserved=bool(meta['empty_domain'] and np.max(np.abs(t-meta['T_ambient_K']))<1e-8 and np.max(np.abs(y-meta['Yv_inlet']))<1e-12)
    if constant_preserved:scalar_ok=True
    checks=(case/'log.checkMesh').read_text();mesh_ok='Mesh OK.' in checks
    # Neutral RANS check for the largest imposed temperature contrast.
    ri=9.80665*11.8872*max(abs(meta['T_wall_K']-meta['T_ambient_K']),abs(meta['T_panel_K']-meta['T_ambient_K']))/(meta['T_ambient_K']*meta['wind10_m_s']**2)
    if meta['empty_domain']:ri=0.
    result={'case':case.name,'time':latest.name,'cells':len(centres),'geometry':geometry,'standard_checkMesh_passed':mesh_ok,
       'last_initial_residuals':residual,'continuity_sum_local_last':float(continuity[-1]) if continuity else None,
       'finite_fields':finite,'flow_residual_gate_passed':bool(flow_ok),'scalar_residual_and_bounds_gate_passed':bool(scalar_ok),
       'empty_constant_scalar_preservation':constant_preserved,'scalar_residual_note':'Zero-contrast normalized residual is not used if the constant solution is preserved to the stated tolerance' if constant_preserved else None,
       'T_range_K':[float(np.min(t)),float(np.max(t))],'Yv_range':[float(np.min(y)),float(np.max(y))],
       'U_max_m_s':float(np.linalg.norm(u,axis=1).max()),'bulk_Ri':ri,'neutral_RANS_Ri_below_0_1':ri<.1,
       'flow_available_for_exploratory_particle_tracking':bool(mesh_ok and geometry['passed'] and finite and flow_ok),
       'qualified_scalar_transfer_available':False,
       'qualified_for_yield_surrogate_training':False,
       'why_not_qualified':['scalar wall treatment has no resolved/prismatic boundary layers','independent physical validation absent','grid convergence must be assessed on quantities of interest'],
       'OpenFOAM_converged_message':'solution converged' in text.lower()}
    if geometry['passed']:
        result['quantities_of_interest']=quantities(case,centres,u,extra,meta,latest)
    if meta['empty_domain']:
        z=centres[:,2];us=meta['wind10_m_s']*.41/math.log((10+meta['z0_m'])/meta['z0_m'])
        exact=us/.41*np.log((z+meta['z0_m'])/meta['z0_m']);sel=(z>=2)&(z<=30)
        result['empty_ABL_relative_L2_U']=float(np.linalg.norm(u[sel,0]-exact[sel])/np.linalg.norm(exact[sel]))
        result['empty_ABL_max_vertical_velocity_m_s']=float(np.max(np.abs(u[sel,2])))
    out=case/'quality.json';out.write_text(json.dumps(result,indent=2))
    meta['case_generated_only']=False;meta['not_a_converged_solution']=not bool(flow_ok and scalar_ok)
    meta['quality_report']='quality.json';(case/'case.json').write_text(json.dumps(meta,indent=2))
    export_compact(case,centres,u,extra,meta,latest)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.case),indent=2))


if __name__=='__main__':main()
