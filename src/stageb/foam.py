"""Pinned OpenCFD 2312 case generation and auditable local execution.

CFD coordinates: X downwind, Y lateral, Z up. Facade coordinates of v2:
(x,y,z) -> (-z-standoff,x,y). Rotation of geometry handles wind azimuth.
No bundled solver binary, invented velocity field, or implicit Docker usage.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,re,shutil,subprocess,time
import numpy as np
from assembly import ROOT,compile_assembly
from weather import model_config
from geometry import panel_frame

PIN='2312';PATCH='260127'
RESULTS=ROOT/'results/stage-b'


def header(name,cls='dictionary'):
    return f'FoamFile\n{{ version 2.0; format ascii; class {cls}; object {name}; }}\n'


def box_vertices(lo,hi):
    return np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
       [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])


def write_stl(path,vertices,name):
    quads=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    lines=['solid '+name]
    for q in quads:
        for ids in [(q[0],q[1],q[2]),(q[0],q[2],q[3])]:
            a,b,c=vertices[list(ids)];n=np.cross(b-a,c-a);n/=np.linalg.norm(n)
            if np.dot(n,(a+b+c)/3-vertices.mean(0))<0:b,c=c,b;n=-n
            lines+=['facet normal '+' '.join(map(str,n)),'outer loop']
            lines+=['vertex '+' '.join(map(str,x)) for x in (a,b,c)];lines+=['endloop','endfacet']
    path.write_text('\n'.join(lines+['endsolid '+name])+'\n')


def generate(path,level='coarse',wind=5.,z0=.03,angle=50.,direction=0.,iterations=800,empty=False):
    """Neutral RANS + independent passive thermal/vapour reference scalars.

    Scalars do not feed buoyancy/latent heat back into RANS. Richardson number
    and this one-way coupling limitation are explicitly recorded in metadata.
    """
    c=model_config();a=compile_assembly(c);h=c['building_height_m'];width=c['building_width_m'];depth=33.2738
    if wind<=0 or z0<=0:raise ValueError('Positive wind and aerodynamic roughness required')
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    if (path/'execution.json').exists() or (path/'compact-export.json').exists():
        raise FileExistsError('This case has execution evidence; choose a new case directory to preserve provenance')
    for d in ['0','constant/triSurface','system']:(path/d).mkdir(parents=True,exist_ok=True)
    lo=np.array([-5*h,-width/2-10*h,0]);hi=np.array([depth+15*h,width/2+10*h,7*h])
    nx,ny,nz={'coarse':(32,32,20),'medium':(44,44,28),'fine':(60,60,38)}[level]
    verts=box_vertices(lo,hi);vtext='\n'.join('('+' '.join(map(str,v))+')' for v in verts)
    block=header('blockMeshDict')+f'''scale 1;
vertices ( {vtext} );
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
 inlet {{type patch; faces ((0 4 7 3));}}
 outlet {{type patch; faces ((1 2 6 5));}}
 side1 {{type symmetryPlane; faces ((0 1 5 4));}}
 side2 {{type symmetryPlane; faces ((3 7 6 2));}}
 ground {{type wall; faces ((0 3 2 1));}}
 top {{type symmetryPlane; faces ((4 5 6 7));}}
);
mergePatchPairs ();
'''
    (path/'system/blockMeshDict').write_text(block)
    # Rotate building and panel about the vertical through the building centre.
    theta=np.radians(-direction);rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    origin=np.array([depth/2,0,0.])
    building=box_vertices([0,-width/2,0],[depth,width/2,h]);building=(building-origin)@rot.T+origin
    u,v,n=panel_frame(angle);centre=np.array([0.,c['mounting_height_m'],0.])
    points=[]
    for sx,sy,sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
        p=centre+sx*c['panel_width_m']/2*u+sy*c['rotation_chord_m']/2*v+sz*a['assembly_depth_m']/2*n
        points.append([-p[2]-c['wall_standoff_m'],p[0],p[1]])
    panel=(np.array(points)-origin)@rot.T+origin
    write_stl(path/'constant/triSurface/building.stl',building,'building')
    write_stl(path/'constant/triSurface/cassette.stl',panel,'cassette')
    # Building mesh and cassette mesh have separate scale requirements.
    # Surfaces stay at physical scale, never thickened to make meshing pass.
    panel_level={'coarse':7,'medium':7,'fine':7}[level]
    region_lo=panel.min(0)-.4;region_hi=panel.max(0)+.4
    geom='building.stl {type triSurfaceMesh; name building;} cassette.stl {type triSurfaceMesh; name cassette;} '
    geom+='cassetteRegion {type searchableBox; min ('+' '.join(map(str,region_lo))+'); max ('+' '.join(map(str,region_hi))+');}'
    refinement='building {level (2 2); patchInfo {type wall;}} cassette {level ('+str(panel_level)+' '+str(panel_level)+'); patchInfo {type wall;}}'
    snappy=header('snappyHexMeshDict')+f'''castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls
{{
 maxLocalCells 1800000; maxGlobalCells 2500000; minRefinementCells 0;
 maxLoadUnbalance 0.1; nCellsBetweenLevels 3; features ({{file "cassette.eMesh"; level {panel_level};}});
 refinementSurfaces {{ {refinement} }}
 resolveFeatureAngle 30;
 refinementRegions {{cassetteRegion {{mode inside; levels ((1e15 4));}}}}
 locationInMesh ({lo[0]+1} 0 {h*2});
 allowFreeStandingZoneFaces true;
}}
snapControls {{nSmoothPatch 5; tolerance 1; nSolveIter 100; nRelaxIter 8; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;}}
addLayersControls {{relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}}
meshQualityControls
{{maxNonOrtho 60; maxBoundarySkewness 4; maxInternalSkewness 2; maxConcave 30; minVol 1e-15; minTetQuality 1e-12; minArea -1; minTwist 0.05; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.03; minTriangleTwist -1; nSmoothScale 5; errorReduction 0.75;}}
mergeTolerance 1e-6;
'''
    (path/'system/snappyHexMeshDict').write_text(snappy)
    (path/'system/surfaceFeatureExtractDict').write_text(header('surfaceFeatureExtractDict')+'''cassette.stl
{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {includedAngle 150;} writeObj no; }
''')
    abl=f'flowDir (1 0 0); zDir (0 0 1); Uref {wind}; Zref 10; z0 uniform {z0}; d uniform 0; kappa 0.41; Cmu 0.09; C1 0; C2 1;'
    us=wind*.41/math.log((10+z0)/z0);k=us**2/math.sqrt(.09);eps=us**3/(.41*(10+z0))
    patches=['inlet','outlet','side1','side2','ground','top']+([] if empty else ['building','cassette'])
    def field(name,dimensions,internal,bcs,vector=False):
        t=header(name,'volVectorField' if vector else 'volScalarField')+f'dimensions {dimensions};\ninternalField uniform {internal};\nboundaryField\n{{\n'
        for patch in patches:t+=patch+' { '+bcs(patch)+' }\n'
        (path/'0'/name).write_text(t+'}\n')
    field('U','[0 1 -1 0 0 0 0]',f'({wind} 0 0)',lambda p:
       ('type atmBoundaryLayerInletVelocity; '+abl+f' value uniform ({wind} 0 0);') if p=='inlet' else
       ('type pressureInletOutletVelocity; value uniform (0 0 0);') if p=='outlet' else
       'type symmetryPlane;' if p in ['side1','side2','top'] else 'type noSlip;',True)
    field('p','[0 2 -2 0 0 0 0]','0',lambda p:'type fixedValue; value uniform 0;' if p=='outlet' else 'type symmetryPlane;' if p in ['side1','side2','top'] else 'type zeroGradient;')
    for name,value,bc in [('k',k,'atmBoundaryLayerInletK'),('epsilon',eps,'atmBoundaryLayerInletEpsilon')]:
        field(name,'[0 2 -2 0 0 0 0]' if name=='k' else '[0 2 -3 0 0 0 0]',str(value),lambda p,n=name,val=value,bc=bc:
          ('type '+bc+'; '+abl+f' value uniform {val};') if p=='inlet' else
          f'type inletOutlet; inletValue uniform {val}; value uniform {val};' if p=='outlet' else
          'type symmetryPlane;' if p in ['side1','side2','top'] else
          ('type kqRWallFunction; value uniform '+str(val)+';') if n=='k' else 'type epsilonWallFunction; value uniform '+str(val)+';')
    field('nut','[0 2 -1 0 0 0 0]','0',lambda p:
       f'type atmNutkWallFunction; z0 uniform {z0}; value uniform 0;' if p=='ground' else
       'type nutkWallFunction; value uniform 0;' if p in ['building','cassette'] else
       'type symmetryPlane;' if p in ['side1','side2','top'] else 'type calculated; value uniform 0;')
    for name,ambient,surface,dimensions in [('T',293.15,288.15,'[0 0 0 1 0 0 0]'),('Yv',.0102,.0076,'[0 0 0 0 0 0 0]')]:
        field(name,dimensions,str(ambient),lambda p,n=name,ambient=ambient,surface=surface:
          f'type fixedValue; value uniform {ambient};' if p=='inlet' else
          f'type inletOutlet; inletValue uniform {ambient}; value uniform {ambient};' if p=='outlet' else
          'type symmetryPlane;' if p in ['side1','side2','top'] else
          f'type fixedValue; value uniform {surface};' if p=='cassette' else
          'type fixedValue; value uniform 295.15;' if p=='building' and n=='T' else 'type zeroGradient;')
    (path/'constant/transportProperties').write_text(header('transportProperties')+'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1.5e-5;\n')
    sigma=.41**2/((1.92-1.44)*math.sqrt(.09))
    (path/'constant/turbulenceProperties').write_text(header('turbulenceProperties')+f'simulationType RAS; RAS {{RASModel kEpsilon; turbulence on; printCoeffs on; kEpsilonCoeffs {{Cmu 0.09; C1 1.44; C2 1.92; sigmak 1; sigmaEps {sigma};}}}}\n')
    control=header('controlDict')+f'''application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {iterations}; deltaT 1;
writeControl timeStep; writeInterval {iterations}; purgeWrite 1; writeFormat ascii; writePrecision 10; writeCompression off; timeFormat general; timePrecision 8; runTimeModifiable false;
libs ("libatmosphericModels.so");
functions
{{
 heat {{type scalarTransport; libs (solverFunctionObjects); field T; alphaD 1.408450704; alphaDt 1.176470588; nCorr 1; resetOnStartUp false; executeControl timeStep; executeInterval 1; writeControl writeTime;}}
 vapour {{type scalarTransport; libs (solverFunctionObjects); field Yv; alphaD 1.666666667; alphaDt 1.428571429; nCorr 3; resetOnStartUp false; executeControl timeStep; executeInterval 1; writeControl writeTime;}}
 yplus {{type yPlus; libs (fieldFunctionObjects); executeControl writeTime; writeControl writeTime;}}
}}
'''
    (path/'system/controlDict').write_text(control)
    (path/'system/fvSchemes').write_text(header('fvSchemes')+'''ddtSchemes {default steadyState;}
gradSchemes {default cellLimited Gauss linear 1;}
divSchemes {default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind; div(phi,T) bounded Gauss upwind; div(phi,Yv) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear;}
laplacianSchemes {default Gauss linear limited 0.5;}
interpolationSchemes {default linear;}
snGradSchemes {default limited 0.5;}
fluxRequired {default no; p;}
''')
    (path/'system/fvSolution').write_text(header('fvSolution')+'''solvers
{ p {solver GAMG; tolerance 1e-8; relTol 0.05; smoother GaussSeidel;}
  "(U|k|epsilon)" {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.05;}
  "(T|Yv)" {solver PBiCGStab; preconditioner DILU; tolerance 1e-11; relTol 0; maxIter 500;}}
SIMPLE {nNonOrthogonalCorrectors 2; consistent no; residualControl {p 1e-5; U 1e-6; k 1e-6; epsilon 1e-6; T 1e-6; Yv 1e-6;}}
relaxationFactors {fields {p 0.3;} equations {U 0.6; k 0.5; epsilon 0.5; T 0.3; Yv 0.3;}}
''')
    meta={'schema':'stage-b-cfd/v1','openfoam_family':'OpenCFD','release':PIN,'tested_patch':PATCH,
      'mesh_level':level,'wind10_m_s':wind,'z0_m':z0,'wind_direction_deg':direction,'panel_angle_deg':angle,
      'empty_domain':empty,'bounds_m':[lo.tolist(),hi.tolist()],'panel_vertices_m':panel.tolist(),'building_vertices_m':building.tolist(),
      'area_m2':c['area_m2'],'blockage_fraction':width*h/((hi[1]-lo[1])*hi[2]),
      'T_ambient_K':293.15,'T_wall_K':295.15,'T_panel_K':288.15,'Yv_inlet':.0102,'Yv_panel':.0076,
      'Pr':.71,'Prt':.85,'Sc_molecular':.6,'Sct':.7,
      'limitations':['steady neutral RANS','passive T/Yv: no buoyancy or condensation latent feedback into CFD','no prismatic scalar boundary layers','no particle turbulence dispersion','no measured facade validation yet'],
      'not_a_converged_solution':True,'case_generated_only':True}
    (path/'case.json').write_text(json.dumps(meta,indent=2));return meta


def environment(prefix=None):
    env=os.environ.copy()
    if prefix:
        root=Path(prefix).resolve();plat=root/'platforms/linux64GccDPInt32Opt'
        env['WM_PROJECT_DIR']=str(root);env['FOAM_ETC']=str(root/'etc');env['WM_PROJECT_VERSION']=PIN
        env['PATH']=str(plat/'bin')+os.pathsep+env.get('PATH','')
        lib=[plat/'lib/dummy',plat/'lib']
        ancestor=root.parents[3] if len(root.parents)>3 else root
        extra=ancestor/'usr/lib/x86_64-linux-gnu'
        if extra.exists():lib.append(extra)
        env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,lib))+os.pathsep+env.get('LD_LIBRARY_PATH','')
    return env


def run(case,prefix=None,mesh_only=False):
    case=Path(case).resolve();env=environment(prefix);metadata=json.loads((case/'case.json').read_text())
    exe=shutil.which('simpleFoam',path=env['PATH'])
    if not exe:raise RuntimeError('OpenCFD OpenFOAM 2312 is required; case generated, no synthetic CFD substituted')
    version=subprocess.run([exe,'-help'],env=env,capture_output=True,text=True,check=True).stdout
    if 'OpenFOAM-2312' not in version:raise RuntimeError('Wrong OpenFOAM branch/release; use OpenCFD 2312')
    if f'patch={PATCH}' not in version:raise RuntimeError('Untested patch: reference lock requires OpenCFD 2312 patch '+PATCH)
    records=[]
    commands=[['blockMesh']]+([] if metadata['empty_domain'] else [['surfaceFeatureExtract'],['snappyHexMesh','-overwrite']])+[['checkMesh']]
    if not mesh_only:commands+=[['simpleFoam'],['postProcess','-func','writeCellCentres','-latestTime']]
    for cmd in commands:
        log=case/('log.'+cmd[0]);start=time.time()
        with log.open('w') as out:
            proc=subprocess.run([*cmd,'-case',str(case)],env=env,stdout=out,stderr=subprocess.STDOUT)
        records.append({'command':cmd,'returncode':proc.returncode,'seconds':time.time()-start,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        print(cmd[0],case.name,'exit',proc.returncode,flush=True)
        if proc.returncode:
            (case/'execution.json').write_text(json.dumps({'version_output':version,'commands':records,'status':'FAILED'},indent=2))
            raise RuntimeError(log.read_text()[-3500:])
        if cmd[0]=='checkMesh' and 'Mesh OK.' not in log.read_text():
            (case/'execution.json').write_text(json.dumps({'version_output':version,'commands':records,'status':'MESH_REJECTED'},indent=2))
            raise RuntimeError('Default mesh geometry/topology checks failed; solver not executed')
        if cmd[0]=='checkMesh':
            from stageb.cfd_quality import geometry_gate
            gate=geometry_gate(case);(case/'geometry-gate.json').write_text(json.dumps(gate,indent=2))
            if not gate['passed']:
                (case/'execution.json').write_text(json.dumps({'version_output':version,'commands':records,'status':'GEOMETRY_REJECTED','geometry':gate},indent=2))
                raise RuntimeError('Physical cassette was lost/distorted in meshing; do not run/use this geometry')
    quality=(case/'log.checkMesh').read_text()
    if 'Mesh OK.' not in quality:raise RuntimeError('checkMesh did not pass; no valid CFD claim')
    status={'version_output':version,'commands':records,'status':'MESH_ONLY' if mesh_only else 'EXECUTED_NOT_YET_GRID_OR_EXPERIMENTALLY_VALIDATED'}
    (case/'execution.json').write_text(json.dumps(status,indent=2))
    if not mesh_only:
        from stageb.cfd_quality import analyze
        status['quality']=analyze(case)
        (case/'execution.json').write_text(json.dumps(status,indent=2))
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['generate','run'])
    p.add_argument('--case',type=Path,default=RESULTS/'cfd/coarse');p.add_argument('--level',choices=['coarse','medium','fine'],default='coarse')
    p.add_argument('--wind',type=float,default=5);p.add_argument('--z0',type=float,default=.03);p.add_argument('--angle',type=float,default=50)
    p.add_argument('--direction',type=float,default=0);p.add_argument('--iterations',type=int,default=800);p.add_argument('--empty',action='store_true')
    p.add_argument('--foam-prefix',type=Path);p.add_argument('--mesh-only',action='store_true');a=p.parse_args()
    if a.operation=='generate':generate(a.case,a.level,a.wind,a.z0,a.angle,a.direction,a.iterations,a.empty);print(a.case)
    else:run(a.case,a.foam_prefix,a.mesh_only)


if __name__=='__main__':main()
