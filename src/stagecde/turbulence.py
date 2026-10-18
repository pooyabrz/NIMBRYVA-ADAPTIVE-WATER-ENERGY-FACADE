"""Matched-mesh turbulence sensitivity; not a transient time-step study."""
import argparse,hashlib,json,math,re,shutil,subprocess,time
from pathlib import Path
from assembly import ROOT
from stageb.foam import environment,header,PIN,PATCH
from stageb.cfd_quality import analyze,geometry_gate

def run(source,prefix):
    source=Path(source);dest=ROOT/'results/research/C/cfd'/('sst-'+source.name)
    if dest.exists():raise FileExistsError('Preserve previous CFD evidence: '+str(dest))
    dest.mkdir(parents=True)
    for name in ['0','system','constant']:shutil.copytree(source/name,dest/name)
    meta=json.loads((source/'case.json').read_text());meta.update(RAS_model='kOmegaSST',case_generated_only=True,not_a_converged_solution=True,
      reused_mesh_source=str(source.relative_to(ROOT)),turbulence_sensitivity_only=True)
    meta['limitations']+=['SST neutral ABL consistency must be checked by the separately run empty-domain counterpart']
    (dest/'case.json').write_text(json.dumps(meta,indent=2))
    (dest/'constant/turbulenceProperties').write_text(header('turbulenceProperties')+'simulationType RAS; RAS {RASModel kOmegaSST; turbulence on; printCoeffs on;}\n')
    z0=meta['z0_m'];u=meta['wind10_m_s'];us=u*.41/math.log((10+z0)/z0);omega=us/(.41*.3*(10+z0))
    eps=(dest/'0/epsilon').read_text();body=eps.replace('object epsilon','object omega').replace('[0 2 -3 0 0 0 0]','[0 0 -1 0 0 0 0]')
    body=body.replace('atmBoundaryLayerInletEpsilon','atmBoundaryLayerInletOmega').replace('epsilonWallFunction','omegaWallFunction')
    body=re.sub(r'(internalField uniform )[^;]+',lambda m:m[1]+str(omega),body)
    body=re.sub(r'((?:inletValue|value) uniform )([\deE+.-]+)',lambda m:m[1]+str(omega),body)
    body=re.sub(r'(ground\s*\{\s*type )omegaWallFunction;',lambda m:m[1]+f'atmOmegaWallFunction; z0 uniform {z0};',body)
    (dest/'0/omega').write_text(body);(dest/'0/epsilon').unlink()
    for name in ['fvSchemes','fvSolution']:
        p=dest/'system'/name;p.write_text(p.read_text().replace('epsilon','omega'))
    p=dest/'system/fvSchemes';p.write_text(p.read_text()+'\nwallDist {method meshWave;}\n')
    p=dest/'system/controlDict';text=re.sub(r'endTime\s+\d+','endTime 1200',p.read_text());text=re.sub(r'writeInterval\s+\d+','writeInterval 200',text);p.write_text(text)
    env=environment(prefix);version=subprocess.run(['simpleFoam','-help'],env=env,capture_output=True,text=True,check=True).stdout
    if 'OpenFOAM-2312' not in version or 'patch='+PATCH not in version:raise RuntimeError('Pinned solver required')
    records=[]
    for cmd in [['checkMesh'],['simpleFoam'],['postProcess','-func','writeCellCentres','-latestTime']]:
        start=time.time();log=dest/('log.'+cmd[0])
        with log.open('w') as f:r=subprocess.run(cmd+['-case',str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT)
        records.append({'command':cmd,'returncode':r.returncode,'seconds':time.time()-start,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        if r.returncode:
            (dest/'execution.json').write_text(json.dumps({'status':'FAILED','version_output':version,'commands':records},indent=2))
            raise RuntimeError(log.read_text()[-4000:])
        if cmd[0]=='checkMesh':
            gate=geometry_gate(dest);(dest/'geometry-gate.json').write_text(json.dumps(gate,indent=2))
            if 'Mesh OK.' not in log.read_text() or not gate['passed']:raise RuntimeError('Reused mesh rejected')
    q=analyze(dest);(dest/'execution.json').write_text(json.dumps({'status':'EXECUTED_NOT_EXPERIMENTALLY_VALIDATED','version_output':version,'commands':records,'quality':q},indent=2))
    print(dest.name,q['flow_residual_gate_passed'],flush=True)
    return {'source':source.name,'SST_quality':q,'kEpsilon_quality':json.loads((source/'quality.json').read_text())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--relaxed',action='store_true');a=p.parse_args();out=[]
    if a.relaxed:
        relaxed(a.prefix);return
    for name in ['empty-medium','reference-medium']:out.append(run(ROOT/'results/stage-b/cfd'/name,a.prefix))
    (ROOT/'results/research/C/turbulence.json').write_text(json.dumps({'comparisons':out,'model_form_sensitivity':True,'independent_validation':False},indent=2))

def relaxed(prefix):
    """A failed residual gate triggers a preserved, lower-relaxation retry."""
    source=ROOT/'results/research/C/cfd/sst-reference-medium';dest=source.with_name('sst-reference-medium-relaxed')
    if dest.exists():raise FileExistsError('Preserve existing retry evidence')
    dest.mkdir(parents=True)
    for name in ['constant','system']:shutil.copytree(source/name,dest/name)
    latest=max((p for p in source.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)>0),key=lambda p:int(p.name))
    (dest/'0').mkdir()
    for name in ['U','p','k','omega','nut','T','Yv']:shutil.copy2(latest/name,dest/'0'/name)
    meta=json.loads((source/'case.json').read_text());meta.update(initialization_source=str(latest.relative_to(ROOT)),iteration_relaxation_retry=True,case_generated_only=True)
    (dest/'case.json').write_text(json.dumps(meta,indent=2))
    p=dest/'system/fvSolution';text=p.read_text().replace('p 0.3','p 0.15').replace('U 0.6','U 0.2').replace('k 0.5','k 0.2').replace('omega 0.5','omega 0.2');p.write_text(text)
    p=dest/'system/controlDict';p.write_text(re.sub(r'endTime\s+\d+','endTime 800',p.read_text()))
    env=environment(prefix);records=[]
    version=subprocess.run(['simpleFoam','-help'],env=env,capture_output=True,text=True,check=True).stdout
    for cmd in [['checkMesh'],['simpleFoam'],['postProcess','-func','writeCellCentres','-latestTime']]:
        start=time.time();log=dest/('log.'+cmd[0])
        with log.open('w') as f:r=subprocess.run(cmd+['-case',str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT)
        records.append({'command':cmd,'returncode':r.returncode,'seconds':time.time()-start,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        if r.returncode:raise RuntimeError(log.read_text()[-3000:])
    q=analyze(dest);(dest/'execution.json').write_text(json.dumps({'status':'EXECUTED_NOT_EXPERIMENTALLY_VALIDATED','version_output':version,'commands':records,'quality':q},indent=2))
    path=ROOT/'results/research/C/turbulence.json';result=json.loads(path.read_text());result['relaxed_retry']=q
    result['matched_mesh_model_difference_usable']=bool(q['flow_residual_gate_passed'])
    path.write_text(json.dumps(result,indent=2));print('SST relaxed residual gate',q['flow_residual_gate_passed'],flush=True)
if __name__=='__main__':main()
