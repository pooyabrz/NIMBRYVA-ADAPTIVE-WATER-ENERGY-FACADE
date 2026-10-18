"""Package source, inputs and lossless solved-field exports; omit rebuildable meshes."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def include(path):
    rel=path.relative_to(ROOT);parts=rel.parts
    if '__pycache__' in parts or '.git' in parts or path.suffix=='.pyc' or path.name=='MANIFEST.json':return False
    if parts[:2]==('data','raw') and path.suffix=='.json' and not path.name.endswith('.provenance.json'):
        if any(f'-{year}.json' in path.name for year in [2020,2021,2022,2023]):return False
    if 'cfd' in parts and parts[:2]==('results','research'):
        i=parts.index('cfd');rest=parts[i+2:]
        if rest and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',rest[0]) and float(rest[0])>0:return False
        if rest[:2]==('constant','polyMesh') or 'sets' in rest or path.suffix in ['.vtk','.vtp','.obj']:return False
    if len(parts)>4 and parts[:3]==('results','stage-b','cfd'):
        rest=parts[4:]
        if rest and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',rest[0]) and float(rest[0])>0:return False
        if rest[:2]==('constant','polyMesh'):return False
        if 'sets' in rest or path.suffix in ['.vtk','.vtp','.obj']:return False
        if path.name=='reference-field.npz' and parts[3] not in ['reference-coarse','reference-medium','reference-fine','empty-medium']:return False
    if parts[:3]==('results','stage-b','rain'):
        if path.name.endswith('-cfd.npz'):return False
        if path.name.endswith('-paths.npz') and '-union' not in path.name:return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    files=sorted(x for x in ROOT.rglob('*') if x.is_file() and include(x))
    records=[{'path':x.relative_to(ROOT).as_posix(),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]
    manifest={'model_version':'cassette-v2.0 + stage-b-reference-1 + CDE-1 + presentation-F1','scientific_gate':'BLOCKED_FINAL_DEVICE_AND_HIGH_FIDELITY_CLAIMS',
        'omitted_rebuildable_data':['OpenFOAM constant/polyMesh','OpenFOAM positive-time ASCII field directories','duplicate CFD NPZ caches','2020-2023 raw weather JSON; normalized annual data and source provenance retained; research weather downloads raw sources again'],
        'retained_CFD_evidence':'case inputs + execution/quality/logs + lossless internal-field NPZ with checksum and geometry signature',
        'files':records}
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for file in files+[ROOT/'MANIFEST.json']:z.write(file,Path('water-facade')/file.relative_to(ROOT))
    with zipfile.ZipFile(a.output) as z:
        if z.testzip():raise ValueError('ZIP CRC failed')
        for entry in records:
            data=z.read('water-facade/'+entry['path'])
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Packaged file checksum mismatch')
    print(json.dumps({'archive':str(a.output.resolve()),'files':len(records)+1,'bytes':a.output.stat().st_size,
        'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'zip_and_manifest':'verified'},indent=2))


if __name__=='__main__':main()
