"""Export implemented specification and reject unsupported final claims."""
import argparse,json,csv,hashlib
import numpy as np
from weather import model_config,ROOT
from assembly import reference,compile_assembly,MODEL_VERSION
from spectral import SpectralRadiation
from actuator import load_envelope,transition,allowable_wind


GAPS=[
 {'id':'G01','topic':'PV-specific internal stack and optics','status':'OPEN_DATA',
  'implemented':'5 PV finite volumes + XPS, electrical rating anchored to Trina TSM_DE_2023_A',
  'missing':'actual encapsulant grade/thickness, wafer/passivation/metallization stack, n/k/EQE/angular optics and contact resistance of this serial product',
  'close_with':'manufacturer BOM or a fully specified research laminate with public optical constants; no physical lab required to ingest those data'},
 {'id':'G02','topic':'full spectral/angle/hourly radiation','status':'PARTIAL_DATA_AND_SOLVER',
  'implemented':'2321 raw ETH optical samples, QC, trusted .25..20 um, spectral thermal integral 2.5..1000 um, G173 reference and external NPZ interface; broadband identifiability bounds',
  'missing':'reliable emitter data beyond 20 um, angular optical measurements, water-covered optics, real hourly atmospheric profiles/clouds/aerosols and radiance spectra',
  'close_with':'author/supplier optical data plus a documented libRadtran/ARTS calculation with actual atmospheric profiles; broadband LW alone is insufficient'},
 {'id':'G03','topic':'coating durability and condensate delivery','status':'OPEN_TRANSFER_VALIDATION',
  'implemented':'actual ETH 1.5 h/12 h flux series, film mass balance, jumping/sliding/collector scenarios, wet hours and UV exposure counters without lifetime inference',
  'missing':'outdoor ageing law, adhesion on selected Ag/Cr stack, droplet population/jumping/capture under the proposed moving 20 mm cavity; orientation-dependent collection plumbing',
  'close_with':'existing independent durability/transport datasets for this stack or a validated droplet-scale numerical model; accelerated steam hours cannot be mapped to facade years'},
 {'id':'G04','topic':'cassette flow and thermal bridges','status':'IMPLEMENTED_REDUCED_ORDER_OPEN_VALIDATION',
  'implemented':'8 solid nodes, 2-port pressure/stack network, moist-air energy+humidity balance, 2 explicit parallel bridge paths and contact resistance',
  'missing':'resolved external/entrance flow, local port Cp, developing unequal-wall transfer, 3D thermal spreading, drain/cover/frame geometry and condensation on other internal walls',
  'close_with':'converged CFD/CHT + 3D thermal FE for a fixed CAD and comparison with published benchmark cases'},
 {'id':'G05','topic':'final motor/wind qualification','status':'PRELIMINARY_SIZE_ONLY',
  'implemented':'one HangTuo SE3+24V reference, rated/hold/tilt/radial checks, finite motion/inertia, upper running-energy scenario and action protection',
  'missing':'order-specific load-speed-current/start/stall/duty curves, verified rotation limits/IP/corrosion, local design gust/site code, anchors/bearings/cable/drain routing and aeroelastic fatigue',
  'close_with':'supplier confirmed order sheet and site-specific CFD/structural sizing; a catalogue static pass is not final compatibility'}]


def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(str(type(x)))


def audit():
    c=model_config();a=compile_assembly(c);r=SpectralRadiation(c)
    # Representative catalogue reference; this is explicitly not the old
    # 40 Nm Belimo gearbox combined with somebody else's electrical rating.
    summary={'model_version':MODEL_VERSION,'scientific_gate':'BLOCKED_FINAL_DEVICE_AND_HIGH_FIDELITY_CLAIMS',
      'config_sha256':hashlib.sha256((ROOT/'config/model.json').read_bytes()).hexdigest(),
      'closed_implementation':['original project integration','finite physical rotation','8-node laminate/cavity thermal and moisture balances','source-data/QC separation','policy version guard'],
      'critical_gaps':GAPS,'assembly':{k:v for k,v in a.items() if k not in ['capacity_J_K','laplacian_W_K']},
      'spectral_audit':r.audit(),'flip_180':transition(c,50,230),
      'preliminary_wind_limit_local_mean_m_s':allowable_wind(c,a),
      'wind_cases':[{'mean_wind_m_s':u,**load_envelope(c,a,u)} for u in [5,10,15,20,30,40]],
      'actuator_reference':{'manufacturer':'Changzhou Hangtuo Mechanical Co., Ltd.','model':'SE3 with 24VDC motor; webpage reference, not confirmed order code',
        'url':'https://www.solarslewdrive.com/sale-12281963-se3-solar-slew-drive-enclosed-housing-with-24vdc-planetary-motor-for-solar-tracker-and-csp.html',
        'rated_Nm':600,'hold_Nm':3000,'tilt_Nm':1500,'output_rpm':.065,'motor_nominal_W':30,'running_current_A_less_than':3,'IP_rating':None,'qualified_final':False},
      'coating_evidence':[{'source':'Haechler 2021 Fig4C','duration_h':12,'replicates':2,'environment':'humid air, Peltier-driven lab collection, not facade service'},
        {'source':'Donati et al. DOI 10.1002/admi.202001176','environment':'111 C steam, 1.42 bar, approximately 3 m/s; Cu with Al primer',
         'jumping_h_approx':10,'additional_dropwise_h_approx':50,'filmwise_failure_h_approx':72,'transfer_to_facade_years':False}],
      'prohibited_claims':['all layer values specific to purchased PV module','complete measured full-spectrum forcing in 12 cities','outdoor coating lifetime established','CFD/3D FE validated cassette','final wind-compatible motor','validated annual water yield','old RL policy remains compatible']}
    out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'config/evidence/runtime-parameter-metadata.json').read_text())
    additional=json.loads((ROOT/'config/evidence/additional-sources.json').read_text())
    available_sources=set(reference()['sources'])|set(additional)
    rows=[]
    for key,value in c.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            if key not in meta:raise ValueError('Missing parameter metadata '+key)
            if any(s not in available_sources for s in meta[key]['source_id'].split(';')):
                raise ValueError('Unknown source ID for active parameter '+key)
            rows.append({'parameter':key,'active_value':value,**meta[key]})
    with (ROOT/'config/evidence/active-parameters.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'evidence-audit.json').write_text(json.dumps(summary,indent=2,default=native)+'\n')
    with (out/'layers.csv').open('w',newline='') as f:
        keys=sorted({k for row in a['layers'] for k in row});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(a['layers'])
    with (out/'wind-sizing.csv').open('w',newline='') as f:
        rows=summary['wind_cases'];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');a=p.parse_args()
    result=audit()
    print(json.dumps({k:result[k] for k in ['model_version','scientific_gate','preliminary_wind_limit_local_mean_m_s']},indent=2))
    print('Five evidence gaps remain explicit. --strict exits 2 intentionally; this is a scientific evidence gate.')
    if a.strict:raise SystemExit(2)


if __name__=='__main__':main()
