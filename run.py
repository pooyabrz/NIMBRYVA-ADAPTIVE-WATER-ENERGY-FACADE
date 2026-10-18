"""Convenient LOCAL launcher; all outputs stay in this project directory."""
from pathlib import Path
import subprocess
import sys

TASKS={'check':'verify.py','view':'view_local.py','benchmark':'benchmark.py',
       'train':'train_rl.py','fetch':'fetch_weather.py','prepare':'prepare_weather.py',
       'sensitivity':'sensitivity.py','evaluate':'evaluate_rl.py',
       'audit':'evidence_audit.py','specs':'evidence_audit.py',
       'uncertainty':'uncertainty.py','configure':'configure_v2.py',
       'export':'export_geometry.py','materials':'prepare_materials.py',
       'release-report':'release_report.py','stage-b':'stage_b.py','research':'stage_cde.py','presentation':'presentation_cli.py','concept-demo':'concept_demo.py'}

if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in TASKS:
        print('Usage: python run.py {'+'|'.join(TASKS)+'} [options]')
        raise SystemExit(0 if len(sys.argv)<2 else 2)
    path=Path(__file__).resolve().parent/'src'/TASKS[sys.argv[1]]
    result=subprocess.run([sys.executable,str(path),*sys.argv[2:]])
    raise SystemExit(result.returncode)
