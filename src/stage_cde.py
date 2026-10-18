"""Entry points for numerical verification, experiments and research control."""
import sys
def main():
    if len(sys.argv)<2:raise SystemExit('Usage: python run.py research {verify|sensitivity|experiments|weather|transitions|surrogate|control|tracking|turbulence|assess|warmup|view|check|report}')
    task=sys.argv.pop(1)
    if task=='verify':from stagecde.numerics import main as command
    elif task=='sensitivity':from stagecde.sensitivity import main as command
    elif task=='experiments':from stagecde.experiments import main as command
    elif task=='weather':from stagecde.datasets import main as command
    elif task=='transitions':from stagecde.transitions import main as command
    elif task=='surrogate':from stagecde.surrogate import main as command
    elif task=='control':from stagecde.control import main as command
    elif task=='check':from stagecde.verify import main as command
    elif task=='report':from stagecde.report import main as command
    elif task=='turbulence':from stagecde.turbulence import main as command
    elif task=='view':from stagecde.view_local import main as command
    elif task=='tracking':from stagecde.tracking import main as command
    elif task=='assess':from stagecde.assessment import main as command
    elif task=='warmup':from stagecde.warmup_audit import main as command
    else:raise SystemExit('Unknown research command')
    command()
if __name__=='__main__':main()
