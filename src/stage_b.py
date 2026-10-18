"""Single local entry point for independent stage-B references."""
import sys


def main():
    if len(sys.argv)<2:
        print('Usage: python run.py stage-b {cfd|thermal|surface|rain|doe|check|report|couple|quality|view|study} [options]');return
    task=sys.argv.pop(1)
    if task=='cfd':from stageb.foam import main as command
    elif task=='thermal':from stageb.thermal import main as command
    elif task=='surface':from stageb.surface import main as command
    elif task=='rain':from stageb.rain import main as command
    elif task=='doe':from stageb.design import main as command
    elif task=='check':from stageb.verify import main as command
    elif task=='report':from stageb.report import main as command
    elif task=='couple':from stageb.couple import main as command
    elif task=='quality':from stageb.cfd_quality import main as command
    elif task=='view':from stageb.view_local import main as command
    elif task=='study':from stageb.study import main as command
    else:raise SystemExit('Unknown stage-B command')
    command()


if __name__=='__main__':main()
