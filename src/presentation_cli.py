import sys
def main():
    if len(sys.argv)<2:raise SystemExit('Usage: python run.py presentation {fetch|episodes|run|sensitivity|check|report|view}')
    task=sys.argv[1]
    if task in ['fetch','episodes','run','sensitivity']:
        from presentation.evaluate import main as command
    else:
        sys.argv.pop(1)
        if task=='check':from presentation.verify import main as command
        elif task=='report':from presentation.report import main as command
        elif task=='view':
            from stagecde.view_local import main as command
            sys.argv.extend(['--source','presentation'])
        else:raise SystemExit('Unknown presentation task')
    command()
if __name__=='__main__':main()
