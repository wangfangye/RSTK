import sys
from .DataProcess.TypeTransfer import Transfer
from .Workflow.LFM_workflow import run as lfm

def choose():
    arg = sys.argv[1]
    if arg == 'lfm':
        lfm()
    sys.exit()

if __name__ == '__main__':
    # Transfer().process()
    choose()