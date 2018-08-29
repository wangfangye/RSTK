import sys
from DataProcess.TypeTransfer import Transfer
from Workflow.LFM_workflow import run as lfm
from Workflow.FM_workflow import run as fm
from Workflow.KNN_workflow import run as knn


def choose():
    arg = sys.argv[1]
    if arg == 'lfm':
        lfm()
    elif arg == 'fm':
        fm()
    elif arg == 'knn':
        knn()    
    sys.exit()

if __name__ == '__main__':
    # Transfer().process()
    choose()