from RSTK.Workflow import LFM
from RSTK.Workflow import SVD
from RSTK.Workflow import FM
from RSTK.Workflow import BPR
from RSTK.Workflow import KNN
import sys


def choose():
    arg = sys.argv[1]
    if arg == 'lfm':
        LFM()
    elif arg == 'fm':
        FM()
    elif arg == 'bpr':
        BPR()
    elif arg == 'knn':
        KNN()
    elif arg == 'SVD':
        SVD()
    sys.exit()

if __name__ == '__main__':
    # Transfer().process()
    choose()
