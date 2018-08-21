# -*- coding: utf-8 -*-
import time
import os
from DataProcess.SplitData import Split
from Model.LFM import LFM
from Evaluation.LFM_Evaluator import ModelRmse

train=[]
test=[]
P=dict()
Q=dict()
bu=dict()
bi=dict()
mu=0

def run():
    assert os.path.exists('Data/u.data'), \
        'File not exists in path, please add data file first.'
    print('Start LFM..')
    start = time.time()

    train, test = Split().split(10, 1, 1)
    LFM(train, 20, 10, 0.02, 0.02, P, Q, bu, bi, mu)
    print('RMSE : ', ModelRmse(train, test, P, Q, bu, bi, mu))

    print('LFM Cost time: %f' % (time.time() - start))