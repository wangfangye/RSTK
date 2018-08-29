# -*- coding: utf-8 -*-
import os
from Model.KNN import UserKNN
from Model.KNN import ItemKNN
from Utils.Crossvalidate import CrossValidation

db = 'Data/u.data'
folds_path = 'Data/'
tr = 'Data/folds/0/train.dat'
te = 'Data/folds/0/test.dat'

def run():
    assert os.path.exists('Data/u.data'), \
        'File not exists in path, please add data file first.'
    print('Start KNN..')

    model = UserKNN()
    CrossValidation(source=db, model=model, targets=folds_path, n_folds=5).compute()
    UserKNN(tr, te).compute()

    # model = ItemKNN()
    # CrossValidation(source=db, model=model, targets=folds_path, n_folds=5).compute()
    # ItemKNN(tr, te).compute()