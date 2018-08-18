# -*- coding: utf-8 -*-
from Model.userknn import UserKNN
from Utils.Crossvalidate import CrossValidation

db = 'Data/u.data'
folds_path = 'Data/'
tr = 'Data/folds/0/train.dat'
te = 'Data/folds/0/test.dat'

def run():
    # Cross Validation
    recommender = UserKNN()

    CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

    # # Simple
    UserKNN(tr, te, verbose=True).compute()

    # # Cross Validation
    # recommender = ItemKNN()
    #
    # CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

    # # Simple
    # ItemKNN(tr, te).compute()