# coding=utf-8

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import shutil

from sklearn.model_selection import KFold, ShuffleSplit
from Utils.extra_functions import check_error_file

class CrossValidation(object):
    def __init__(self, input_file, recommender, dir_folds, k_folds=10, header=None, sep='\t', write_predictions=False,
                 write_sep='\t',  write_mode='w', recommender_verbose=False, evaluation_in_fold_verbose=True, metrics=None, n_splits=10,
                 as_table=False, table_sep='\t', del_folds=False, random_seed=None, names=None, as_binary=False, binary_col=2):
        """
        Cross Validation

        This strategy is responsible to divide the database in K folds, in which each fold contain a train and a test
        set. Its also responsible to run and evaluate the recommender results in each fold and calculate the mean and
        the standard deviation.

        Usage:
            >> rec = MostPopular(as_binary=True)
            >> CrossValidation(db, rec, fold_d, evaluation_in_fold_verbose=False).compute()

        :param input_file: Database file
        :type input_file: str

        :param recommender: Initialize the recommender algorithm. e.g.: MostPopular(as_binary=True)
        :type recommender: class

        :param dir_folds: Directory to write folds (train and test files)
        :type dir_folds: str

        :param k_folds: How much folds the strategy will divide
        :type k_folds: int, default 10

        :param header: Skip header line
        :type header: int, default None

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param write_predictions: Write the recommender predictions in each fold
        :type write_predictions: bool, default False

        :param write_sep: Delimiter for output files
        :type write_sep: str, default '\t'

        :param recommender_verbose: Print header of recommender in each fold
        :type recommender_verbose: bool, default False

        :param evaluation_in_fold_verbose: Print evaluation of recommender in each fold
        :type evaluation_in_fold_verbose: bool, default True

        :param metrics: List of evaluation metrics
        :type metrics: str, default None

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        :param del_folds: Delete folds after evaluation
        :type del_folds: bool, default False

        :param random_seed: Random seed
        :type random_seed: int, default None

        """

        self.input_file = input_file
        self.recommender = recommender
        self.dir_folds = dir_folds
        self.k_folds = k_folds
        self.header = header
        self.sep = sep
        self.write_predictions = write_predictions
        self.write_sep = write_sep
        self.recommender_verbose = recommender_verbose
        self.evaluation_in_fold_verbose = evaluation_in_fold_verbose
        self.metrics = metrics
        self.as_table = as_table
        self.table_sep = table_sep
        self.del_folds = del_folds
        self.random_seed = random_seed

        # internal vars
        self.folds_results = defaultdict(list)

        # split data
        self.n_splits = n_splits
        self.sep_write = write_sep
        self.write_mode = write_mode

        # process data
        self.names = names
        self.as_binary = as_binary
        self.binary_col = binary_col



    def generate_folds(self, shuffle=True, random_state=None):
        """
        Method to generate folds with k fold cross validation

        """
        check_error_file(self.input_file)

        self.df = self.read_with_pandas()

        if self.dir_folds is not None:
            self.create_folds()

        kfold = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        trained_model = list(kfold.split(self.df))

        if self.dir_folds is not None:
            self.write_files(trained_model)

        return trained_model

    def execute_algorithm(self):
        """
        Method to run recommender algorithm in k folds

        """

        for k in range(self.k_folds):
            train_file = self.dir_folds + 'folds/%d/train.dat' % k
            test_file = self.dir_folds + 'folds/%d/test.dat' % k

            self.recommender.train_file = train_file
            self.recommender.test_file = test_file

            if self.write_predictions:
                output_file = self.dir_folds + 'folds/%d/output.dat' % k
                self.recommender.output_file = output_file

            self.recommender.compute(verbose=self.recommender_verbose,
                                     verbose_evaluation=self.evaluation_in_fold_verbose, metrics=self.metrics)

            if self.metrics is None:
                self.metrics = self.recommender.evaluation_results.keys()

            for metric in self.metrics:
                self.folds_results[metric.upper()].append(self.recommender.evaluation_results[metric.upper()])

    def evaluate(self, verbose=True):
        """
        Method to evaluate folds results and generate mean and standard deviation

        :param verbose: If True, print evaluation results
        :type verbose: bool, default True

        """

        mean_dict = defaultdict(dict)
        std_dict = defaultdict(dict)

        for metric in self.metrics:
            mean_dict[metric.upper()] = np.mean(self.folds_results[metric.upper()])
            std_dict[metric.upper()] = np.std(self.folds_results[metric.upper()])

        if verbose:
            if self.as_table:
                header = ''
                values_mean = ''
                values_std = ''
                for metric in self.metrics:
                    header += metric.upper() + self.table_sep
                    values_mean += str(round(mean_dict[metric.upper()], 6)) + self.table_sep
                    values_std += str(round(std_dict[metric.upper()], 6)) + self.table_sep
                print('Metric%s%s' % (self.table_sep, header))
                print('Mean%s%s' % (self.table_sep, values_mean))
                print('STD%s%s' % (self.table_sep, values_std))
            else:
                evaluation_mean = 'Mean:: '
                evaluation_std = 'STD:: '
                for metrics in self.metrics:
                    evaluation_mean += "%s: %.6f " % (metrics.upper(), mean_dict[metrics.upper()])
                    evaluation_std += "%s: %.6f " % (metrics.upper(), std_dict[metrics.upper()])
                print(evaluation_mean)
                print(evaluation_std)

    def erase_folds(self):
        """
        Method to delete folds after evaluation

        """

        folds = self.dir_folds + 'folds/'
        shutil.rmtree(folds)

    def compute(self, verbose=True):
        """
        Method to run the cross validation

        :param verbose: If True, print header
        :type verbose: bool, default True

        """

        if verbose:

            print("[Case Recommender: Cross Validation]\n")
            print("Database:: %s \nRecommender Algorithm:: %s | K Folds: %d\n" % (self.input_file,
                                                                                  self.recommender.recommender_name,
                                                                                  self.k_folds))

        self.generate_folds(random_state=self.random_seed)
        self.execute_algorithm()
        self.evaluate(verbose)

        if self.del_folds:
            self.erase_folds()

    def create_folds(self):

        if not os.path.exists(self.dir_folds):
            self.dir_folds += "folds/"
            os.mkdir(self.dir_folds)

        for n in range(self.n_splits):
            if not os.path.exists(self.dir_folds + str(n)):
                os.mkdir(self.dir_folds + str(n))

    def read_with_pandas(self):
        """
        Method to read file with pandas

        :return DataFrame with file lines

        """

        df = pd.read_csv(self.input_file, sep=self.sep, skiprows=self.header, header=None, names=self.names)

        if self.header is not None:
            df.columns = [i for i in range(len(df.columns))]

        if self.as_binary:
            df.iloc[:, self.binary_col] = 1
        return df.sort_values(by=[0, 1])

    def write_files(self, trained_model):
        fold = 0
        for train_index, test_index in trained_model:
            if self.dir_folds is not None:
                train_file = self.dir_folds + str(fold) + '/train.dat'
                test_file = self.dir_folds + str(fold) + '/test.dat'

                df_train = self.df.ix[train_index]
                df_test = self.df.ix[test_index]

                df_train.sort_values(by=[0, 1]).to_csv(train_file, sep=self.sep_write, mode=self.write_mode, header=None, index=False)
                df_test.sort_values(by=[0, 1]).to_csv(test_file, sep=self.sep_write, mode=self.write_mode, header=None, index=False)
                fold += 1