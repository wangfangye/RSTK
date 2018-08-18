# coding=utf-8
"""
    UserKNN based on Collaborative Filtering Recommender
    [Rating Prediction]

    Literature:
        KAggarwal, Charu C.:
        Chapter 2: Neighborhood-Based Collaborative Filtering
        Recommender Systems: The Textbook. 2016
        file:///home/fortesarthur/Documentos/9783319296579-c1.pdf

"""

# © 2018. Case Recommender (MIT License)
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import random

from Utils.extra_functions import timed
from DataProcess.processdata import ReadFile, WriteFile
from Evaluation.itemfunctions import precision_at_k, mean_average_precision, ndcg_at_k
__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class UserKNN():
    def __init__(self, train_file=None, test_file=None, output_file=None, similarity_metric="cosine", k_neighbors=None,
                 reg_bi=10, reg_bu=15, all_but_one_eval=False, n_ranks=list([1, 3, 5, 10]), verbose=True, as_rank=False,
                 as_similar_first=False, sep='\t', output_sep='\t', metrics=list(['MAE', 'RMSE']), as_table=False, table_sep='\t'):
        """
        UserKNN for rating prediction

        This algorithm predicts ratings for each user based on the similar items that his neighbors
        (similar users) consumed.

        Usage::

            >> UserKNN(train, test).compute()
            >> UserKNN(train, test, ranking_file, as_similar_first=True, k_neighbors=60).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param similarity_metric: Pairwise metric to compute the similarity between the users. Reference about
        distances: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
        :type similarity_metric: str, default cosine

        :param k_neighbors: Number of neighbors to use. If None, k_neighbor = int(sqrt(n_users))
        :type k_neighbors: int, default None

        :param as_similar_first: If True, for each unknown item, which will be predicted, we first look for its k
        most similar users and then take the intersection with the users that
        seen that item.
        :type as_similar_first: bool, default False

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param all_but_one_eval: If True, considers only one pair (u, i) from the test set to evaluate the ranking
        :type all_but_one_eval: bool, default False

        :param n_ranks: List of positions to evaluate the ranking
        :type n_ranks: list, default [1, 3, 5, 10]

        :param verbose: Print the evaluation results
        :type verbose: bool, default True
        """

        self.recommender_name = 'UserKNN Algorithm'

        self.as_similar_first = as_similar_first
        self.k_neighbors = k_neighbors

        # internal vars
        self.su_matrix = None
        self.users_id_viewed_item = None

        # base knn
        self.reg_bi = reg_bi
        self.reg_bu = reg_bu

        # internal vars
        self.number_users = None
        self.number_items = None
        self.bu = {}
        self.bi = {}
        self.bui = {}

        # base prediction
        self.train_file = train_file
        self.test_file = test_file
        self.similarity_metric = similarity_metric
        self.output_file = output_file
        self.sep = sep
        self.output_sep = output_sep

        # internal vars
        self.item_to_item_id = {}
        self.item_id_to_item = {}
        self.user_to_user_id = {}
        self.user_id_to_user = {}
        self.train_set = None
        self.test_set = None
        self.users = None
        self.items = None
        self.matrix = None
        self.evaluation_results = None
        self.extra_info_header = None
        self.predictions = []

        self.as_rank = as_rank
        self.all_but_one_eval = all_but_one_eval
        self.n_ranks = n_ranks
        self.verbose = verbose
        self.metrics = metrics
        self.as_table = as_table
        self.table_sep = table_sep

    def init_model(self):
        """
        Method to initialize the model. Compute similarity matrix based on user (user x user)

        """

        self.number_users = len(self.users)
        self.number_items = len(self.items)

        self.create_matrix()

        self.users_id_viewed_item = {}

        # Set the value for k
        if self.k_neighbors is None:
            self.k_neighbors = int(np.sqrt(len(self.users)))

        self.su_matrix = self.compute_similarity(transpose=False)

        # Map the users which seen an item with their respective ids
        for item in self.items:
            for user in self.train_set['users_viewed_item'].get(item, []):
                self.users_id_viewed_item.setdefault(item, []).append(self.user_to_user_id[user])

    def train_baselines(self):
        """
        Method to train baselines for each pair user, item

        """

        self.bu = {}
        self.bi = {}
        self.bui = {}

        for i in range(10):
            self.compute_bi()
            self.compute_bu()
        self.compute_bui()

    def predict(self):
        """
        Method to predict ratings for all known users in the train set.

        """

        for user in self.users:
            if len(self.train_set['feedback'].get(user, [])) != 0:
                if self.test_file is not None:
                    if self.as_similar_first:
                        self.predictions += self.predict_similar_first_scores(user, self.test_set['items_seen_by_user']
                                                                              .get(user, []))
                    else:
                        self.predictions += self.predict_scores(user, self.test_set['items_seen_by_user'].get(user, []))
                else:
                    # Selects items that user has not interacted with.
                    items_seen_by_user = []
                    u_list = list(np.flatnonzero(self.matrix[self.user_to_user_id[user]] == 0))
                    for item_id in u_list:
                        items_seen_by_user.append(self.item_id_to_item[item_id])

                    if self.as_similar_first:
                        self.predictions += self.predict_similar_first_scores(user, items_seen_by_user)
                    else:
                        self.predictions += self.predict_scores(user, items_seen_by_user)
            else:
                # Implement cold start user
                pass

    def predict_scores(self, user, unpredicted_items):
        """
        In this implementation, for each unknown item,
        which will be predicted, we first look for users that seen that item and calculate the similarity between them
        and the user. Then we sort these similarities and get the most similar k's. Finally, the score of the
        unknown item will be the sum of the similarities.

        rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

        :param user: User
        :type user: int

        :param unpredicted_items: A list of unknown items for each user
        :type unpredicted_items: list

        :return: Sorted list with triples user item rating
        :rtype: list

        """

        u_id = self.user_to_user_id[user]
        predictions = []

        for item in unpredicted_items:
            neighbors = []
            rui = 0
            sim_sum = 0
            for user_v_id in self.users_id_viewed_item.get(item, []):
                user_v = self.user_id_to_user[user_v_id]
                neighbors.append((user_v, self.su_matrix[u_id, user_v_id], self.train_set['feedback'][user_v][item]))
            neighbors = sorted(neighbors, key=lambda x: -x[1])

            if neighbors:
                for triple in neighbors[:self.k_neighbors]:
                    rui += (triple[2] - self.bui[triple[0]][item]) * triple[1] if triple[1] != 0 else 0.001
                    sim_sum += triple[1] if triple[1] != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def predict_similar_first_scores(self, user, unpredicted_items):
        """
        In this implementation, for each unknown item, which will be
        predicted, we first look for its k most similar users and then take the intersection with the users that
        seen that item. Finally, the score of the unknown item will be the sum of the  similarities.

        rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

        :param user: User
        :type user: int

        :param unpredicted_items: A list of unknown items for each user
        :type unpredicted_items: list

        :return: Sorted list with triples user item rating
        :rtype: list

        """
        u_id = self.user_to_user_id[user]
        predictions = []

        # Select user neighbors, sorting user similarity vector. Returns a list with index of sorting values
        neighbors = sorted(range(len(self.su_matrix[u_id])), key=lambda m: -self.su_matrix[u_id][m])

        for item in unpredicted_items:
            rui = 0
            sim_sum = 0

            # Intersection bt. the neighbors closest to the user and the users who accessed the unknown item.
            common_users = list(set(
                self.users_id_viewed_item.get(item, [])).intersection(neighbors[1:self.k_neighbors]))

            if common_users:
                for user_v_id in common_users:
                    user_v = self.user_id_to_user[user_v_id]
                    sim_uv = self.su_matrix[u_id, user_v_id]
                    rui += (self.train_set['feedback'][user_v][item] - self.bui[user_v][item]) * \
                        sim_uv if sim_sum != 0 else 0.001
                    sim_sum += sim_uv if sim_sum != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):
        """
        Extends compute method from BaseItemRecommendation. Method to run recommender algorithm

        :param verbose: Print recommender and database information
        :type verbose: bool, default True

        :param metrics: List of evaluation metrics
        :type metrics: list, default None

        :param verbose_evaluation: Print the evaluation results
        :type verbose_evaluation: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        # read files
        self.read_files()

        # initialize empty predictions (Don't remove: important to Cross Validation)
        self.predictions = []

        if verbose:
            test_info = None

            main_info = {
                'title': 'Rating Prediction > ' + self.recommender_name,
                'n_users': len(self.train_set['users']),
                'n_items': len(self.train_set['items']),
                'n_interactions': self.train_set['number_interactions'],
                'sparsity': self.train_set['sparsity']
            }

            if self.test_file is not None:
                test_info = {
                    'n_users': len(self.test_set['users']),
                    'n_items': len(self.test_set['items']),
                    'n_interactions': self.test_set['number_interactions'],
                    'sparsity': self.test_set['sparsity']
                }

            self.print_header(main_info, test_info)

        if verbose:
            self.init_model()
            print("training_time:: %4f sec" % timed(self.train_baselines))
            if self.extra_info_header is not None:
                print(self.extra_info_header)
            print("prediction_time:: %4f sec" % timed(self.predict))

        else:
            # Execute all in silence without prints
            self.extra_info_header = None
            self.init_model()
            self.train_baselines()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)

    def read_files(self):
        """
        Method to initialize recommender algorithm.

        """

        self.train_set = ReadFile(self.train_file, sep=self.sep).read()

        if self.test_file is not None:
            self.test_set = ReadFile(self.test_file, sep=self.sep).read()
            self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
            self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))
        else:
            self.users = self.train_set['users']
            self.items = self.train_set['items']

        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

    def write_predictions(self):
        """
        Method to write final ranking

        """

        if self.output_file is not None:
            WriteFile(self.output_file, data=self.predictions, sep=self.sep).write()

    def print_header(self, header_info, test_info=None):
        """
        Function to print the header with information of the files

        :param header_info: Dictionary with information about dataset or train file
        :type header_info: dict

        :param test_info: Dictionary with information about test file
        :type test_info: dict

        """

        print("[Case Recommender: %s]\n" % header_info['title'])
        print("train data:: %d users and %d items (%d interactions) | sparsity:: %.2f%%" %
              (header_info['n_users'], header_info['n_items'], header_info['n_interactions'], header_info['sparsity']))

        if test_info is not None:
            print("test data:: %d users and %d items (%d interactions) | sparsity:: %.2f%%\n" %
                  (test_info['n_users'], test_info['n_items'], test_info['n_interactions'], test_info['sparsity']))

    def evaluate(self, metrics, verbose=True, as_table=False, table_sep='\t'):
        """
        Method to evaluate the final ranking

        :param metrics: List of evaluation metrics
        :type metrics: list, default ('MAE', 'RMSE')

        :param verbose: Print the evaluation results
        :type verbose: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        self.evaluation_results = {}

        metrics = list(['MAE', 'RMSE'])

        results = self.evaluate_recommender(predictions=self.predictions,
                                                                    test_set=self.test_set)

        for metric in metrics:
            self.evaluation_results[metric.upper()] = results[metric.upper()]

    def evaluate_recommender(self, predictions, test_set):
        """
        Method to evaluate recommender results. This method should be called by item recommender algorithms

        :param predictions: List with recommender output. e.g. [[user, item, score], [user, item2, score] ...]
        :type predictions: list

        :param test_set: Dictionary with test set information.
        :type test_set: dict

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        predictions_dict = {}

        for sample in predictions:
            predictions_dict.setdefault(sample[0], {}).update({sample[1]: sample[2]})

        return self.ratingEvaluate(predictions_dict, test_set)

    def ratingEvaluate(self, predictions, test_set):
        """
        Method to calculate all the metrics for item recommendation scenario using dictionaries of ranking
        and test set. Use read() in ReadFile to transform your prediction and test files in a dict

        :param predictions: Dict of predictions
        :type predictions: dict

        :param test_set: Dictionary with test set information.
        :type test_set: dict

        :return: Dictionary with all evaluation metrics and results
        :rtype: dict

        """

        eval_results = {}
        predictions_list = []
        test_list = []

        if not self.as_rank:
            # Create All but one set, selecting only one sample from the test set for each user
            if self.all_but_one_eval:
                for user in test_set['users']:
                    # select a random item
                    item = random.choice(test_set['feedback'][user])
                    test_set['feedback'][user] = {item: test_set['feedback'][user][item]}

            for user in predictions:
                for item in predictions[user]:
                    rui_predict = predictions[user][item]
                    rui_test = test_set["feedback"].get(user, {}).get(item, np.nan)
                    if not np.isnan(rui_test):
                        predictions_list.append(rui_predict)
                        test_list.append(float(rui_test))

            eval_results.update({
                'MAE': round(mean_absolute_error(test_list, predictions_list), 6),
                'RMSE': round(np.sqrt(mean_squared_error(test_list, predictions_list)), 6)
            })

            if self.verbose:
                self.print_results(eval_results)

        else:
            new_predict_set = []
            new_test_set = {}

            for user in predictions:
                partial_predictions = []
                for item in predictions[user]:

                    if predictions[user][item] > 3:
                        partial_predictions.append([user, item, predictions[user][item]])

                    if test_set["feedback"].get(user, {}).get(item, 0) > 3:
                        new_test_set.setdefault(user, []).append(item)

                partial_predictions = sorted(partial_predictions, key=lambda x: -x[2])
                new_predict_set += partial_predictions

            new_test_set['items_seen_by_user'] = new_test_set
            new_test_set['users'] = test_set['users']

            self.itemEvaluate(
                new_predict_set, new_test_set)

        return eval_results

    def print_results(self, evaluation_results):
        """
        Method to print the results

        :param evaluation_results: Dictionary with results. e.g. {metric: value}
        :type evaluation_results: dict

        """

        if self.as_table:
            header = ''
            values = ''
            for metric in self.metrics:
                header += metric.upper() + self.table_sep
                values += str(evaluation_results[metric.upper()]) + self.table_sep
            print(header)
            print(values)

        else:
            evaluation = 'Eval:: '
            for metrics in self.metrics:
                evaluation += metrics.upper() + ': ' + str(evaluation_results[metrics.upper()]) + ' '
            print(evaluation)

    def itemEvaluate(self, predictions, test_set):

        self.metrics = list(['PREC', 'RECALL', 'MAP', 'NDCG'])
        self.metrics = [m + '@' + str(n) for m in self.metrics for n in self.n_ranks]
        predictions_dict = {}

        for sample in predictions:
            predictions_dict.setdefault(sample[0], {}).update({sample[1]: sample[2]})
        predictions=predictions_dict
        eval_results = {}
        num_user = len(test_set['users'])
        partial_map_all = None

        if self.all_but_one_eval:
            for user in test_set['users']:
                # select a random item
                test_set['items_seen_by_user'][user] = [random.choice(test_set['items_seen_by_user'].get(user, [-1]))]

        for i, n in enumerate(self.n_ranks):
            if n < 1:
                raise ValueError('Error: N must >= 1.')

            partial_precision = list()
            partial_recall = list()
            partial_ndcg = list()
            partial_map = list()

            for user in test_set['users']:
                hit_cont = 0
                # Generate user intersection list between the recommended items and test.
                list_feedback = set(list(predictions.get(user, []))[:n])
                intersection = list(list_feedback.intersection(test_set['items_seen_by_user'].get(user, [])))

                if len(intersection) > 0:
                    ig_ranking = np.zeros(n)
                    for item in intersection:
                        hit_cont += 1
                        ig_ranking[list(predictions[user]).index(item)] = 1

                    partial_precision.append(precision_at_k([ig_ranking], n))
                    partial_recall.append((float(len(intersection)) / float(len(test_set['items_seen_by_user'][user]))))
                    partial_map.append(mean_average_precision([ig_ranking]))
                    partial_ndcg.append(ndcg_at_k(list(ig_ranking)))

                partial_map_all = partial_map

            # create a dictionary with final results
            eval_results.update({
                'PREC@' + str(n): round(sum(partial_precision) / float(num_user), 6),
                'RECALL@' + str(n): round(sum(partial_recall) / float(num_user), 6),
                'NDCG@' + str(n): round(sum(partial_ndcg) / float(num_user), 6),
                'MAP@' + str(n): round(sum(partial_map) / float(num_user), 6),
                'MAP': round(sum(partial_map_all) / float(num_user), 6)

            })

        if self.verbose:
            self.print_results(eval_results)

        return eval_results

    def create_matrix(self):
        """
        Method to create a feedback matrix

        """

        self.matrix = np.zeros((len(self.users), len(self.items)))

        for user in self.train_set['users']:
            for item in self.train_set['feedback'][user]:
                self.matrix[self.user_to_user_id[user]][self.item_to_item_id[item]] = \
                    self.train_set['feedback'][user][item]

    def compute_similarity(self, transpose=False):
        """
        Method to compute a similarity matrix from original df_matrix

        :param transpose: If True, calculate the similarity in a transpose matrix
        :type transpose: bool, default False

        """

        # Calculate distance matrix
        if transpose:
            similarity_matrix = np.float32(squareform(pdist(self.matrix.T, self.similarity_metric)))
        else:
            similarity_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))

        # Remove NaNs
        similarity_matrix[np.isnan(similarity_matrix)] = 1.0
        # transform distances in similarities. Values in matrix range from 0-1
        similarity_matrix = (similarity_matrix.max() - similarity_matrix) / similarity_matrix.max()

        return similarity_matrix

    def compute_bi(self):
        """
        Method to compute bi values

        bi = (rui - mi - bu) / (regBi + number of interactions)

        """

        self.bi = dict()

        for item in self.items:
            count = 0

            for user in self.train_set['users_viewed_item'].get(item, []):
                self.bi[item] = self.bi.get(item, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                self.train_set['mean_value'] - self.bu.get(user, 0)
                count += 1

            if count > 1:
                self.bi[item] = float(self.bi[item]) / float(self.reg_bi + count)
            elif count == 0:
                self.bi[item] = self.train_set['mean_value']

    def compute_bu(self):
        """
        Method to compute bu values

        bu = (rui - mi - bi) / (regBu + number of interactions)

        """

        self.bu = dict()
        for user in self.users:
            count = 0

            for item in self.train_set['items_seen_by_user'].get(user, []):
                self.bu[user] = self.bu.get(user, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                self.train_set['mean_value'] - self.bi.get(item, 0)
                count += 1

            if count > 1:
                self.bu[user] = float(self.bu[user]) / float(self.reg_bu + count)
            elif count == 0:
                self.bu[user] = self.train_set['mean_value']

    def compute_bui(self):
        """
        Method to compute bui values

        bui = mi + bu + bi
        """

        for user in self.users:
            for item in self.items:
                self.bui.setdefault(user, {}).update(
                    {item: self.train_set['mean_value'] + self.bu.get(user, 0) + self.bi.get(item, 0)})

        del self.bu
        del self.bi