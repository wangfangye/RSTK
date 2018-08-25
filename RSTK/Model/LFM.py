import random
import math

class LFM(object):
    def __init__(self, train_set=None, test_set=None, metrics=list(['MAE', 'RMSE']), nfactor=20, steps=10, learn_rate=0.02, reg=0.02):
        self.train_file = train_set
        self.test_file = test_set
        self.nfactor = nfactor
        self.steps = steps
        self.learn_rate = learn_rate
        self.reg = reg

        # internal vars
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.mu = 0
        self.model_name = 'LFM'
        self.metrics = metrics
        self.evaluation = None

    def train(self):
        train = []
        with open(self.train_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split('\t')
                    if len(inline) == 1:
                        raise TypeError("Error: Space type (sep) is invalid!")
                    train.append((int(inline[0]), int(inline[1]), int(inline[2])))
        for u, i, rui in train:
            self.bu[u] = 0
            self.bi[i] = 0
            if u not in self.P:
                self.P[u] = [random.random() / math.sqrt(self.nfactor) for x in range(0, self.nfactor)]
            if i not in self.Q:
                self.Q[i] = [random.random() / math.sqrt(self.nfactor) for x in range(0, self.nfactor)]

        # print('Training LFM.. ')
        for step in range(0, self.steps):
            for u, i, rui in train:
                # compute predict value
                ret = self.mu + self.bu[u] + self.bi[i]
                ret += sum(self.P[u][k] * self.Q[i][k] for k in range(0, len(self.P[u])))
                pui = ret
                # update parameters
                self.bu[u] += self.learn_rate * ((rui - pui) - self.reg * self.bu[u])
                self.bi[i] += self.learn_rate * ((rui - pui) - self.reg * self.bi[i])
                for k in range(0, self.nfactor):
                    self.P[u][k] += self.learn_rate * (self.Q[i][k]*(rui - pui) - self.reg*self.P[u][k])
                    self.Q[i][k] += self.learn_rate * (self.P[u][k]*(rui - pui) - self.reg*self.Q[i][k])
            self.learn_rate *= 0.9

    def predict(self):
        rmse = 0
        mae=0
        num = 0
        # print('Testing LFM.. ')
        test = []
        with open(self.test_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split('\t')
                    if len(inline) == 1:
                        raise TypeError("Error: Space type (sep) is invalid!")
                    test.append((int(inline[0]), int(inline[1]), int(inline[2])))
        for u, i, rui in test:
            if ((u in self.P) and (i in self.Q)):
                ret = self.mu + self.bu[u] + self.bi[i]
                ret += sum(self.P[u][k] * self.Q[i][k] for k in range(0, len(self.P[u])))
                pui = ret
                rmse += (rui - pui) * (rui - pui)
                mae += math.sqrt((rui - pui) * (rui - pui))
                num += 1
        rmse = math.sqrt(rmse / num)
        mae = mae/num
        print('RMSE: ', rmse)
        self.evaluation = {}
        self.evaluation.update({
            'MAE': mae,
            'RMSE': rmse
        })
        # for metric in self.metrics:
        #     self.evaluation_results[metric.upper()] = results[metric.upper()]
        return rmse

    def compute(self):
        self.train()
        self.predict()