import random
import math


def InitLFM(train,k,P,Q,bu,bi):
    for u, i, rui in train:
        bu[u] = 0
        bi[i] = 0
        if u not in P:
            P[u] = [random.random() / math.sqrt(k) for x in range(0, k)]
        if i not in Q:
            Q[i] = [random.random() / math.sqrt(k) for x in range(0, k)]


def LFM(train,k,n,lr,reg,P,Q,bu,bi,mu):
    InitLFM(train,k,P,Q,bu,bi)
    print('Training LFM.. ')
    for step in range(0, n):
        for u, i, rui in train:
            pui = Predict(u, i, P, Q, bu, bi, mu)
            eui = rui - pui
            bu[u] += lr * (eui - reg * bu[u])
            bi[i] += lr * (eui - reg * bi[i])
            for f in range(0, k):
                P[u][f] += lr * (Q[i][f]*eui - reg*P[u][f])
                Q[i][f] += lr * (P[u][f]*eui - reg*Q[i][f])
        lr *= 0.9
        # print('step ', step, ' . ')
        # print('bu : ', bu)
        # print('bi : ', bi)
        # print('mu : ', mu)

def Predict(u,i,P,Q,bu,bi,mu):
    ret = mu + bu[u] + bi[i]
    ret += sum( P[u][f] * Q[i][f] for f in range( 0,len(P[u]) ) )
    return ret