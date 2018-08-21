import math


def ModelRmse(train, test, P, Q, bu, bi, mu):
    rmse = 0
    num = 0
    for u, i, rui in test:
        if ((u in P) and (i in Q)):
            rmse += (rui - Predict(u, i, P, Q, bu, bi, mu)) * (rui - Predict(u, i, P, Q, bu, bi, mu))
            num += 1
    rmse = math.sqrt(rmse / num)
    return rmse


def Predict(u, i, P, Q, bu, bi, mu):
    ret = mu + bu[u] + bi[i]
    ret += sum(P[u][f] * Q[i][f] for f in range(0, len(P[u])))
    return ret
