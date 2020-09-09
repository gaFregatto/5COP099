import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

COLORS = ['blue', 'green', 'red', 'yellow', 'purple']
LIM = 10
EPOCAS = 1000
LR = 0.1


def line(x1, theta, bias): return (-theta[0]*x1 - bias)/theta[1]


def prepare_data(file):
    data = pd.read_csv(file, header=None)
    dt = np.array(data)
    X = dt[..., :2]
    D = dt[..., 2:]
    return X, D


def plot_all(title, X, D, theta, bias):
    plt.title(title)
    plt.ylim(-LIM, LIM)
    plt.xlim(-LIM, LIM)

    # print points (x1, x2)
    for i in range(len(D)):
        aux = D[i]
        for j in range(len(aux)):
            if aux[j] == 1:
                plt.scatter(X[i, 0], X[i, 1], c=COLORS[j], s=10)

    # print lines
    for j in range(bias.shape[1]):
        plt.plot([-LIM, LIM], [line(-LIM, theta[j, ...], bias[0, j]),
                               line(LIM, theta[j, ...], bias[0, j])], color=COLORS[j])

    # print origin (0, 0)
    origin = plt.Circle((0, 0), 0.07, color="black")
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(origin)

    plt.show()


def sigmoid(x):
    return 1./(1. + math.exp(-x))


def activation(x, w, b):
    result = np.matmul(w, x)
    result = np.add(result, b)
    return np.array([sigmoid(x) for x in result])


def err(d, y, m):
    return (2 * y[m] - 2 * d[m]) * (y[m] * (1 - y[m]))


def train(X, D, Ep, LR):
    N = X.shape[1]
    M = D.shape[1]

    weights = np.random.uniform(-1, 1, [M, N])
    bias = np.random.uniform(-1, 1, [1, M])
    plot_all("Before trainig", X, D, weights, bias)

    for ep in range(Ep):
        for x, d in zip(X, D):
            x = np.mat(x)
            y = activation(x.T, weights, bias.T)
            # y = np.mat(y)

            for m in range(M):
                for n in range(N):
                    weights[m][n] = weights[m][n] - LR * \
                        err(d, y, m) * x[0, n]/X.shape[0]
                bias[0, m] = bias[0, m] - LR * err(d, y, m)/X.shape[0]
    # plot_all("After trainig", X, D, weights, bias)

    return weights, bias


def metrics(neurons, bias, X, D):
    M = D.shape[1]

    for i in range(M):
        tp = fp = tn = fn = 0
        for x, d in zip(X, D):
            x = np.mat(x)
            ps = activation(x.T, neurons, bias.T)
            p = 0 if ps[i] < .5 else 1

            if p == 1 and d[i] == 1:
                tp += 1
            elif p == 1 and d[i] == 0:
                fp += 1
            elif p == 0 and d[i] == 0:
                tn += 1
            elif p == 0 and d[i] == 1:
                fn += 1

        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        if (precision + recall == 0):
            f_measure = -1
        else:
            f_measure = 2*precision*recall/float(precision + recall)

        print("CLASSE: %s\nPrecisão: %.2f\nRevocação: %.2f\nAcurácia: %.2f\nMedida-F: %.2f\n\n" %
              (COLORS[i], precision, recall, accuracy, f_measure))


def main():
    X, D = prepare_data(sys.argv[1])
    neurons, bias = train(X, D, EPOCAS, LR)
    metrics(neurons, bias, X, D)
    plot_all("After trainig", X, D, neurons, bias)
    exit(0)


if __name__ == '__main__':
    main()
