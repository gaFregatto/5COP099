import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

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
        if aux[0] == 1:
            color = 'b'
        elif aux[1] == 1:
            color = 'g'
        elif aux[2] == 1:
            color = 'r'
        elif D.shape[1] > 3:
            if aux[3] == 1:
                color = 'yellow'
            elif aux[4] == 1:
                color = 'purple'
        plt.scatter(X[i, 0], X[i, 1], c=color, s=10)

    # print lines
    for j in range(bias.shape[1]):
        if j == 0:
            color = 'b'
        elif j == 1:
            color = 'g'
        elif j == 2:
            color = 'r'
        elif j == 3:
            color = 'yellow'
        elif j == 4:
            color = 'purple'
        plt.plot([-LIM, LIM], [line(-LIM, theta[j, ...], bias[0, j]),
                               line(LIM, theta[j, ...], bias[0, j])], color=color)

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
    n = X.shape[1]
    m = D.shape[1]
    weights = np.random.uniform(-1, 1, [m, n])
    bias = np.random.uniform(-1, 1, [1, m])
    plot_all("Before trainig", X, D, weights, bias)

    for ep in range(Ep):
        for x, d in zip(X, D):
            x = np.mat(x)
            y = activation(x.T, weights, bias.T)
            # y = np.mat(y)

            for i in range(m):
                for j in range(n):
                    weights[i][j] = weights[i][j] - LR * \
                        err(d, y, i) * x[0, j]/X.shape[0]
                bias[0, i] = bias[0, i] - LR * err(d, y, i)/X.shape[0]

    plot_all("After trainig", X, D, weights, bias)
    return weights, bias


def metrics():
    exit(1)


def main():
    X, D = prepare_data(sys.argv[1])
    neurons, bias = train(X, D, EPOCAS, LR)
    exit(0)


if __name__ == '__main__':
    main()
