import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

LIM = 10
EPOCAS = 10000
LR = 0.1

LAYERS = [4, 4, 2, 3]
# LAYERS[0] representa a camanda de entrada (x1, x2, x3, x4)
# LAYERS[-1] representa camada de saida
# Entradas representam as colunas e as linhas representam as saídas


def prepare_sets(X, D):
    s = X[:50, ...]
    ve = X[50:100, ...]
    vi = X[100:, ...]

    setosa = np.zeros((s.shape[0], s.shape[1]+D.shape[1]))
    setosa[..., :4] = s
    setosa[..., 4:] = D[:50, ...]

    versicolor = np.zeros((ve.shape[0], ve.shape[1]+D.shape[1]))
    versicolor[..., :4] = ve
    versicolor[..., 4:] = D[50:100, ...]

    virginica = np.zeros((vi.shape[0], vi.shape[1]+D.shape[1]))
    virginica[..., :4] = vi
    virginica[..., 4:] = D[100:, ...]

    train = np.zeros((90, setosa.shape[1]))
    train[:30, ...] = setosa[:30, ...]
    train[30:60, ...] = versicolor[:30, ...]
    train[60:, ...] = virginica[:30, ...]

    val = np.zeros((30, setosa.shape[1]))
    val[:10, ...] = setosa[30:40, ...]
    val[10:20, ...] = versicolor[30:40, ...]
    val[20:30, ...] = virginica[30:40, ...]

    test = np.zeros((30, setosa.shape[1]))
    test[:10, ...] = setosa[40:50, ...]
    test[10:20, ...] = versicolor[40:50, ...]
    test[20:30, ...] = virginica[40:50, ...]

    df1 = pd.DataFrame(train)
    df1.to_csv('trainset.csv', header=None, index=False)

    df2 = pd.DataFrame(val)
    df2.to_csv('valset.csv', header=None, index=False)

    df3 = pd.DataFrame(test)
    df3.to_csv('testset.csv', header=None, index=False)

    return train, val, test


def prepare_data(file):
    data = pd.read_csv(file, header=None)
    dt = np.array(data)
    X = dt[..., :4]
    classes = dt[..., 4]

    D = np.zeros((classes.shape[0], 3))
    for i in range(classes.shape[0]):
        if classes[i] == 'Iris-setosa':
            D[i, 0] = 1.
        elif classes[i] == 'Iris-versicolor':
            D[i, 1] = 1.
        elif classes[i] == 'Iris-virginica':
            D[i, 2] = 1.
    return X, D


def sigmoid(x):
    return 1./(1. + math.exp(-x))


def activation(x, w, b):
    result = np.matmul(w, x)
    result = np.add(result, b)
    return np.array([sigmoid(x) for x in result])


def forward(entry, weights, biases):
    result = entry.T
    hs = [result]
    for w, b in zip(weights, biases):
        result = activation(result, w, b.T)
        result = np.mat(result)
        result = result.T
        hs.append(result)
    return hs


def train(X, D, Ep, Lr):

    weights = [np.random.uniform(-1, 1, [LAYERS[i+1], LAYERS[i]])
               for i in range(len(LAYERS)-1)]
    biases = [np.random.uniform(-1, 1, [1, LAYERS[i+1]])
              for i in range(len(LAYERS)-1)]

    # for i in range(len(biases)):
    #     print(f"W{i}")
    #     print(weights[i])
    #     print(f"bias{i}")
    #     print(biases[i])

    for ep in range(Ep):
        for x, d in zip(X, D):
            x = np.mat(x)
            hs = forward(x, weights, biases)
            for i in range(len(hs)):
                print(f"h{i}")
                print(hs[i])
            exit(0)
    return weights, biases


def main():
    # X, D = prepare_data(sys.argv[1])
    # trainset, valset, testset = prepare_sets(X, D)
    X_train, D_train = prepare_data('trainset.csv')
    layers, bias = train(X_train, D_train, EPOCAS, LR)
    return 0


if __name__ == '__main__':
    main()
