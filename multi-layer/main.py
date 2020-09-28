import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

EPOCAS = 10000
LR = 0.01

LAYERS = [4, 2, 5, 3]
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


def read_data(file):
    data = pd.read_csv(file, header=None)
    dt = np.array(data)
    X = dt[..., :4]
    D = dt[..., 4:]
    return X, D


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def lin_comb(x, w, b):
    return np.matmul(x, w) + b


def err(d, y):
    return (2 * y - 2 * d) * (y * (1 - y))


def err_sigma(w, s, h):
    return np.matmul(s, w) * (h * (1 - h))


def train(Xtrain, Dtrain, Ep, Lr, Xvalid, Dvalid):

    # Inicialização
    weights = [np.random.uniform(-1, 1, [LAYERS[i], LAYERS[i+1]])
               for i in range(len(LAYERS)-1)]
    biases = [np.random.uniform(-1, 1, [1, LAYERS[i+1]])
              for i in range(len(LAYERS)-1)]

    # Parâmetros finais do modelo
    Wfinal, bfinal = None, None

    # Menor error de validação
    min_e = float('inf')

    # Treinamento
    for ep in range(Ep):

        # forward
        h = []
        for l in range(len(LAYERS)-1):
            if l == 0:
                h.append(sigmoid(lin_comb(Xtrain, weights[l], biases[l])))
            else:
                h.append(sigmoid(lin_comb(h[l-1], weights[l], biases[l])))

        # backward
        sigmas = [None for i in range(len(h))]
        for l in range(len(h)-1, 0, -1):
            if l == len(h)-1:
                sigmas[l] = err(Dtrain, h[l])
            else:
                sigmas[l] = err_sigma(weights[l+1].T, sigmas[l+1], h[l])

            weights[l] = weights[l] - \
                (Lr * np.matmul(h[l-1].T if l > 0 else Xtrain, sigmas[l]))
            biases[l] = biases[l] - (Lr * np.sum(sigmas[l], axis=0))

        for l in range(len(LAYERS)-1):
            Yvalid = sigmoid(
                lin_comb(Xvalid if l == 0 else Yvalid, weights[l], biases[l]))

        error = ((Dvalid-Yvalid)**2).mean()
        if error < min_e:
            epfinal = ep
            min_e = error
            Wfinal = weights.copy()
            bfinal = biases.copy()

        print(f"Época: {ep} || error: {error}")
    print(f"Melhor época: {epfinal} || Melhor erro: {min_e}")
    return Wfinal, bfinal


def main():
    # X, D = prepare_data(sys.argv[1])
    # trainset, valset, testset = prepare_sets(X, D)
    X_train, D_train = read_data('trainset.csv')
    X_valid, D_valid = read_data('valset.csv')
    X_test, D_test = read_data('testset.csv')

    layers, biases = train(X_train, D_train, EPOCAS, LR,
                           X_valid, D_valid)

    for l in range(len(LAYERS)-1):
        Ytest = sigmoid(
            lin_comb(X_test if l == 0 else Ytest, layers[l], biases[l]))

    y_pred = np.where(Ytest > 0.5, 1.0, 0.0)
    errors = (np.argmax(Ytest, -1) != np.argmax(D_test, -1)).sum()
    total = X_test.shape[0]
    print("Taxa de erro: %.2f%%" % (100.*(errors/total)))

    return 0


if __name__ == '__main__':
    main()
