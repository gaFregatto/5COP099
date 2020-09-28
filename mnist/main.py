import os
import sys
import numpy as np

EPOCAS = 100
LR = 0.2

LAYERS = [784, 100, 50, 30, 10]


def readCSV(path):
    DIGITS = ["zero", "um", "dois", "tres", "quatro",
              "cinco", "seis", "sete", "oito", "nove"]
    with open(path, "r") as f:
        X, Y = [], []
        for i, line in enumerate(f):
            if i > 1:
                values = line.strip().split(',')
                X.append(list(map(float, values[:-1])))
                y = np.zeros(10, dtype=float)
                y[DIGITS.index(values[-1])] = 1.
                Y.append(y)
        return np.array(X), np.array(Y)


def separate_sets(X, D):
    idx = np.random.permutation(X.shape[0])
    idx_train = idx[:8000]
    idx_valid = idx[8000:9000]
    idx_test = idx[9000:]
    X_train, D_train = X[idx_train], D[idx_train]
    X_valid, D_valid = X[idx_valid], D[idx_valid]
    X_test, D_test = X[idx_test], D[idx_test]
    return X_train, D_train, X_valid, D_valid, X_test, D_test


def lin_comb(x, w, b):
    return np.matmul(x, w) + b


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def err(d, y):
    return (2 * y - 2 * d) * (y * (1 - y))


def err_sigma(w, s, h):
    return np.matmul(s, w.T) * (h * (1 - h))


def train(X, D, Ep, Lr):
    weights = [np.random.uniform(-1, 1, [LAYERS[i], LAYERS[i+1]])
               for i in range(len(LAYERS)-1)]
    biases = [np.random.uniform(-1, 1, [1, LAYERS[i+1]])
              for i in range(len(LAYERS)-1)]

    print(LAYERS)
    # print(X.shape)

    # for i in range(len(biases)):
    #     print(f"W{i}")
    #     print(weights[i].shape)
    #     print(f"bias{i}")
    #     print(biases[i].shape)

    # exit(0)

    for ep in range(Ep):
        h = [X]
        for l in range(0, len(LAYERS)-1):
            r = sigmoid(lin_comb(h[l], weights[l], biases[l]))
            h.append(r)

        # for i in range(len(h)):
        #     print(f"h{i}")
        #     print(h[i])

        sigmas = [None for i in range(len(h))]
        for l in range(len(h)-1, 0, -1):
            if l == len(h)-1:
                print(D.shape)
                print()
                print(h[l].shape)
                exit(0)
                sigmas[l] = err(D, h[l])
            else:
                sigmas[l] = err_sigma(weights[l], sigmas[l+1], h[l])

            weights[l-1] = weights[l-1] - (Lr * np.matmul(h[l-1].T, sigmas[l]))
            biases[l-1] = biases[l-1] - Lr * sigmas[l]
        # print(weights)
        # print()
        # print(biases)
        print(f"Época: {ep}")

    return weights, biases


def main():
    # X, D = readCSV('data.csv')

    # np.save('X', X)
    # np.save('D', D)

    X = np.load('X.npy')
    D = np.load('D.npy')

    X_train, D_train, X_valid, D_valid, X_test, D_test = separate_sets(X, D)
    layers, biases = train(X_train, D_train, EPOCAS, LR)


if __name__ == "__main__":
    main()
