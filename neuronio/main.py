import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

LIM = 4
EPOCAS = 1000
LR = 0.1


def prepare_data(file):
    data = pd.read_csv(file, header=None)
    d = np.array(data)
    X = d[..., :2]
    D = d[..., 2]

    for i in range(D.shape[0]):
        if D[i] == -1:
            D[i] = 0

    return X, D


def data_input(n_input):
    # plt.xlim(-LIM, LIM)
    # plt.ylim(-LIM, LIM)
    # plt.grid()
    # pts = plt.ginput(n_input)
    # plt.close()
    # x = np.array(pts)
    x = np.array([[-2.08064516,  0.24675325],
                  [-1.82258065,  0.8961039],
                  [-0.69354839,  2.54112554],
                  [0.11290323,  3.01731602],
                  [0.79032258,  2.75757576],
                  [0.0483871,   1.37229437],
                  [1.17741935,  1.67532468],
                  [2.27419355, 1.89177489],
                  [-0.40322581,  0.46320346],
                  [-1.01612903, - 0.83549784],
                  [0.5,         0.24675325],
                  [0.72580645, - 2.48051948],
                  [2.40322581, - 0.66233766],
                  [3.11290323, - 1.13852814],
                  [3.59677419, - 3.],
                  [0.56451613,  1.24242424],
                  [-0.5, - 0.27272727],
                  [2.33870968, - 2.17748918],
                  [2.72580645,  0.33333333],
                  [2.9516129, - 1.87445887],
                  [1.14516129, - 2.61038961],
                  [2.66129032, - 3.3030303],
                  [3.46774194, - 2.004329],
                  [2.24193548, - 1.57142857],
                  [3.17741935, - 0.4025974],
                  [3.11290323, - 2.26406926],
                  [2.14516129, - 2.82683983],
                  [2.88709677, - 2.91341991]])
    d = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return x, d


def line(x1, theta): return (-theta[0]*x1 - theta[2])/theta[1]


def plot_all(title, pts, _class, theta):
    plt.title(title)
    plt.ylim(-LIM, LIM)
    plt.xlim(-LIM, LIM)
    plt.plot([-LIM, LIM], [line(-LIM, theta), line(LIM, theta)], color='black')
    plt.plot(pts[_class > .5, 0], pts[_class > .5, 1], '+r')
    plt.plot(pts[_class < .5, 0], pts[_class < .5, 1], '+b')
    origin = plt.Circle((0, 0), 0.07, color="black")
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(origin)
    plt.show()


def sigmoid(x):
    return 1./(1. + math.exp(-x))


def activation(x):
    # return np.tanh(x)
    return sigmoid(x)


def comb_linear(x, weights):
    return x[0]*weights[0] + x[1]*weights[1] + weights[2]


def err(d, y):
    return (2 * y - 2 * d) * (y * (1 - y))


def metrics(neuron, X, D):
    tp = fp = tn = fn = 0

    for x, d in zip(X, D):
        pred = 0 if activation(comb_linear(x, neuron)) < 0.5 else 1

        if pred == 1 and d == 1:
            tp += 1
        elif pred == 1 and d == 0:
            fp += 1
        elif pred == 0 and d == 0:
            tn += 1
        elif pred == 0 and d == 1:
            fn += 1
    # print(tp, tn, fp, fn)
    # plot_all("After training", X, D, neuron)

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    f_measure = 2*precision*recall/(precision + recall)
    # print(
    #     f"Precisão: {precision}\nRevocação: {recall}\nAcurácia: {accuracy}\nMedida-F: {f_measure}")
    print("Precisão: %.2f\nRevocação: %.2f\nAcurácia: %.2f\nMedida-F: %.2f\n" %
          (precision, recall, accuracy, f_measure))


def train(X, D, Ep, LR):
    theta = np.array(
        [rd.uniform(-1, 1), rd.uniform(-1, 1), rd.uniform(-1, 1)])
    plot_all("Before training", X, D, theta)

    for ep in range(Ep):
        for x, d in zip(X, D):
            y = activation(comb_linear(x, theta))
            # for i in range(len(theta[:-1])):
            #     theta[i] = theta[i] - LR * err(d, y) * x[i]/X.shape[0]
            theta[:-1] = theta[:-1] - LR * err(d, y) * x / X.shape[0]
            theta[-1] = theta[-1] - LR * err(d, y) / X.shape[0]
        # plot_all(f"Epoca {ep}", X, D, theta)
    plot_all("After training", X, D, theta)
    return theta


def main():
    X, D = prepare_data(sys.argv[1])
    # X, D = data_input(28)
    neuron = train(X, D, EPOCAS, LR)
    metrics(neuron, X, D)
    return 0


if __name__ == '__main__':
    main()
