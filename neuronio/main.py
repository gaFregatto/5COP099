from pdb import set_trace as pause
from Perceptron import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random

LIM = 4
EPOCAS = 30
LR = 0.2


def set_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True, help="path to input file")
    return vars(ap.parse_args())


def normalize(matrix):
    return (matrix/np.sum(matrix))*-20


def prepare_data(file):
    daux = pd.read_csv(file, header=None)
    x1 = daux[0].str.replace('.', '')
    x2 = daux[1].str.replace('.', '')

    X = {'x1': x1, 'x2': x2}
    X = np.array(pd.DataFrame(X))
    X = X.astype(np.float)
    X_norm = normalize(X)

    D = np.array(daux)
    D = D[..., 2]
    D = D.astype(np.float)

    return X_norm, D


def line(x1, theta): return (-theta[0]*x1 - theta[2])/theta[1]


def set_plot(title, pts, _class, theta):
    plt.title(title)
    plt.ylim(-LIM, LIM)
    plt.xlim(-LIM, LIM)
    plt.plot([-LIM, LIM], [line(-LIM, theta), line(LIM, theta)], color='black')
    plt.plot(pts[_class > 0, 0], pts[_class > 0, 1], '+r')
    plt.plot(pts[_class < 0, 0], pts[_class < 0, 1], '+b')


def data_input(n_input):
    plt.xlim(-LIM, LIM)
    plt.ylim(-LIM, LIM)
    plt.grid()
    pts = plt.ginput(n_input)
    plt.close()
    x = np.array(pts)
    # x = np.array([[-2.08064516,  0.24675325],
    #               [-1.82258065,  0.8961039],
    #               [-0.69354839,  2.54112554],
    #               [0.11290323,  3.01731602],
    #               [0.79032258,  2.75757576],
    #               [0.0483871,   1.37229437],
    #               [1.17741935,  1.67532468],
    #               [2.27419355, 1.89177489],
    #               [-0.40322581,  0.46320346],
    #               [-1.01612903, - 0.83549784],
    #               [0.5,         0.24675325],
    #               [0.72580645, - 2.48051948],
    #               [2.40322581, - 0.66233766],
    #               [3.11290323, - 1.13852814],
    #               [3.59677419, - 3.],
    #               [0.56451613,  1.24242424],
    #               [-0.5, - 0.27272727],
    #               [2.33870968, - 2.17748918],
    #               [2.72580645,  0.33333333],
    #               [2.9516129, - 1.87445887],
    #               [1.14516129, - 2.61038961],
    #               [2.66129032, - 3.3030303],
    #               [3.46774194, - 2.004329],
    #               [2.24193548, - 1.57142857],
    #               [3.17741935, - 0.4025974],
    #               [3.11290323, - 2.26406926],
    #               [2.14516129, - 2.82683983],
    #               [2.88709677, - 2.91341991]])
    d = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1., -1, -1, -1, -1, -1, -1, -1, -1])
    return x, d


# EXEC: python main.py -i data.csv
if __name__ == '__main__':
    args = set_parser()
    X, D = prepare_data(args["i"])

    # X, D = data_input(28)

    neuron = Perceptron(2, LR)
    set_plot("Before training", X, D, neuron.w)
    plt.savefig('images/before-training.png', format='png')
    plt.show()
    neuron.train(X, D, EPOCAS)
    set_plot("After training", X, D, neuron.w)
    plt.savefig('images/after-training.png', format='png')
    print(neuron.w)
    plt.show()
    neuron.result_params(X, D)
