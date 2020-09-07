from pdb import set_trace as pause
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random

from Layer import *

LIM = 10
EPOCAS = 500
LR = 0.1


def set_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True, help="path to input file")
    return vars(ap.parse_args())


def prepare_data(file):
    daux = pd.read_csv(file, header=None)
    daux = np.array(daux)
    X = daux[..., :2]
    D = daux[..., 2:]
    return X, D


def line(x1, theta, bias): return (-theta[0]*x1 - bias)/theta[1]


def set_plot(title, pts, _class, theta, bias):
    plt.title(title)
    plt.ylim(-LIM, LIM)
    plt.xlim(-LIM, LIM)
    for i in range(len(_class)):
        aux = _class[i]
        if aux[0] == 1:
            color = 'b'
        elif aux[1] == 1:
            color = 'g'
        elif aux[2] == 1:
            color = 'r'
        elif len(aux) > 3:
            if aux[3] == 1:
                color = 'yellow'
            elif aux[4] == 1:
                color = 'purple'
        plt.scatter(pts[i, 0], pts[i, 1], c=color, s=10)
    for j in range(len(aux)):
        plt.plot([-LIM, LIM], [line(-LIM, theta[j, ...], bias[j]),
                               line(LIM, theta[j, ...], bias[j])], color='black')


if __name__ == '__main__':
    args = set_parser()
    X, D = prepare_data(args["i"])
    layer = Layer(LR, len(D[0]), len(X[0]))
    set_plot("Before training", X, D, layer.w, layer.b)
    plt.show()
    layer.train(X, D, EPOCAS)
    set_plot("After training", X, D, layer.w, layer.b)
    # plt.savefig('result.png')
    plt.show()
    layer.result_params(X, D)
