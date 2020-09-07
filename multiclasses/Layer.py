from pdb import set_trace as pause
import matplotlib.pyplot as plt
import numpy as np
import random
from main import set_plot
import sys
import os


class Layer:
    def __init__(self, learning_rate, m, dims):
        self.lr = learning_rate
        self.w = np.random.uniform(-1, 1, [m, dims])
        self.b = np.random.uniform(-1, 1, [m])

    # sigmoid
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def activation(self, x):
        r = np.dot(self.w, x)
        r = np.add(r, self.b)
        return np.array([self.sigmoid(x) for x in r])

    def comb_linear(self, x):
        return np.dot(x, self.w)

    def error(self, d, y, i):
        r = (2 * y[i] - 2 * d[i]) * (y[i] * (1 - y[i]))
        return r

    def train(self, X, D, Ep):
        for ep in range(Ep):
            for x, d in zip(X, D):
                y = self.activation(x)
                # os.system('pause')
            for i in range(len(d)):
                for j in range(len(x)):
                    self.w[i][j] = self.w[i][j] - \
                        self.lr * self.error(d, y, i) * x[j]
                self.b[i] = self.b[i] - self.lr * self.error(d, y, i)
                # set_plot("Época: "+str(ep), X, D, self.w, self.b)
                # plt.show()

    def result_params(self, X, D):
        tp = tn = fp = fn = 0

        for x, d in zip(X, D):
            x1 = x[0]
            x2 = x[1]
            pred = 0 if self.activation2(self.comb_linear2(x1, x2)) < .5 else 1

            if pred == 1 and d == 1:
                tp += 1
            elif pred == 1 and d == 0:
                fp += 1
            elif pred == 0 and d == 0:
                tn += 1
            elif pred == 0 and d == 1:
                fn += 1

            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            accuracy = (tp + tn) / float(tp + tn + fp + fn)
            f_measure = 2*precision * \
                recall/(precision+recall)
            print(precision, recall, accuracy, f_measure)
            print("Precisão: %.2f\nRevocação: %.2f\nAcurácia: %.2f\nMedida-F: %.2f\n" %
                  (precision, recall, accuracy, f_measure))
