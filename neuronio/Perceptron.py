from pdb import set_trace as pause
import matplotlib.pyplot as plt
import numpy as np
import random
from main import set_plot


class Perceptron:
    def __init__(self, n, learning_rate):
        self.lr = learning_rate
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.f_measure = 0
        if(n == 2):
            self.w = np.array(
                [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])

    def comb_linear(self, x1, x2):
        return x1*self.w[0] + x2*self.w[1] + self.w[2]

    def activation(self, x):
        return np.tanh(x)

    def model(self, x1, x2):
        return self.activation(self.comb_linear(x1, x2))

    # def erro(self, x1, x2, d):
    #     return (self.model(x1, x2)-d)**2

    def grad(self, x1, x2, d):
        u1 = self.model(x1, x2) - d
        u3 = self.comb_linear(x1, x2)

        return np.array([
            2.*u1*(1.-self.activation(u3)**2)*x1,
            2.*u1*(1.-self.activation(u3)**2)*x2,
            2.*u1*(1.-self.activation(u3)**2),
        ])

    def train(self, X, D, Ep):
        for ep in range(Ep):
            for x, d in zip(X, D):
                x1 = x[0]
                x2 = x[1]
                self.w = self.w - self.lr*self.grad(x1, x2, d)/X.shape[0]
            if(ep == 10 or ep == 20):
                set_plot("Época: "+str(ep), X, D, self.w)
                plt.savefig('images/epoca'+str(ep)+'.png', format='png')
                plt.show()

    def result_params(self, X, D):
        pos = neg = tp = tn = fp = fn = 0

        for x, d in zip(X, D):
            x1 = x[0]
            x2 = x[1]
            pred = -1 if self.activation(self.comb_linear(x1, x2)) < 0 else 1

            if pred == 1 and d == 1:
                tp += 1
            elif pred == 1 and d == -1:
                fp += 1
            elif pred == -1 and d == -1:
                tn += 1
            elif pred == -1 and d == 1:
                fn += 1

        # print(tp, tn, fp, fn)
        self.precision = tp / float(tp + fp)
        self.recall = tp / float(tp + fn)
        self.accuracy = (tp + tn) / float(tp + tn + fp + fn)
        self.f_measure = 2*self.precision * \
            self.recall/(self.precision+self.recall)
        print(self.precision, self.recall, self.accuracy, self.f_measure)
        print("Precisão: %.2f\nRevocação: %.2f\nAcurácia: %.2f\nMedida-F: %.2f\n" %
              (self.precision, self.recall, self.accuracy, self.f_measure))
