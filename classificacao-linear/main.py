import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from pdb import set_trace as pause

# ENTRADA
# -> Parâmetros: [a,b,c] (inicializar com valores aleatórios)
# -> Função: G(a,b,c)
# -> Conjunto de dados (X,Y)

# ROTINA
# -> Para cada época
#     -> Para cada amostra (x,y) pertencente a (X,Y)
#         -> [a,b,c] = [a,b,c] - lamba*derivadaG(a,b,c)(x,y)/|X|
#     -> erro_total = 1/|X| * somatório(x,y) pertencente a (X,Y)^E(x,y)
#     -> Imprimir "Erro nesta época: "+erro_total


LR = 0.1
EPOCAS = 20


def line(x1, Theta): return (-Theta[0]*x1 - Theta[2])/Theta[1]


def comb_linear(x1, x2, Theta):
    return x1*Theta[0] + x2*Theta[1] + Theta[2]


def activation(x):
    return np.tanh(x)


def model(x1, x2, Theta):
    return activation(comb_linear(x1, x2, Theta))


def erro(x1, x2, y, Theta):
    return (model(x1, x2, Theta)-y)**2


def grad(x1, x2, y, Theta):
    u1 = model(x1, x2, Theta) - y
    u3 = comb_linear(x1, x2, Theta)

    return np.array([
        2.*u1*(1.-activation(u3)**2)*x1,
        2.*u1*(1.-activation(u3)**2)*x2,
        2.*u1*(1.-activation(u3)**2),
    ])


# Carrega dataset
# data = np.load("data.npy")
data = np.array([[0.29274194, 2.16785714, 1.],
                 [-0.94475806, 2.32857143, 1.],
                 [-1.72983871, 0.54285714, 1.],
                 [-0.71854839, 1.09642857, 1.],
                 [-0.58548387, 2.16785714, 1.],
                 [0.59879032, 2.70357143, 1.],
                 [1.54354839, 2.23928571, 1.],
                 [-0.26612903, 1.20357143, 1.],
                 [-1.53024194, 1.45357143, 1.],
                 [-2.36854839, -0.74285714, 1.],
                 [-0.91814516, 0.54285714, -1.],
                 [-0.7983871, -0.92142857, -1.],
                 [0.77177419, 0.61428571, -1.],
                 [1.78306452, 1.20357143, -1.],
                 [2.24879032, -0.29642857, -1.],
                 [1.66330645, -1.92142857, -1.],
                 [1.25080645, -0.56428571, -1.],
                 [1.72983871, 0.59642857, -1.],
                 [0.53225806, -0.36785714, -1.],
                 [0.87822581, -1.61785714, -1.]])

# Separa valor dos pontos em X e as classes em Y
X = data[..., :2]
Y = data[..., 2]

plt.plot(X[Y > 0, 0], X[Y > 0, 1], '+b')  # Classe positiva
plt.plot(X[Y < 0, 0], X[Y < 0, 1], '+r')  # Classe negativa

# Construir nossa reta (gerando valores aleatórios para os parâmetros)
a = random.random()
b = random.random()
c = random.random()
Theta = np.array([a, b, c])

plt.plot([-3, 3], [line(-3, Theta), line(3, Theta)])
plt.ylim(-3, 3)
plt.title("Hiperplano gerado com parâmetros aleatórios")
plt.show()

for epoca in range(EPOCAS):
    for x, y in zip(X, Y):
        x1 = x[0]
        x2 = x[1]
        Theta = Theta - LR*grad(x1, x2, y, Theta)/X.shape[0]

    plt.plot(X[Y > 0, 0], X[Y > 0, 1], '+b')  # Classe positiva
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], '+r')  # Classe negativa
    plt.plot([-3, 3], [line(-3, Theta), line(3, Theta)])
    plt.ylim(-3, 3)
    plt.title("Época: "+str(epoca))
    plt.show()
