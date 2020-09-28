
import sys
import numpy as np

EPOCAS = 100
LR = 1.


def readCSV(path):
    DIGITS = ["zero", "um", "dois", "tres", "quatro",
              "cinco", "seis", "sete", "oito", "nove"]
    with open(path, 'r') as f:
        X, Y = [], []
        for i, line in enumerate(f):
            if i > 1:
                values = line.strip().split(',')
                X.append(list(map(float, values[:-1])))
                y = np.zeros(10, dtype=float)
                y[DIGITS.index(values[-1])] = 1.
                Y.append(y)
        return np.array(X), np.array(Y)


def lin_comb(X, W, b):
    r = np.matmul(X, W) + b
    return r


def sigmoid(X):
    return 1./(1.+np.exp(-X))


if __name__ == "__main__":

    epochs = EPOCAS  # Quantidade de epocas
    lr = LR  # taxa de aprendizado

    # Carregar dados (formato CSV)
    # X, D = readCSV('data.csv')

    # Salvar dados (formato NumPy)
    # np.save('X',X)
    # np.save('D',D)

    # Carregar dados (formato NumPy)
    X = np.load('X.npy')
    D = np.load('D.npy')

    # Separar conjuntos de treinamento, validação e teste
    idx = np.random.permutation(X.shape[0])
    idx_train = idx[:8000]
    idx_valid = idx[8000:9000]
    idx_test = idx[9000:]
    Xtrain, Dtrain = X[idx_train], D[idx_train]
    Xvalid, Dvalid = X[idx_valid], D[idx_valid]
    Xtest, Dtest = X[idx_test], D[idx_test]

    # Inicialização dos parâmetros
    W = np.random.random((784, 10))*2. - 1.
    b = np.random.random((1, 10))*2. - 1.

    # Parâmetros finais do modelo
    Wfinal, bfinal = None, None

    # Menor error de validação
    min_validation_error = float('inf')

    # Treinamento
    for ep in range(epochs):

        # Forward (conjunto de treinamento)
        Ytrain = sigmoid(lin_comb(Xtrain, W, b))

        # Backward (conjunto de treinamento)
        grad_error = 2*(Ytrain - Dtrain)
        grad_act = Ytrain*(1. - Ytrain)
        grad = grad_error*grad_act
        gradW = np.matmul(grad.T, Xtrain)/Xtrain.shape[0]
        gradb = np.sum(grad, axis=0)/Xtrain.shape[0]

        # Atualização de pesos
        W = W - lr*gradW.T
        b = b - lr*gradb

        # Error por época (conjunto de validação)
        Yvalid = sigmoid(lin_comb(Xvalid, W, b))
        print(Xvalid.shape)
        print(W.shape)
        print(b.shape)
        print(Yvalid.shape)
        print(Dvalid.shape)
        exit(0)
        error = ((Dvalid-Yvalid)**2).mean()

        # Salvar pesos se menor erro de validação
        if error < min_validation_error:
            min_validation_error = error
            Wfinal = W.copy()
            bfinal = b.copy()

        print("Epoca %d: %f (min: %f)" % (ep, error, min_validation_error))

    # Estatisticas do modelo (conjunto de testes)
    Ytest = sigmoid(lin_comb(Xtest, Wfinal, bfinal))
    errors = (np.argmax(Ytest, -1) != np.argmax(Dtest, -1)).sum()
    total = Xtest.shape[0]

    print("Taxa de erro: %.2f%%" % (100.*(errors/total)))
