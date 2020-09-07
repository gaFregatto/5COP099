# Hiperplano: separam o espaço em que estão inseridos em 2 lados, um para cada classe.
# No caso de um espaço 1-D, o hiperplano é simplesmente um ponto nesta linha.
# O hiperplano de separação terá sempre uma dimensão a mais (N+1) que o problema (N).

# Equação de um hiperrplano:
# 1-D: ax + b = 0
#
#   [a b] [x] = 0
#         [1]


# 2-D: ax + by + c = 0
#
#           [x]
#   [a b c] [y] = 0
#           [1]

# N-D: c1x1 + c2x2 + ... + cnXn + cn+1 = 0
#
#                       [x1]
#                       [x2]
#   [c1 c2 ... cn cn+1] [..] = 0
#                       [xn]
#                       [ 1]

# Lados do hiperplano: f(x)<0    f(x)>0
#                        -     .   +
# 2D: f(x1,x2)
# 3D: f(x1,x2,x3)

# A partir do ponto que temos valores positivos e negativos é possível realizar uma classificação.
# Função sinal (sign(x)):
#            {+1 :x>0}
# sinal(x) = { 0 :x=0}
#            {-1 :x<0}

# Colocando tudo junto:
# Equação do hiperplano: f(x1,x2) = ax1 + bx2 + c
# +Função sinal
# =Modelo final
# M(x1,x2) = sinal(ax1 + bx1 + c)

# Exemplo:
# f(x) = ax + b -> y = ax + b -> y - f(x) = 0
# Pegando pontos que não estão em cima da reta temos: g(x,y) = y - f(x) -> Retorna a distância do ponto em relação a reta (não é a menor distância)
# Modelo: m(x,y) = sign(g(x,y)) -> Mostra em qual lado do hiperplano o dado de entrada está.

# Podemos adicionar uma função para calcular a distância mínima do ponto em relação a reta:
# Assim podemos mensurar a probabilidade do dado de entrada pertencer a uma classe.
# dist(x,y) = abs(g(x,y))/sqrt(a**2 + b**2)

# Exemplo de um modelo que retorna o quão positivo ou negativo um determinado dado de entrada é em relação a uma classe:
# m(x,y) = dist(x,y) * sign(g(x,y))

# Basicamente, um classificador linear é encontrar o que está do lado positivo ou negativo de um hiperplano.

# Função de Perda, também chamada Função de Custo:
# considerando um conjunto de dados (X, Y), uma Função de Perda calcula o erro entre o nosso modelo e os valores esperados.
# ou seja, o quão distante está o resultado de M(X) em relação a Y.
