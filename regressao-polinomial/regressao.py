import matplotlib.pyplot as plt
import numpy as np

lim, npts = 10, 10

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.grid()
pts = plt.ginput(npts)
plt.close()

pts = np.array(pts)

x = pts[:, 0]
y = pts[:, 1]

# A = np.array([x, np.ones(len(x))]).T  # Regressão linear
# A = np.array([x**2, x, np.ones(len(x))]).T  # Regressão quadrática
# A = np.array([x**3, x**2, x, np.ones(len(x))]).T  # Regressão cúbica
# A = np.array([x**4, x**3, x**2, x, np.ones(len(x))]).T  # 4
# A = np.array([x**5, x**4, x**3, x**2, x, np.ones(len(x))]).T  # 5
A = np.array([x**6, x**5, x**4, x**3, x**2, x, np.ones(len(x))]).T  # 6

m = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
print(m)


# def func(XX):
#     return m[0]*XX + m[1]  # Regressão linear

# def func(XX):
#     return m[0]*(XX**2) + m[1]*XX + m[2]  # Regressão quadrática


# def func(XX):
#     return m[0]*(XX**3) + m[1]*XX**2 +  m[2]*XX + m[3]  # Regressão cúbica


# def func(XX):
#     return m[0]*(XX**4) + m[1]*XX**3 + m[2]*XX**2 + m[3]*XX + m[4]  # 4


# def func(XX):
#     return m[0]*(XX**5) + m[1]*XX**4 + m[2]*XX**3 + m[3]*XX**2 + m[4]*XX + m[5]  # 5


def func(XX):
    # 6
    return m[0]*(XX**6) + m[1]*XX**5 + m[2]*XX**4 + m[3]*XX**3 + m[4]*XX**2 + m[5]*XX + m[6]


X = np.linspace(-lim, lim, 200)
Y = func(X)

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X, Y)
plt.grid()
plt.show()
