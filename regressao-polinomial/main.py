import matplotlib.pyplot as plt
import numpy as np

lim, npts = 10, 40

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.grid()
pts = plt.ginput(npts)
plt.close()

pts = np.array(pts)
print(pts)
f = open('./3/pontos.txt', 'a')
f.write(str(pts))

x = pts[:, 0]
y = pts[:, 1]

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, '+')
plt.grid()
plt.savefig('./3/entrada.png')
plt.close()

A = np.array([x, np.ones(len(x))]).T  # Regressão linear
A2 = np.array([x**2, x, np.ones(len(x))]).T  # Regressão quadrática
A3 = np.array([x**3, x**2, x, np.ones(len(x))]).T  # Regressão cúbica
A4 = np.array([x**4, x**3, x**2, x, np.ones(len(x))]).T  # 4
A5 = np.array([x**5, x**4, x**3, x**2, x, np.ones(len(x))]).T  # 5
A6 = np.array([x**6, x**5, x**4, x**3, x**2, x, np.ones(len(x))]).T  # 6
A12 = np.array([x**12, x**11, x**10, x**9, x**8, x**7, x**6, x **
                5, x**4, x**3, x**2, x, np.ones(len(x))]).T  # 12

m = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
m2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A2.T, A2)), A2.T), y)
m3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A3.T, A3)), A3.T), y)
m4 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A4.T, A4)), A4.T), y)
m5 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A5.T, A5)), A5.T), y)
m6 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A6.T, A6)), A6.T), y)
m12 = np.matmul(np.matmul(np.linalg.inv(np.matmul(A12.T, A12)), A12.T), y)


def reg1(XX):
    return m[0]*XX + m[1]  # Regressão linear


def reg2(XX):
    return m2[0]*(XX**2) + m2[1]*XX + m2[2]  # Regressão quadrática


def reg3(XX):
    return m3[0]*(XX**3) + m3[1]*XX**2 + m3[2]*XX + m3[3]  # Regressão cúbica


def reg4(XX):
    return m4[0]*(XX**4) + m4[1]*XX**3 + m4[2]*XX**2 + m4[3]*XX + m4[4]  # 4


def reg5(XX):
    # 5
    return m5[0]*(XX**5) + m5[1]*XX**4 + m5[2]*XX**3 + m5[3]*XX**2 + m5[4]*XX + m5[5]


def reg6(XX):
    # 6
    return m6[0]*(XX**6) + m6[1]*XX**5 + m6[2]*XX**4 + m6[3]*XX**3 + m6[4]*XX**2 + m6[5]*XX + m6[6]


def reg12(XX):
    # 6
    return m12[0]*(XX**12) + m12[1]*XX**11 + m12[2]*XX**10 + m12[3]*XX**9 + m12[4]*XX**8 + m12[5]*XX**7 + m12[6]*XX**6 + m12[7]*XX**5 + m12[8]*XX**4 + m12[9]*XX**3 + m12[10]*XX**2 + m12[11]*XX + m12[12]


X = np.linspace(-lim, lim, 200)
Y = reg1(X)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X, Y)
plt.grid()
# plt.show()
plt.savefig('./3/reg1.png')
plt.close()

X2 = np.linspace(-lim, lim, 200)
Y2 = reg2(X2)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X2, Y2)
plt.grid()
# plt.show()
plt.savefig('./3/reg2.png')
plt.close()

X3 = np.linspace(-lim, lim, 200)
Y3 = reg3(X3)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X3, Y3)
plt.grid()
# plt.show()
plt.savefig('./3/reg3.png')
plt.close()

X4 = np.linspace(-lim, lim, 200)
Y4 = reg4(X4)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X4, Y4)
plt.grid()
# plt.show()
plt.savefig('./3/reg4.png')
plt.close()

X5 = np.linspace(-lim, lim, 200)
Y5 = reg5(X5)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X5, Y5)
plt.grid()
# plt.show()
plt.savefig('./3/reg5.png')
plt.close()

X6 = np.linspace(-lim, lim, 200)
Y6 = reg6(X5)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
plt.plot(X6, Y6)
plt.grid()
# plt.show()
plt.savefig('./3/reg6.png')
plt.close()

X12 = np.linspace(-lim, lim, 200)
Y12 = reg12(X12)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.plot(x, y, "+")
# plt.plot(X4, Y4)
# plt.plot(X5, Y5)
plt.plot(X12, Y12)
plt.grid()
# plt.show()
plt.savefig('./3/reg12.png')
plt.close()
