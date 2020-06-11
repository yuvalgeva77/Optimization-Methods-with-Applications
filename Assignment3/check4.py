import numpy as np
import math
from numpy import matmul as mul
from numpy.linalg import inv
from random import gauss
import loadMNIST


def sigmoid(x_i, w):
    return (1 / (1 + math.exp(mul(x_i, w))))


# preforms logistic_func on each row on array
# return [logistic_func(x_1, w)|...|logistic_func(x_m, w)].traspose()  Rmx1
def sigmoid_Marix(X, w):
    dim1 = (1 / (1 + np.exp(np.dot(-X, w))))
    np.reshape(dim1, (dim1.size, 1))
    # return  np.vstack(([np.array(sigmoid(-X[i],w)) for i in range(0, X.shape[0])]))
    return np.reshape(dim1, (dim1.size, 1))


def sigmoid_Minus1(X, w):
    # return np.vstack(([np.array(1-sigmoid(-X[i],w)) for i in range(0, X.shape[0])]))
    return 1 - sigmoid_Marix(X, w)


# scalar
def f(X, Y, w):
    n, m = X.shape  # columns
    c1 = Y
    c2 = Y
    oppsite = lambda y: 1 - y
    vfunc = np.vectorize(oppsite)
    c2 = vfunc(c2)
    sigmoid_Marix(X.transpose(), w)
    (sigmoid_Minus1(X.transpose(), w))
    mul(c1.transpose(), np.log(sigmoid_Marix(X.transpose(), w))) + mul(c2.transpose(),
                                                                       np.log(sigmoid_Minus1(X.transpose(), w)))
    # return (-1/m)*mul(c1.transpose(),np.apply_along_axis(math.log, 1, (sigmoid_Marix(X.transpose(),w))) +mul(c2.transpose() ,np.apply_along_axis(math.log, 1, (sigmoid_Minus1(X.transpose(),w)))))#TODO np.log
    return (-1 / m) * (mul(c1.transpose(), np.log(sigmoid_Marix(X.transpose(), w))) + mul(c2.transpose(), np.log(
        sigmoid_Minus1(X.transpose(), w))))


# vector size n: 784
def gradient(X, Y, w):
    (1 / m) * (np.dot(X, sigmoid_Marix(X.transpose(), w) - Y))

    return (mul(X, sigmoid_Marix(X.transpose(), w) - Y)) / m


# matrix sized n*n:784*784
def hessian(X, w):
    # np.multiply -Multiply arguments element-wise.
    n, m = X.shape  # columns
    # D=np.diag(logistic_func_Marix(X.transpose())( w).mul(logistic_func_Marix_Minus1(X.transpose())( w)))
    # X.mul(D.mul(X.transpose())) / m
    D = np.diagflat(np.multiply(sigmoid_Marix(X.transpose(), w), (sigmoid_Minus1(X.transpose(), w))))
    D_Xt = mul(D, X.transpose())
    x_D_Xt = mul(X, D_Xt)
    return (1 / m) * x_D_Xt


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]

#-------------------
(X, labels) = loadMNIST.random_shuffeled_Mnist()
n, m = X.shape
d = np.array(make_rand_vector(n))
eps = 1
factor = 0.5
limit = 6
print("A")
w_0 = np.zeros((n, 1))
w_i = w_0 + eps * d
eps_i = (factor ** 0) * eps
d = np.reshape(d, (d.size, 1))
w_i = w_0 + eps_i * d

slope = (eps * d)
f(X, labels, w_i)  # check id to send x.transpose()
print("s")
f(X, labels, w_0)
print("s")

res_1_prev = abs(f(X, labels, w_i) - f(X, labels, w_0))
res_2_prev = abs(f(X, labels, w_i) - f(X, labels, w_0) - eps * mul(d.transpose(), (gradient(X, labels, w_0))))
res_3_prev = np.linalg.norm(gradient(X, labels, w_i) - gradient(X, labels, w_0))
res_4_prev = np.linalg.norm(((gradient(X, labels, w_i) - gradient(X, labels, w_0)).transpose() - (eps * mul(
    d.transpose(), (hessian(X, w_0))))))
for i in range(1, limit):
    eps_i = (factor ** i) * eps
    w_i = w_0 + eps_i * d
    res1 = abs(f(X, labels, w_i) - f(X, labels, w_0))
    res2 = abs((f(X, labels, w_i) - f(X, labels, w_0) - eps_i * mul(d.transpose(), (gradient(X, labels, w_0)))))

    if (abs((res_1_prev / res1) - 1 / factor) > 1 or abs((res_2_prev / res2) - 1 / (factor ** 2)) > 1):
        print("failed The gradient test with eps:{0}, iteration {1}".format(eps, i))

    res3 = np.linalg.norm(gradient(X, labels, w_i) - gradient(X, labels,
                                                             w_0))
    res4 = np.linalg.norm(((gradient(X, labels, w_i) - gradient(X, labels, w_0)).transpose() - eps_i * mul(
        d.transpose(), (hessian(X, w_0)))))
#todo abs((res_4_prev / res4) - 1 / (factor ** 2)) is about 3-4 not 1 check whay 2*(factor ** 2)
    if (abs((res_3_prev / res3) - 1 / factor) > 1 or abs((res_4_prev / res4) - 1 / (factor ** 2)) > 1):
        print("failed The Jacobian test with eps:{0}, iteration {1}, res is ({2},{3})\n".format(eps, i,abs((res_3_prev / res3) - 1 / factor),abs((res_4_prev / res4) - 1 / (factor ** 2))))
    res_1_prev = res1
    res_2_prev = res2
    res_3_prev = res3
    res_4_prev = res4
print("seccesed The gradient and the jacobian test  with eps:{0}, iteration {1}".format(eps, i))
