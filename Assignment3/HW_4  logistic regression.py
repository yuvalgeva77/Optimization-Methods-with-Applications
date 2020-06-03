# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
import math
from numpy import matmul as mul
from mlxtend.data import mnist_data
import loadMNIST
from numpy.linalg import inv

#f,J is  lambdas
def armijo(x,f,d,J, b=0.5, a_0=1, c=1,max_iter=100):
    a_k = a_0
    for i in range(0,max_iter):
        f_k = mul(f(x) + a_k * d)
        limit = f(x) + c * a_k * mul(J(x).transpose(),d)
        if f_k <= limit:
           break
        a_k = a_k * b
    return a_k

#H, lambdas
def gradient_descent(x_0, H, eps = 0.00001,a = 0.01):
    cur_x = x_0  # The algorithm starts at x0
    previous_step_size = cur_x
    while previous_step_size > eps:
        prev_x = cur_x
        cur_x += -a * H(prev_x)
        previous_step_size = abs(cur_x - prev_x)#TODO check if abs works on vector
    return cur_x

#J,H,f lambdas
def exact_Newton(f, H,J, x_0,  eps = 0.00001):
    x_k=x_0
    while abs(f(x_k)) > eps:
        a=armijo(x_k, f, J)
        d_n=mul(inv(H(x_k)),J(x_k))
        x_k = x_k +a*d_n
    return x_k

def logistic_func(x_i, w):
    return lambda w: 1 / 1 + math.exp(mul(-x_i, w))


def logistic_func_Marix(X, w):
    n, m = X.shape
    res = np.zeros((n, 1))
    for i in range(0, m):
        x_i = X[:, i]  # ith First Column
        res[i] = 1 / 1 + math.exp(mul(-x_i, w))
    return lambda w:res(w)


def logistic_regression_objective(X, y, w):
    n, m = X.shape  # columns
    res = 0
    for i in range(0, m):
        x_i = X[:, i]  # ith First Column
        res =+(y[i] * math.log2(logistic_func(x_i.transpose(), w)) + 1 - y[i]) * math.log2(1 - logistic_func(x_i, w))
    return lambda w:res(w) / m


def gradient(X, Y, w):
    n, m = X.shape  # columns
    return lambda w:X.mul(logistic_func_Marix(X.transpose(), w) - Y[0]) / m


def hessian(X, w):
    n, m = X.shape  # columns
    d1 = logistic_func_Marix(X.transpose(), w)
    d2 = np.ones((m, 1)) - d1
    for i in range(0, m):
        res = d1[i] * d2[i]
    D = p.diag(res)
    return lambda w:X.mul(D.mul(X.transpose())) / m


def task_4a(X, labels):
    J = logistic_regression_objective(X, labels)
    print("logistic_regression_objective  {0}\n".format(J))
    G = gradient(X, labels, w)
    print("gradient is:  {0}\n".format(G))
    H = hessian(X, w)
    print("hessian is:  {0}\n".format(H))
    return (J, G, H)


(X, labels) = loadMNIST.random_shuffeled_Mnist()
w=np.ones(X.shape[])
(J, G, H)=task_4a(X, labels)
G(w)
