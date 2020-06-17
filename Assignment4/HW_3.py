import numpy as np
from numpy.linalg import inv
from math import exp as exp
from numpy import matmul as mul
from numpy import linalg as LA
import matplotlib.pyplot as plot


def stop(x, next_x, epsilon):
    if LA.norm(x) != 0:
        return LA.norm(next_x - x) / LA.norm(x) < epsilon
    return False


def get_min(result, a, b):
    if result > b:
        result = b
    if result < a:
        result = a
    return result


def f_xi(H, g, x, a, b):
    next_x = x.copy()
    n = x.shape[0]
    for i in range(n):
        h = H[i][i]
        g_i = 0
        for j in range(n):
            if j != i:
                g_i = g_i + H[i][j] * next_x[j]
        x_i = -(g_i - g[i]) / h
        next_x[i] = get_min(x_i, a[i], b[i])
    return next_x


def coordinate_desent(H, g, a, b, f, max_iter, x, epsilon):

    for iter_x in range(max_iter):
        next_x = f(H, g, x, a, b)
        if stop(x, next_x, epsilon):
            break
        x = next_x
    return x


def d():
    h = np.identity(5) * 6 - 1
    g = np.array([18, 6, -12, -6, 18])
    a = np.zeros(5)
    b = np.zeros(5) + 5
    output = coordinate_desent(h, g, a, b, f_xi, 100, a, 10 ** -3)
    print(output.reshape(5, 1))


d()
