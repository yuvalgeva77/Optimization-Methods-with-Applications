import numpy as np
from numpy.linalg import inv
from math import exp as exp
from numpy import matmul as mul
from numpy import linalg as LA
import matplotlib.pyplot as plot


def f_d(p, mu):
    def func(x):
        x1 = x[0]
        x2 = x[1]

        eq = 3 * x1 + x2 - 6
        ieq_1 = x1**2 + x2**2 - 5
        ieq_2 = -x1

        v = (x1 + x2)**2 - 10*(x1 + x2) + mu*(p(eq) + p(max(0, ieq_1)) + p(max(0, ieq_2)))

        return v

    def derivative(x):
        x1 = x[0]
        x2 = x[1]

        ieq_1_x1 = 0
        ieq_1_x2 = 0
        ieq_2_x1 = 0

        # x1:
        eq_x1 = 6*(3*x1 + x2 - 6)
        if x1**2 + x2**2 - 5 > 0:
            ieq_1_x1 = 4 * x1 * (x1**2 + x2**2 - 5)
            ieq_1_x2 = 4 * x2 * (x1**2 + x2**2 - 5)
        if x1 < 0:
            ieq_2_x1 = 2 * x1
        d_x1 = 2 * (x1 + x2) - 10 + mu * (eq_x1 + ieq_1_x1 + ieq_2_x1)

        # x2:
        eq_x2 = 2 * (3 * x1 + x2 - 6)
        d_x2 = 2 * (x1 + x2) - 10 + mu * (eq_x2 + ieq_1_x2)
        return np.array([d_x1, d_x2])
    return func, derivative


def armijo(max_iter, x, f, gradient, direction, alpha=1, b=0.5,  c=10**-5):
    while max_iter > 0:
        objective = f(x + alpha * direction)
        limit = f(x) + c * alpha * mul(gradient.transpose(), direction)
        if objective <= limit:
            break
        alpha = alpha * b
        max_iter = max_iter - 1
    return alpha


def stop(x, next_x, epsilon):
    if LA.norm(x) != 0:
        return LA.norm(next_x - x) / LA.norm(x) < epsilon
    return False


def desent(max_iter, x_k, alpha, epsilon, f, df):
    start_alpha = alpha
    while max_iter > 0:
        gradient = df(x_k)
        gradient_norm = LA.norm(gradient)
        normal_gradient = gradient / gradient_norm
        direction = - normal_gradient

        alpha = armijo(10, x_k, f, normal_gradient, direction, start_alpha)
        next_x = x_k + alpha * direction

        if stop(x_k, next_x, epsilon):
            return next_x

        x_k = next_x
        max_iter = max_iter - 1

    return x_k


def d():
    mus = [10**i for i in range(-2, 3)]
    # mus = [100]
    for m in mus:
        f, df = f_d(lambda y: y**2, m)
        result = desent(500, np.array([0, 0]), 1, 2**-10, f, df)
        print(m, ':', result)

d()

