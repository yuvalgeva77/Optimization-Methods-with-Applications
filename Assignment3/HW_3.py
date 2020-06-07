import numpy as np
from numpy.linalg import inv
from math import exp as exp
from numpy import matmul as mul
from numpy import linalg as LA
import matplotlib.pyplot as plot

file = open('Covid-19-USA.txt', 'r')
data = [int(line.replace("\n", '')) for line in file.readlines()]
days = list(range(1, 100))


def f_e(teta):
    t_1 = teta[0]
    t_2 = teta[1]
    t_3 = teta[2]

    def f(x):
        return t_1 * exp(-t_2 * ((x - t_3) ** 2))
    return np.array([f(d) for d in days])


def f_f(teta):
    t_1 = 1000000 * teta[0]
    t_2 = -0.001 * teta[1]
    t_3 = 110 * teta[2]

    def f(x):
        return t_1 * exp(t_2 * ((x - t_3) ** 2))
    return np.array([f(d) for d in days])


def df_e(teta):
    t_1 = teta[0]
    t_2 = teta[1]
    t_3 = teta[2]

    def f(x):
        computed_exp = exp(-t_2 * ((x - t_3) ** 2))
        return [computed_exp,
                (-t_1 * ((x - t_3) ** 2)) * computed_exp,
                2 * t_1 * t_2 * (x - t_3) * computed_exp]
    return np.array([f(d) for d in days])


def df_f(teta):
    t_1 = teta[0]
    t_2 = teta[1]
    t_3 = teta[2]

    def f(x):
        computed_exp = exp(-0.001 * t_2 * ((x - 110 * t_3) ** 2))
        gradient = [1000000 * computed_exp,
                    (1000000 * t_1 * -0.001 * ((x - 110 * t_3) ** 2)) * computed_exp,
                    (1000000 * t_1 * -0.001 * t_2 * 2 * -110 * (x - t_3)) * computed_exp]
        return gradient
    return np.array([f(d) for d in days])


def F(teta, f):
    return 0.5 * (LA.norm(f(teta) - data)**2)


def F_gradient(teta, f, df):
    J = df(teta)
    return mul(J.transpose, f(teta) - data)


def armijo(max_iter, x, f, gradient, direction, alpha=10**-7, b=0.5,  c=1):
    while max_iter > 0:
        h = x + alpha * direction
        objective = F(h, f)
        limit = F(x, f) + c * alpha * mul(gradient.transpose(), direction)
        if objective <= limit:
            break
        alpha *= b
        max_iter -= max_iter
    print(max_iter)
    return alpha


def stop(curr_gradient, first_gradient, epsilon):
    convergence = LA.norm(curr_gradient) / LA.norm(first_gradient)
    return convergence < epsilon


def desent(max_iter, x_k, alpha, epsilon, f, df):
    output = [x_k]

    f_theta = f(x_k)
    J = df(x_k)

    first_gradient = mul(J.transpose(), f_theta - data)

    while max_iter > 0:
        f_theta = f(x_k)
        J = df(x_k)

        curr_gradient = mul(J.transpose(), f_theta - data)
        gradient_norm = LA.norm(curr_gradient)
        normal_gradient = curr_gradient / gradient_norm

        curr_alpha = armijo(30, x_k, f, normal_gradient, -normal_gradient, alpha)
        D = curr_alpha * normal_gradient
        next_x = x_k + D

        if stop(curr_gradient, first_gradient, epsilon):
            output = output + [x_k]
            break

        x_k = next_x
        output = output + [x_k]
        max_iter = max_iter - 1
    return output

list_desent_e = desent(100, np.array([100000, 0.001, 110]), 10**-7, 10**-3, f_e, df_e)
list_desent_f = desent(100,  np.array([1, 1, 1]), 10**-7, 10**-2, f_f, df_f)


def stopi(x, next_x, epsilon):
    return LA.norm(next_x - x) / LA.norm(x) < epsilon


def newthon(max_iter, x_k, alpha, epsilon, f, df):
    output = []
    first_alpha = alpha
    while max_iter > 0:
        output = output + [x_k]

        f_theta = f(x_k)
        J = df(x_k)
        Jt_J = mul(J.transpose(), J)
        gradient = mul(J.transpose(), f_theta - data)
        d_LM = mul(inv(Jt_J), gradient)

        alpha = armijo(30, x_k, f, gradient, d_LM, first_alpha)
        print('alpha',alpha)
        next_x = np.array(x_k) - alpha * d_LM

        if stopi(x_k, next_x, epsilon):
            break

        x_k = next_x
        max_iter = max_iter - 1
    output = output + [x_k]
    return output


list_newthon_e = newthon(100,  np.array([1000000, 0.001, 110]), 0.5, 10**-2, f_e, df_e)
list_newthon_f = newthon(100,  np.array([1, 1, 1]), 0.5, 10**-2, f_f, df_f)


def show_asix(list, title, f):
    last = list[-1:][0]
    F_last = F(last, f)
    x_asix = [abs(F(theta, f) - F_last) for theta in list]
    print(x_asix)
    plot.semilogy(x_asix, label='')

    plot.title(title)
    plot.xlabel('x')
    plot.ylabel('y')
    plot.show()


show_asix(list_desent_e, 'steeps desent e', f_e)
show_asix(list_desent_f, 'steeps desent f', f_f)


show_asix(list_newthon_e, 'newthon e', f_e)
print('original', list_newthon_e)
show_asix(list_newthon_f, 'newthon f', f_f)

