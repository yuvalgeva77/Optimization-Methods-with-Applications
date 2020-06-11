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
    t_1 = teta[0]
    t_2 = teta[1]
    t_3 = teta[2]

    def f(x):
        return 1000000 * t_1 * exp(-0.001 * t_2 * ((x - 110 * t_3) ** 2))
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
                    (1000000 * t_1 * -0.001 * t_2 * 2 * -110 * (x - 110 * t_3)) * computed_exp]
        return gradient
    return np.array([f(d) for d in days])


def F(teta, f):
    return 0.5 * (LA.norm(f(teta) - data)**2)


def F_gradient(teta, f, df):
    J = df(teta)
    return mul(J.transpose, f(teta) - data)


def armijo(max_iter, x, f, gradient, direction, alpha, b=0.5,  c=10**-5):
    while max_iter > 0:
        h = x + alpha * direction
        objective = F(h, f)
        limit = F(x, f) + c * alpha * mul(gradient.transpose(), direction)
        if objective <= limit:
            break
        alpha = alpha * b
        max_iter = max_iter - 1
    return alpha


def stop(x, next_x, epsilon):
    return LA.norm(next_x - x) / LA.norm(x) < epsilon


def desent(max_iter, x_k, alpha, epsilon, f, df):
    output = [x_k]

    while max_iter > 0:
        f_theta = f(x_k)
        J = df(x_k)

        curr_gradient = mul(J.transpose(), f_theta - data)
        gradient_norm = LA.norm(curr_gradient)
        normal_gradient = curr_gradient / gradient_norm

        curr_alpha = armijo(30, x_k, f, normal_gradient, -normal_gradient, alpha)
        D = curr_alpha * normal_gradient
        next_x = x_k + D

        # if stop(x_k, next_x, epsilon):
        #     output = output + [x_k]
        #     break

        x_k = next_x
        output = output + [x_k]
        max_iter = max_iter - 1
    return output


def newton(max_iter, x_k, alpha, epsilon, f, df):
    output = []
    first_alpha = alpha
    while max_iter > 0:
        output = output + [x_k]

        f_theta = f(x_k)
        J = df(x_k)
        Jt_J = mul(J.transpose(), J)
        gradient = mul(J.transpose(), f_theta - data)
        d_LM = mul(inv(Jt_J), -gradient)

        alpha = armijo(20, x_k, f, gradient, d_LM, first_alpha)
        next_x = np.array(x_k) + alpha * d_LM

        # if stop(x_k, next_x, epsilon):
        #     break

        x_k = next_x
        max_iter = max_iter - 1
    output = output + [x_k]
    return output


def add_graph_line(ans, name_label, f):
    last = ans[-1:][0]
    F_last = F(last, f)
    x_asix = [abs(F(theta, f) - F_last) for theta in ans]
    plot.semilogy(x_asix, label=name_label)


def show_graph(descent_list, newton_list, title, f):
    add_graph_line(descent_list, 'steeps descent', f)
    add_graph_line(newton_list, 'newton', f)

    plot.title(title)
    plot.xlabel('iterations')
    plot.ylabel('|f(θ) - f(θ)*|')
    plot.legend()
    plot.show()


def question_e():
    list_desent_e = desent(100, np.array([1000000, 0.001, 110]), 10**-7, 10**-3, f_e, df_e)
    list_newton_e = newton(100, np.array([1000000, 0.001, 110]), 1, 10 ** -5, f_e, df_e)
    show_graph(list_desent_e, list_newton_e, 'Question e', f_e)


def question_f():
    list_desent_f = desent(100,  np.array([1, 1, 1]), 10**-7, 10**-2, f_f, df_f)
    list_newton_f = newton(100,  np.array([1, 1, 1]), 1, 10**-5, f_f, df_f)
    show_graph(list_desent_f, list_newton_f, 'Question f', f_f)


question_e()
question_f()
