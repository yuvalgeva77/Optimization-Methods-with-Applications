from numpy import matmul as mul
from math import log2
from numpy.linalg import inv
import numpy as np
from functools import reduce
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt

def f_i(x):
    if x >= 0.0 and x < 1.0:
        return 0.5 * x
    if x >= 1.0 and x < 2.0:
        return 0.8 - 0.2 * log2(x)
    if x >= 2.0 and x < 3.0:
       return 0.7 - 0.2 * x
    if x >= 3.0 and x < 4.0:
       return 0.3
    if x >= 4 and x <= 5:
       return 0.5 - 0.1 * x

def f(x):
    n = x.shape[0]
    F_x = np.zeros((n, 1))
    for i in range(n - 1):
        F_x[i][0] = f_i(x[i][0])
    return(F_x)


def y_func(x):
    n=x.shape[0]
    I = np.identity(n)
    s = np.random.multivariate_normal(np.zeros(n), math.pow(0.1,2) * I)
    F_x= f(x)
    return (F_x+np.matrix(s).transpose())

def plot_graph():
    x_1 = np.arange (0.0,5.1,0.01)
    x=x_1.reshape((510, 1))
    plt.plot(x_1,f(x),'b')
    plt.plot(x_1,y_func(x),'r')
    plt.show()

def G_matrix(n):
    A = np.zeros((n -1 , n))
    for i in range(0,n-1):
        A[i][i] = -1
        A[i][i+1] = 1
    return A


# # weighted least squares
# def least_squares(A, b,l):
#     At = A.transpose()
#     At_A = reduce(mul, [At, A])
#     x = reduce(mul, [inv(At_A), At,  b])
#     return (x)
#     r = mul(A, x) - b
#     plt.plot( x.reshape(x.shape[0], 1), f(x), 'b')
#     plt.plot(x.reshape(x.shape[0],1), y_func(x), 'r')
#     plt.show()
#
# def task_3a(l=80):
#     x_1 = np.arange(0.0, 5.1, 0.1)
#     x = x_1.reshape((51, 1))
#     n = x.shape[0]
#     I = np.identity(n)
#     G = G_matrix(n)
#     A = np.vstack((I, G * math.sqrt(80)))
#     A = np.vstack((A,np.zeros(n)))
#     Y = y_func(x)
#     y_1 = np.vstack((Y, np.zeros((n, 1))))
#     print(least_squares(A,y_1,l))

def least_squares(A, b):
    At = A.transpose()
    At_A = reduce(mul, [At, A])
    x = reduce(mul, [inv(At_A), At,  b])
    return (x)
    r = mul(A, x) - b
    # plt.plot( x.reshape(x.shape[0], 1), f(x), 'b')
    # plt.plot(x.reshape(x.shape[0],1), y_func(x), 'r')
    # plt.show()

def regularizedLS(A,l,C,Y,n):
    A = np.vstack((A,C*math.sqrt(l)))
    y = np.vstack((Y, np.zeros((n, 1))))
    return (least_squares(A,y))

def weightedLS(A,Y,w):
        At = A.transpose()
        w = np.diag(w)
        At_w_A = reduce(mul, [At, w, A])
        x_weighted = reduce(mul, [inv(At_w_A), At, w, Y])
        r = mul(A, x_weighted) - Y
        return(x_weighted)


def task_3a(l=80/2):
    x_1 = np.arange(0.0, 5.1, 0.1)
    x = x_1.reshape((51, 1))
    n = x.shape[0]
    I = np.identity(n)
    G = G_matrix(n)
    C = np.vstack((G,np.zeros(n)))
    Y = y_func(x)
    x_ans=regularizedLS(I,l,C,Y,n)
    print(x_ans)
    # plt.plot(x.reshape(x.shape[0], 1), f(x), 'b')
    # plt.plot(x_ans.reshape(x.shape[0],1), y_func(x_ans), 'r')
    plt.title("task 3a")
    plt.plot(x, f(x), 'b')
    plt.plot(x, x_ans, 'r')
    plt.show()

def task_3b(l=1,eps=0.001):
    x_1 = np.arange(0.0, 5.1, 0.1)
    x_1 = x_1.reshape((51, 1))
    x_i=x_1
    n = x_i.shape[0]
    w= np.ones(n)
    A = np.identity(n)
    G = G_matrix(n)
    for i in range(0,10):
        print("w:  {0}\n".format(w))
        Y = y_func(x_i)
        print("Y:  {0}\n".format(Y))
        x_i=weightedLS(A,Y,w)
        print("x_i:  {0}\n".format(x_i))
        G_x_i=mul(G,x_i)
        print("G_x_i:  {0}\n".format(G_x_i))
        for j in range(0,n-1):
          w[j]=1/(math.fabs(G_x_i[j])+eps)
    # plt.plot(x_1, y_func(x), 'r')
    # plt.plot(x_1.reshape(x_1.shape[0],1, f(x_1), 'b'))
    # plt.plot(x_i.reshape(x_i.shape[0],1, f(x_i), 'r'))
    # plt.title("task 3b")
    # plt.show()
    # plt.plot(x_i, x_i, 'b')
    # plt.plot(x_1, y_func(x_1), 'r')
    plt.plot(x_1, f(x_1), 'b')
    plt.plot(x_1, x_i, 'r')
    plt.title("task 3b")
    plt.show()

task_3a()
task_3b()

