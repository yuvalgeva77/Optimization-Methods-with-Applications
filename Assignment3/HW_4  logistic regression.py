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

def logistic_func(x_i,w):
    return  (1 / 1 + math.exp(mul(-x_i, w)))

#preforms logistic_func on each row on array
#return [logistic_func(x_1, w)|...|logistic_func(x_m, w)].traspose()  Rmx1
def logistic_func_Marix(X):
    return lambda w: np.vstack(([np.array(logistic_func(X[i],w)) for i in range(0, X.shape[0])]))
def logistic_func_Marix_Minus1(X):
    return lambda w:np.vstack(([np.array(1-logistic_func(X[i],w)) for i in range(0, X.shape[0])]))




def logistic_regression_objective(X, Y):
    n, m = X.shape  # columns
    c1=Y
    c2=Y
    oppsite = lambda y: 1-y
    vfunc = np.vectorize(oppsite)
    c2=vfunc(c2)
    return lambda w: (-1/m)*(c1.transpose(),np.apply_along_axis(math.log2, 1, (logistic_func_Marix(X.transpose())( w)))) +mul(c2.transpose() ,np.apply_along_axis(math.log2, 1, (logistic_func_Marix_Minus1(X.transpose())( w))))



def gradient(X, Y):
    n, m = X.shape  # columns
    return lambda w:X.mul(logistic_func_Marix(X.transpose())( w) - Y) / m

#TODO fix
def hessian(X):
    n, m = X.shape  # columns
    return lambda w:X.mul((np.diag(mul(logistic_func_Marix(X.transpose())( w).transpose(),(logistic_func_Marix_Minus1(X.transpose())( w))))).mul(X.transpose())) / m


def task_4a(X, labels):
    f = logistic_regression_objective(X, labels)
    j = gradient(X, labels)
    H = hessian(X)
    return (f, j, H)

# return X=[x1|...|Xm]   R:nxm
# labels=[y1|...|ym].traranspose   R:mx1
(X, labels) = loadMNIST.random_shuffeled_Mnist()
(f, j, H)=task_4a(X, labels)
n, m = X.shape
# w=np.ones(n)
w=(np.ones(n)/n)
w_0=gradient_descent(w,H)
