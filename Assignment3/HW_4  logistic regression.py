# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
import math
from numpy import matmul as mul
import loadMNIST
from random import gauss
import matplotlib.pyplot as plot

#f,J is  lambdas
def armijo(x,d, f,  gardient, b=0.5, a_0=1, c=1,max_iterations=100):
    a_k=a_0
    for i in range(0,max_iterations):
        f_k = f(x+ a_k * d)
        limit = f(x) + c * a_k * np.dot(gardient(x).transpose(),d)
        if f_k <= limit :
           return a_k
        a_k = a_k * b
        max_iterations=max_iterations-1
    return a_k


def gradient_descent(f, gardient, x_0, alpha= 0.01, eps = 10**-3,max_iterations=100):
    x_k=np.clip(x_0, -1, 1)
    output=[]
    for i in range(0,max_iterations):
        d_sd = -gardient(x_k)
        d_sd = np.reshape(d_sd, d_sd.size)
        a = armijo(x_k, d_sd, f, gardient)
        # a=1
        next_x = x_k +a*d_sd
        np.clip(next_x, -1, 1)
        if(np.linalg.norm(x_k)!=0):
            if np.linalg.norm(next_x - x_k)/np.linalg.norm(x_k) < eps:
                output = output + [next_x]
                return output
        output = output + [next_x]
        x_k = next_x
    return output


def exact_Newton(f, jacobian,gardient, x_0,  eps =10**-3,max_iterations=100):
    x_k=np.clip(x_0, -1, 1)
    output=[]
    for i in range(0,max_iterations):
        d_n=-mul(np.linalg.inv(jacobian(x_k)),gardient(x_k))
        d_n = np.reshape(d_n, d_n.size)
        a = armijo(x_k, d_n, f, gardient)
        # a=1
        next_x = x_k +a*d_n
        np.clip(next_x, -1, 1)
        if (np.linalg.norm(x_k) != 0):
            if np.linalg.norm(next_x - x_k)/np.linalg.norm(x_k) < eps:
                output = output + [next_x]
                return output
        output = output + [next_x]
        x_k = next_x
    return output
# -----------------------------------------------------------

def sigmoid(x_i,w):
    return (1 / (1 + math.exp(mul(x_i, w))))

#preforms logistic_func on each row on array
#return [logistic_func(x_1, w)|...|logistic_func(x_m, w)].traspose()  Rmx1
def sigmoid_Marix(X):
    def sigmoid_Marix_w(w):
        dim1 = (1 / (1 + np.exp(np.dot(-X, w))))
        return np.reshape(dim1, (dim1.size, 1))
    return lambda w:sigmoid_Marix_w(w)

def sigmoid_Minus1(X):
    return lambda w: 1-sigmoid_Marix(X)(w)


def logistic_regression_objective(X, Y):
    n, m = X.shape  # columns
    c1=Y
    c2=Y
    oppsite = lambda y: 1-y
    vfunc = np.vectorize(oppsite)
    c2=vfunc(c2)
    return lambda w: (-1/m)*(mul(c1.transpose(), np.log(sigmoid_Marix(X.transpose())( w))) + mul(c2.transpose(),np.log( sigmoid_Minus1(X.transpose())(w))))

def gradient(X, Y):
    n, m = X.shape  # columns
    # returns vector size n: 784
    def gradiesnt_w(w):
        return (1 / m) * (mul(X, sigmoid_Marix(X.transpose())( w) - Y))
    return lambda w:gradiesnt_w(w)

def hessian(X):
    n, m = X.shape  # columns
    # matrix sized n*n:784*784
    def hessian_w(w):
        # np.multiply -Multiply arguments element-wise.
        n, m = X.shape  # columns
        # D=np.diag(logistic_func_Marix(X.transpose())( w).mul(logistic_func_Marix_Minus1(X.transpose())( w)))
        # X.mul(D.mul(X.transpose())) / m
        D = np.diagflat(np.multiply(sigmoid_Marix(X.transpose())( w), (sigmoid_Minus1(X.transpose())(w))))
        D_Xt = mul(D, X.transpose())
        x_D_Xt = mul(X, D_Xt) + np.identity(n)
        return (1 / m) * x_D_Xt
    return lambda w:hessian_w(w)

# The gradient test ,The Jacobian test:
def gradient_jacobian_test(w_0,f,gradient,hessian,eps=1,factor=0.5,limit=6):
    def make_rand_vector(dims):
        vec = [gauss(0, 1) for i in range(dims)]
        mag = sum(x ** 2 for x in vec) ** .5
        return [x / mag for x in vec]
    # -------------------
    # (X, labels) = loadMNIST.random_shuffeled_Mnist()
    n = w_0.shape[0]
    d = np.array(make_rand_vector(n))
    d = np.reshape(d, (d.size, 1))
    res = True
    # w_0 = np.zeros((n, 1))
    eps_i = (factor ** 0) * eps
    w_i = w_0 + eps_i * d

    res_1_prev = abs(f( w_i) - f( w_0))
    res_2_prev = abs(f( w_i) - f( w_0) - eps * mul(d.transpose(), (gradient( w_0))))
    res_3_prev = np.linalg.norm(gradient( w_i) - gradient( w_0))
    res_4_prev = np.linalg.norm(((gradient( w_i) - gradient( w_0)).transpose() - (eps * mul(
        d.transpose(), (hessian(w_0))))))

    for i in range(1, limit):
        eps_i = (factor ** i) * eps
        w_i = w_0 + eps_i * d
        #-----gardient test------
        res1 = abs(f(w_i) - f(w_0))
        res2 = abs((f(w_i) - f(w_0) - eps_i * mul(d.transpose(), (gradient( w_0)))))
        if (abs((res_1_prev / res1) - 1 / factor) > 1 or abs((res_2_prev / res2) - 1 / (factor ** 2)) > 1):
            print("failed The gradient test with eps:{0}, iteration {1}".format(eps, i))
            res = False
        #-------jacobian test------
        res3 = np.linalg.norm(gradient(w_i) - gradient(w_0))
        res4 = np.linalg.norm(((gradient(w_i) - gradient(w_0)).transpose() - eps_i * mul(
            d.transpose(), (hessian( w_0)))))
        # todo abs((res_4_prev / res4) - 1 / (factor ** 2)) is about 3-4 not 1 check whay 2*(factor ** 2)
        if (abs((res_3_prev / res3) - 1 / factor) > 1 or abs((res_4_prev / res4) - 1 / (factor ** 2)) > 1):
            print("failed The Jacobian test with eps:{0}, iteration {1}, res is ({2},{3})\n".format(eps, i, abs(
                (res_3_prev / res3) - 1 / factor), abs((res_4_prev / res4) - 1 / (factor ** 2))))
            res=False

        res_1_prev = res1
        res_2_prev = res2
        res_3_prev = res3
        res_4_prev = res4

    if(res==True):
        print("seccesed The gradient and the jacobian test  with eps:{0}, iteration {1}".format(eps, i))
    return res


def task_4a():
    # return X=[x1|...|Xm]   R:nxm
    # labels=[y1|...|ym].traranspose   R:mx1
    (X, labels)=loadMNIST.random_shuffeled_Mnist()
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    return (f_objective, grad, hess)

def task_4b():
    (X, labels) = loadMNIST.random_shuffeled_Mnist()
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    n, m = X.shape
    w_0 = np.zeros((n, 1))
    return (gradient_jacobian_test(w_0 ,f_objective, grad,hess))

def task_4c():
    (X, labels) = loadMNIST.random_shuffeled_Mnist()
    n, m = X.shape
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    w_0 = np.zeros(n)

    w_i_res_gradient = gradient_descent(f_objective, grad, w_0)
    f_res_gradient=[]
    w_res_gradient=w_i_res_gradient[len(w_i_res_gradient)-1]
    for w_i in w_i_res_gradient:
        f_res_gradient=f_res_gradient+[abs(f_objective(w_i)[0][0]-f_objective(w_res_gradient)[0][0])]

    w_i_res_Newton = exact_Newton(f_objective, hess,grad, w_0)
    f_res_Newton = []
    w_res_Newton = w_i_res_Newton[len(w_i_res_Newton) - 1]
    for w_i in w_i_res_Newton:
        f_res_Newton = f_res_Newton + [abs(f_objective(w_i)[0][0] - f_objective(w_res_Newton)[0][0])]

    plot.semilogy(f_res_gradient,label="train Gradient Descent")
    plot.semilogy(f_res_Newton, label="train Exact Newton")
    plot.title("task_4c")
    plot.xlabel('iterations')
    plot.ylabel('|f(w_k) - f(w_*)|')
    plot.legend()
    plot.show()
    print("res")
    # exact_Newton(f_objective, hess,J, w_0):

task_4c()