# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
import math
from numpy import matmul as mul
from עבודות.עבודות.Assignment3 import loadMNIST
from random import gauss


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
    res=[]
    while abs(f(x_k)) > eps:
        a=armijo(x_k, f, J)
        d_n=mul(np.linalg.inv(H(x_k)),J(x_k))
        x_k = x_k +a*d_n
        res=res.append(x_k)
    return x_k
#-----------------------------------------------------------

def sigmoid(x_i,w):
    return   (1 /( 1 + math.exp(mul(x_i, w))))

#preforms logistic_func on each row on array
#return [logistic_func(x_1, w)|...|logistic_func(x_m, w)].traspose()  Rmx1
def sigmoid_Marix(X):
    return lambda w: np.reshape(X,(1, X.size))1 / (1 + np.exp(np.dot(-X, w))),

def sigmoid_Minus1(X):
    return lambda w: 1-sigmoid_Marix(X,w)


def logistic_regression_objective(X, Y):
    n, m = X.shape  # columns
    c1=Y
    c2=Y
    oppsite = lambda y: 1-y
    vfunc = np.vectorize(oppsite)
    c2=vfunc(c2)
    return lambda w: (-1/m)*(mul(c1.transpose(), np.log(sigmoid_Marix(X.transpose(), w))) + mul(c2.transpose(),np.log( sigmoid_Minus1(X.transpose(),w))))

#todo check: gradient(X.transpose(), labels, w_i) size m*n = 784*1000?? #should be vector but is matrix??
def gradient(X, Y):
    n, m = X.shape  # columns
    return lambda w:(mul(X,sigmoid_Marix(X.transpose(),w) - Y)) / m

#todo check (hessian(X.transpose(), w_0)) size m*m = 784*784??
def hessian(X):
    n, m = X.shape  # columns
    def hessian(w):
        # np.multiply -Multiply arguments element-wise.
        D = np.diag(np.multiply(sigmoid_Marix(X.transpose(), w), (sigmoid_Minus1(X.transpose(), w))))
        D_Xt = mul(D, X.transpose())
        x_D_Xt = mul(X, D_Xt)
        return x_D_Xt

    return lambda w:hessian(w)

# The gradient test ,The Jacobian test:
def gradient_jacobian_test(X,labels,f,gradient,hessian,eps=1,factor=0.5,limit=6):
    # (X, labels) = loadMNIST.random_shuffeled_Mnist()
    n, m = X.shape
    def make_rand_vector(dims):
        vec = [gauss(0, 1) for i in range(dims)]
        mag = sum(x ** 2 for x in vec) ** .5
        return [x / mag for x in vec]
    d = np.array(make_rand_vector(m))
    w_0 = np.zeros(m)
    eps_i = (factor ** 0) * eps
    w_i = w_0 + eps_i * d

    res_1_prev = abs(f(X.transpose(), labels, w_i) - f(X.transpose(), labels, w_0))

    res_2_prev = abs(f(X.transpose(), labels, w_i) - f(X.transpose(), labels, w_0) - eps * mul(d.transpose(), (
        gradient(X.transpose(), labels, w_0))))  #todo first part is vector gradient matrix. todo norm not abs?
    res_3_prev = np.linalg.norm(gradient(X.transpose(), labels, w_i) - gradient(X.transpose(), labels, w_0))

    res_4_prev = np.linalg.norm(
        (gradient(X.transpose(), labels, w_i) - gradient(X.transpose(), labels, w_0) - eps * mul(
            d.transpose(), (hessian(X.transpose(), w_0)))))  #todo first part is marix 784*1000, hessian matrix 784*784. todo norm not abs?

    for i in range(1, limit):
        eps_i = (factor ** i) * eps
        w_i = w_0 + eps_i * d
        res1 = abs(f(X.transpose(), labels, w_i) - f(X.transpose(), labels, w_0))
        res2 = abs((f(X.transpose(), labels, w_i) - f(X.transpose(), labels, w_0) - eps_i * mul(d.transpose(), (
            gradient(X.transpose(), labels, w_0)))))  # todo check
        if (abs((res_1_prev / res1) - factor) > 1 or abs((res_2_prev / res2) - (factor ** 2)) > 1):
            print("failed The gradient test with eps:{0}, iteration {1}".format(eps, i))
            return False


        res3 = np.linalg.norm(gradient(X.transpose(), labels, w_i) - gradient(X.transpose(), labels, w_0))
        res4 = np.linalg.norm((gradient(X.transpose(), labels, w_i) - gradient(X.transpose(), labels,
                                                                               w_0) - eps_i * mul(d.transpose(), (
            hessian(X.transpose(), w_0)))))  # todo check
        if (abs((res_3_prev / res3) - factor) > 1 or abs((res_4_prev / res4) - (factor ** 2)) > 1):
            print("failed The gradient test with eps:{0}, iteration {1}".format(eps, i))
            return False

        res_1_prev = res1
        res_2_prev = res2
        res_3_prev = res3
        res_4_prev = res4
    print("seccesed The gradient and the jacobian test  with eps:{0}, iteration {1}".format(eps, i))
    return True

def task_4a():
    # return X=[x1|...|Xm]   R:nxm
    # labels=[y1|...|ym].traranspose   R:mx1
    (X, labels)=loadMNIST.random_shuffeled_Mnist()
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    return (f_objective, grad, hess)

def task_4b():
    # return X=[x1|...|Xm]   R:nxm
    # labels=[y1|...|ym].traranspose   R:mx1
    (X, labels) = loadMNIST.random_shuffeled_Mnist()
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    return (gradient_jacobian_test(X,labels ,f_objective, grad,hess))

def task_4c():
    (X, labels) = loadMNIST.random_shuffeled_Mnist()
    n, m = X.shape
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    w_0 = np.zeros(m)
    gradient_descent(w_0, hess)
    # exact_Newton(f_objective, hess,J, w_0):

