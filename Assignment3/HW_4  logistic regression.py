# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
import math
from numpy import matmul as mul
from mlxtend.data import mnist_data
import loadMNIST
from numpy.linalg import inv
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
    while abs(f(x_k)) > eps:
        a=armijo(x_k, f, J)
        d_n=mul(inv(H(x_k)),J(x_k))
        x_k = x_k +a*d_n
    return x_k
#-----------------------------------------------------------

def sigmoid(x_i,w):
    return  (1 / 1 + math.exp(mul(-x_i, w)))

#preforms logistic_func on each row on array
#return [logistic_func(x_1, w)|...|logistic_func(x_m, w)].traspose()  Rmx1
def sigmoid_Marix(X): #todo np.exp
    return lambda w: np.vstack(([np.array(sigmoid(X[i],w)) for i in range(0, X.shape[0])]))

def sigmoid_Minus1(X):#todo:np.log
    return lambda w:np.vstack(([np.array(1-sigmoid(X[i],w)) for i in range(0, X.shape[0])]))


def logistic_regression_objective(X, Y):
    n, m = X.shape  # columns
    c1=Y
    c2=Y
    oppsite = lambda y: 1-y
    vfunc = np.vectorize(oppsite)
    c2=vfunc(c2)
    return lambda w: (-1/m)*mul(c1.transpose(),np.apply_along_axis(math.log, 1, (sigmoid_Marix(X.transpose())( w)))) +mul(c2.transpose() ,np.apply_along_axis(math.log, 1, (sigmoid_Minus1(X.transpose())( w))))#TODO np.log
    #todo check c1.transpose() or c1


def gradient(X, Y):
    n, m = X.shape  # columns
    return lambda w: X.mul(sigmoid_Marix(X.transpose())( w) - Y) / m

def hessian(X):
    n, m = X.shape  # columns
    #D=np.diag(logistic_func_Marix(X.transpose())( w).mul(logistic_func_Marix_Minus1(X.transpose())( w)))
    # X.mul(D.mul(X.transpose())) / m
    return lambda w:X.mul(np.diag(sigmoid_Marix(X.transpose())( w).mul(sigmoid_Minus1(X.transpose())( w))).mul(X.transpose())) / m


def task_4a():
    # return X=[x1|...|Xm]   R:nxm
    # labels=[y1|...|ym].traranspose   R:mx1
    (X, labels)=loadMNIST.random_shuffeled_Mnist()
    f_objective = logistic_regression_objective(X, labels)
    grad = gradient(X, labels)
    hess = hessian(X)
    return (f_objective, grad, hess,X)



# The gradient test ,The Jacobian test:
def gradient_jacobian_test(X,f,grad,eps=1,factor=0.5,limit=20):
    n, m = X.shape

    def make_rand_vector(dims):
        vec = [gauss(0, 1) for i in range(dims)]
        mag = sum(x ** 2 for x in vec) ** .5
        return [x / mag for x in vec]

    d = make_rand_vector(n, )
    print("A")
    res_1_prev = math.abs(f(X + eps * d) - f(X))
    print("A")
    res_2_prev = math.abs(f(X + eps * d) - f(X) - eps * d.transpose().mul(grad(X)))
    res_2_prev= math.abs(f(X + eps * d) - f(X) - eps * d.transpose().mul(grad(X)))
    jackMV = grad(eps * d)
    res_3_prev= math.abs(f(X + eps * d) - f(X) - jackMV)

    for i in range(1,limit):
        eps_i=(factor**i)*eps
        res1=math.abs(f(X+eps_i*d)-f(X))
        res2 = math.abs(f(X + eps_i * d) - f(X)-eps_i*d.transpose().mul(grad(X)))
        if(res_1_prev/factor!=res1 or res_2_prev/(factor**2)!=res2):
            print("failed The gradient test with eps:{0}, iteration {1}".format(eps,i))
            return  False
        jackMV = grad(eps_i * d)
        res_3 = math.abs(f(X + eps_i * d) - f(X) - jackMV)
        if res_1_prev / 2 != res1 or res_3_prev / (factor ** 2) != res2:
            print("failed The Jacobian test with eps:{0}, iteration {1}".format(eps, i))
            return False
        res_1_prev=res1
        res_2_prev = res2
        rres_3_prev=res_3
    print("seccesed The gradient and the jacobian test  with eps:{0}, iteration {1}".format(eps, i))
    return True


(f, grad, hess,X)= task_4a()
n, m = X.shape
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]

d = make_rand_vector(m)
eps=1
factor=0.5
limit=20
print("A")
w_0=np.zeros(m)
x=w_0
res_1_prev = math.abs(f(x + eps * d) - f(X))
print("A")

res_2_prev = math.abs(f(x + eps * d) - f(x) - eps * d.transpose().mul(grad(X)))
res_2_prev = math.abs(f(X + eps * d) - f(x) - eps * d.transpose().mul(grad(X)))
jackMV = grad(eps * d)
res_3_prev = math.abs(f(x + eps * d) - f(x) - jackMV)

for i in range(1, limit):
    eps_i = (factor ** i) * eps
    res1 = math.abs(f(x + eps_i * d) - f(X))
    res2 = math.abs(f(x + eps_i * d) - f(X) - eps_i * d.transpose().mul(grad(x)))
    if (res_1_prev / factor != res1 or res_2_prev / (factor ** 2) != res2):
        print("failed The gradient test with eps:{0}, iteration {1}".format(eps, i))

    jackMV = grad(eps_i * d)
    res_3 = math.abs(f(x + eps_i * d) - f(x) - jackMV)
    if res_1_prev / 2 != res1 or res_3_prev / (factor ** 2) != res2:
        print("failed The Jacobian test with eps:{0}, iteration {1}".format(eps, i))

    res_1_prev = res1
    res_2_prev = res2
    rres_3_prev = res_3
print("seccesed The gradient and the jacobian test  with eps:{0}, iteration {1}".format(eps, i))
