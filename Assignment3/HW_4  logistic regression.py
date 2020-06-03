# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
import math
from numpy import matmul as mul
from mlxtend.data import mnist_data
import loadMNIST

def logistic_func(x_i, w):
    return 1/1+math.exp(mul(-x_i,w))



def logistic_func_Marix(X, w):
    n,m=X.shape
    res= np.zeros((n, 1))
    for i in range(0,m):
       x_i = X[:, i]  # ith First Column
       res[i]= 1/1+math.exp(mul(-x_i,w))
    return res

def logistic_regression_objective(X,y,w):
     n,m=X.shape #columns
     res=0
     for i in range(0,m):
         x_i=X[:,i] #ith First Column
         res=+(y[i]*math.log2(logistic_func(x_i.transpose(), w))+1-y[i])*math.log2(1-logistic_func(x_i, w))
     return (res/m)

def gradient(X,Y,w):
    n, m = X.shape  # columns
    return( (X.mul(logistic_func_Marix(X.transpose(),w)-y[0])/m))

def hessian(X,w):
    n,m=X.shape   # columns
    d1=logistic_func_Marix(X.transpose(),w)
    d2= np.ones((m, 1))-d1
    for i in range(0,m):
        res=d1[i]*d2[i]
    D=p.diag(res)
    return( (X.mul(D.mul(X.transpose())))/m)


def task_4a(X,labels):
    w=np.array([1/50,1/50,1/50])
    J=logistic_regression_objective(X,labels,w)
    print("logistic_regression_objective  {0}\n".format(J))
    G=gradient(X,labels,w)
    print("gradient is:  {0}\n".format(G))
    H=hessian(X,w)
    print("hessian is:  {0}\n".format(H))

(X,labels)=loadMNIST.loadMnist()

# task_4a()
