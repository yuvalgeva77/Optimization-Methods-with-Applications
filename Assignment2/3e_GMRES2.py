import numpy as np
import matplotlib.pyplot as plt

#---------3.e ----------------------------------

def GMRES2(A, b, x_0, epsilon, max_iterations):
   res_list=[]
   r_k1=b-np.dot(A, x_0)
   r_k2=b              #x_-1 = 0
   R_k=np.concatenate((r_k1, r_k2), axis=1) #R Matrix
   x_k1=x_0
   for i in range(max_iterations):
       d=np.dot(A, R_k)
       a=np.dot(np.dot(np.transpose(d),r_k1),np.linalg.inv(np.dot(np.transpose(d),d)))   # a Vector
       print(a)
       x_k=x_k1+np.dot(R_k,a)
       r_k=b- np.dot(A,x_k)
       print(np.dot(R_k,a))
       residul_norm=np.linalg.norm(np.dot(A, x_k) - b)
       res_list.append(residul_norm)
       if (residul_norm < epsilon):
            plotGraph(res_list,range(i))
            return x_k
       r_k2=r_k1
       r_k1=r_k
       R_k=np.concatenate((r_k1, r_k2), axis=1)
       print(r_k1)
       x_k1 = x_k
   plotGraph(res_list,range(max_iterations))
   return x_k

def plotGraph(Xlist,Ylist):
    plt.semilogx(Xlist,Ylist)
    plt.ylabel("Iterations")
    plt.xlabel("Residul norm")
    plt.show()




