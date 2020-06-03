
from mlxtend.data import mnist_data
import numpy as np

def loadMnist(): #return [train, test]
    # Dimensions x: 5000 x 784
    # Dimensions: y: 5000
    X, Y = mnist_data()
    nSamples = 30000; # Number of samples from the dataset
    X_t=X.transpose()
    X=X[0:nSamples]
    Y=Y[0:nSamples]
    #Digits to classify
    d1 = 0;
    d2 = 1;
    # Take only d1 and d2 digits
    filtered_data=np.array([X[i] for i in range(0,X.shape[0]) if (Y[i]==d1 or Y[i]==d2)]) #(1000, 1)
    filtered_labels=np.array([[Y[i]] for i in range(0,X.shape[0]) if (Y[i]==d1 or Y[i]==d2)]) #(1000, 784)
    return(filtered_data.astype('float')/255,filtered_labels)


