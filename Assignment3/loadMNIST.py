
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


# calculate the standard deviation and mean along each coloum.
def standardize_arrays(data):
    #rows std
    s = data.std(0)
    m = np.mean(data, axis=0)
    data = data - m
    data = data / (s + 1.0)
    return (data)

# Randomly shuffle the data,lables
def shuffle_array_the_same(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def random_shuffeled_Mnist():
    (data, labels) = loadMnist()
    (data_shufled, labels_shufled) = shuffle_array_the_same(data, labels)
    data_random_shuffeled=standardize_arrays(data_shufled)
    return (data_random_shuffeled,labels_shufled)


