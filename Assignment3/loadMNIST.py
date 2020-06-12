from mlxtend.data import mnist_data, loadlocal_mnist
import numpy as np
from sklearn.model_selection import train_test_split



# return X=[x1|...|Xm].trnspode()   R:mxn
# Y=[y1|...|ym].traranspose   R:mx1
def loadMnist(X,Y,d1=0,d2=1,nSamples = 30000):  # return [X,Y]
    # Dimensions x: 5000 x 784
    # Dimensions: y: 5000
    # X, Y = mnist_data()
    X_t = X.transpose()
    X = X[0:nSamples]
    Y = Y[0:nSamples]
    # Digits to classify
    # d1 = 0
    # d2 = 1
    # Take only d1 and d2 digits
    filtered_data = np.array([X[i] for i in range(0, X.shape[0]) if (Y[i] == d1 or Y[i] == d2)])  # (1000, 1)
    filtered_labels = np.array([[Y[i]] for i in range(0, X.shape[0]) if (Y[i] == d1 or Y[i] == d2)])  # (1000, 784)
    return (filtered_data.astype('float') / 255, filtered_labels)


# calculate the standard deviation and mean along each coloum.
def standardize_arrays(data):
    # rows std
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


# return (X,Y)tset,(X,Y)train
# X=[x1|...|Xm].transpose()   R:nxm, n=num images, m=28^2 num of pixels in an image. x[0][4] pixel 4 in image 0
# Y=[y1|...|ym].transform()   R:mx1  Y[0] label of image 0 (X[0])
def random_shuffeled_Mnist(d1=0,d2=1):
    # (data, labels) = loadMnist()
    X_train, y_train = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')
    X_train, y_train=loadMnist(X_train, y_train,d1,d2)

    X_test, y_test = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')
    X_test, y_test = loadMnist(X_test, y_test,d1,d2)

    (train_data_shufled, train_labels_shufled) = shuffle_array_the_same(X_train, y_train)
    train_data_random_shuffeled = standardize_arrays(train_data_shufled)

    (test_data_shufled, test_labels_shufled) = shuffle_array_the_same(X_test, y_test)
    test_data_random_shuffeled = standardize_arrays(test_data_shufled)

    return train_data_random_shuffeled.transpose(), train_labels_shufled,test_data_random_shuffeled.transpose(), test_labels_shufled

