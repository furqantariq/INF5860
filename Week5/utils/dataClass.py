from six.moves import cPickle as pickle
import numpy as np
import os
import platform


class dataClass:
    def __init__(self, cifar10_dir):
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

        # Preprocessing: reshape the image data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test  -= mean_image

        # add bias dimension and transform into columns
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_test  = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        self.numbOfTrainSamples = self.X_train.shape[0]
        self.numbOfFeatures     = self.X_train.shape[1]
        self.numbOfClasses      = 10
        return

    def next_training_batch(self, batch_size):
        ind      = np.random.randint(self.numbOfTrainSamples, size=batch_size)
        y_onehot = np.zeros((batch_size, self.numbOfClasses))
        y_onehot[np.arange(batch_size), self.y_train[ind]] = 1
        return self.X_train[ind, :], y_onehot

    def get_test_data(self):
        batch_size = self.X_test.shape[0]
        y_onehot = np.zeros((batch_size, self.numbOfClasses))
        y_onehot[np.arange(batch_size), self.y_test] = 1
        return self.X_test, y_onehot


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte




