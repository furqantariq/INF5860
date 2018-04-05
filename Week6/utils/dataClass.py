import numpy as np
import matplotlib.pyplot as plt

class dataClassCNN:
    def __init__(self, datapath):
        X_train, y_train = load_mnist(datapath, kind='train')
        X_test, y_test = load_mnist(datapath, kind='t10k')

        self.numbOfTrainSamples = X_train.shape[0]
        self.numbOfTestSamples  = X_test.shape[0]

        #reshape to 28x28
        X_train = np.resize(X_train, (self.numbOfTrainSamples, 28, 28))
        X_test  = np.resize(X_test, (self.numbOfTestSamples, 28, 28))

        #add depth channel
        X_train = X_train[:,:,:,np.newaxis]
        X_test  = X_test[:, :, :, np.newaxis]

        #cast to float32
        X_train = X_train.astype(dtype=np.float32)
        X_test  = X_test.astype(dtype=np.float32)

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test  -= mean_image

 #       plt.figure()
  #      plt.imshow(X_train[1,:,:,0])
  #      plt.colormaps('gray')

        self.X_train = X_train
        self.X_test  =  X_test
        self.y_train = y_train
        self.y_test  = y_test
        self.numbOfClasses = 10
        self.numbOfFeatures = [X_train.shape[1], X_train.shape[2], X_train.shape[3]]
        self.label_strings = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.testCounter = 0
        self.test_batch_size = 500
        return

    def next_training_batch(self, batch_size):
        ind      = np.random.randint(self.numbOfTrainSamples, size=batch_size)
        y_onehot = np.zeros((batch_size, self.numbOfClasses))
        y_onehot[np.arange(batch_size), self.y_train[ind]] = 1
        return self.X_train[ind, :,:,:], y_onehot

    def get_test_data(self):
        ind = np.linspace(self.testCounter*self.test_batch_size, (self.testCounter+1)*self.test_batch_size-1, num=self.test_batch_size, dtype=np.int32)
        y_onehot = np.zeros((self.test_batch_size, self.numbOfClasses))
        y_onehot[np.arange(self.test_batch_size), self.y_test[ind]] = 1
        self.testCounter = self.testCounter + 1
        if self.testCounter*self.test_batch_size >= self.numbOfTestSamples:
            self.testCounter = 0
        return self.X_test[ind, :,:,:], y_onehot


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

