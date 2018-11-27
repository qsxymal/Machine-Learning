import numpy as np
from keras.datasets import cifar10

class KNN(object):
    def __init__(self, k=1):
        self.k = k
        self.x_train = None
        self.y_train = None
        pass

    def train(self, x_train, y_train):
        """
        x_train:  A numpy array of shape (num_test, D) containing test data
                    consisting of num_test samples each of dimension D
        y_train: A numpy array of shape (num_test,) containing the training labels, where
                     y[i] is the label for X[i]
        """
        assert x_train.shape[0] == y_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x):
        '''
        :param x: A numpy array of shape (1, D)
        :return:
        '''
        labels = self._get_k_neighbor_labels(x)
        y_pred = self._vote(labels)
        return y_pred

    def _get_k_neighbor_labels(self, x):
        distance = np.sum(np.square(x - self.x_train), axis=1)
        index = np.argsort(distance)
        index = index[0:self.k]
        return self.y_train[index]


    def _vote(self, labels):
        y_pred, count = np.unique(labels, return_counts=True)
        return y_pred[np.argmax(count)]


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_datas = train_data[0:5000]
    train_labels = train_labels[0:5000]
    test_datas = test_data[0:500]
    test_labels = test_labels[0:500]

    train_datas = train_datas.reshape(train_datas.shape[0], -1)
    test_datas = test_datas.reshape(test_datas.shape[0], -1)
    train_labels = train_labels.reshape(train_labels.shape[0],)
    test_labels = test_labels.reshape(test_labels.shape[0],)
    print(train_datas.shape)
    print(test_datas.shape)
    print(test_labels.shape)

    knn = KNN(5)
    knn.train(train_datas, train_labels)
    labels = []
    for i in range(test_datas.shape[0]):
        label = knn.predict(test_datas[i])
        labels.append(label)
        print(i)
    labels = np.asarray(labels)
    acc = np.mean(np.equal(labels, test_labels))
    print('acc:', acc) #0.2
