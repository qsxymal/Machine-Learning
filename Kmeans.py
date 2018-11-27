import numpy as np
from matplotlib import pyplot as plt

class Kmeans(object):
    def __init__(self, k=1):
        self._k = k

    def train(self):
        pass

    def predict(self,x_train):
        '''
        :param x_train: A numpy array of shape (num_test,dim)
        :return:
        '''
        return self._get_center(x_train)

    def _get_distance(self, x, y):
        return np.sqrt(np.sum(np.power((x-y), 2),axis=1))

    def _get_initCenter(self, x):
        initCenter = np.zeros((self._k, x.shape[1]))
        for i in range(x.shape[1]):
            x_max = np.max(x[:, i])
            x_min = np.min(x[:, i])
            initCenter[:, i] = x_min + (x_max - x_min) * np.random.randn(self._k,1)[:, 0]
        return initCenter

    def _get_center(self, x_train):
        lastCenter = np.zeros((self._k, x_train.shape[1]))
        center = self._get_initCenter(x_train)
        while self._get_distance(lastCenter, center) > 1e-3:
            sumDistance = np.zeros((self._k, x_train.shape[1]))
            count = np.zeros((self._k, 1)) + 1e-3
            labels = []#记录每个样本的所属类别
            lastCenter = center
            #给每个样本分类
            for i in range(x_train.shape[0]):
                distance = self._get_distance(x_train[i], center)
                # print(distance.shape)
                index = np.argmin(distance)
                sumDistance[index] += x_train[i]
                count[index] += 1
                labels.append(index)
            #计算新的类中心
            center = sumDistance / count

            #计算每个类别的距离和
            labelDistance = np.zeros((self._k, 1))
            for i in range(x_train.shape[0]):
                labelDistance[labels[i]] = self._get_distance(x_train[i], center[labels[i]])
            print(center)
        return labels, center, labelDistance




def generate(sample_size, num_classes, mean, cov, diff, one_hot):
    per_sample_size = sample_size // num_classes
    x = np.random.multivariate_normal(mean, cov, per_sample_size)
    y = np.zeros(per_sample_size)

    for i, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean+d, cov, per_sample_size)
        y1 = (1 + i) * np.ones(per_sample_size)
        x = np.vstack((x, x1))
        y = np.hstack((y, y1))
    y = np.reshape(y, (-1, 1))
    z = np.concatenate((x, y), axis=1)
    np.random.shuffle(z)
    x = z[:, :2]
    y = z[:, 2]
    return x, y




if __name__ == "__main__":

    sample_size = 1000
    num_classes = 4
    dim = 2
    mean = np.random.randn(dim)
    cov = np.eye(dim)
    diff = [[4.0, 4.0], [5.0, 0], [0, 5.0]]
    x, y = generate(sample_size, num_classes, mean, cov, diff, one_hot=True)
    print(x.shape)
    print(y.shape)

    color = ['r' if i == 0 else 'b' if i == 1 else 'y' if i == 2 else 'g' for i in y] #[r,b,y,g]
    plt.scatter(x[:, 0], x[:, 1], c=color)
    plt.show()

    km = Kmeans(4)
    labels, _, _ = km.predict(x)
    print("finished!")

    color_show = ['r' if i == 0 else 'b' if i == 1 else 'y' if i == 2 else 'g' for i in labels] #[r,b,y,g]
    plt.scatter(x[:, 0], x[:, 1], c=color_show)
    plt.show()