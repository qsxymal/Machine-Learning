from random import sample
from math import exp
import numpy as np

class RegressionBase(object):
    def __init__(self, rate=0.1):
        self.bias = None
        self.weight = None
        self.learningRate = rate

    def train(self, x_train, y_train, epochs, sampleRate=1.0, method='batch'):
        assert method in ['batch', 'stochastic']
        if method == 'batch':
            self._batchGradientDescent(x_train, y_train, epochs)
        else:
            self._stochasticGradientDescent(x_train, y_train, epochs, sampleRate)

    def predict(self, x):
        NotImplemented

    def _predict(self, x):
        NotImplemented

    def _getGradientDelta(self, xi, yi):
        return NotImplemented

    def _batchGradientDescent(self, x_train, y_train, epochs):
        num_input = len(x_train)
        dim = len(x_train[0])
        self.bias = 0
        self.weight = [0] * dim
        for _ in range(epochs):
            bias_grad = 0
            weight_grad = [0] * dim
            for i in range(num_input):
                weight_grad_delta, bias_grad_delta = self._getGradientDelta(x_train[i], y_train[i])#这个函数用到了权值
                bias_grad += bias_grad_delta
                weight_grad = [w + w_delta for w, w_delta in zip(weight_grad, weight_grad_delta)]
            self.weight = [self.learningRate * delta_w + w for delta_w, w in zip(weight_grad, self.weight)]
            self.bias += self.learningRate * bias_grad

    def _stochasticGradientDescent(self, x_train, y_train, epochs, sampleRate):
        '''
        随机梯度下降，每来一个样本更新一下权值
        :param x_train: list
        :param y_train: list
        :param epochs: 迭代次数
        :param sanmleNum: 抽样数量
        :return:
        '''
        num_input = len(x_train)
        dim = len(x_train[0])
        sampleNum = int(sampleRate * num_input)
        self.bias = 0
        self.weight = [0] * dim
        for _ in range(epochs):
            for i in sample(range(num_input), sampleNum):
            # for i in range(num_input):
                weight_grad_delta, bias_grad_delta = self._getGradientDelta(x_train[i], y_train[i])
                # self._weight += self._learningRate * weight_grad_delta #list不能这么操作，数组可以
                self.weight = [self.learningRate * delta_w + w for delta_w, w in zip(weight_grad_delta, self.weight)]
                self.bias += self.learningRate * bias_grad_delta

class LinearRegrssion(RegressionBase):

    def _getGradientDelta(self, x, y):
        '''
        y_hat = w * x + b
        Loss = sum(L)
        L = 1/2 * (y - y_hat)^2
        dL/dy_hat = y_hat - y
        dy_hat/dw = x
        dy_hat/db = 1
        dL/dw = dL/dy_hat * dy_hat/dw = (y_hat - y) * x
        dL/db = dL/dy_hat * dy_hat/db = (y_hat - y) * 1
        -dL/dw
        -dL/db
        :param xi:
        :param yi:
        :return:
        '''
        y_hat = self._predict(x)
        bias_grad_delta = y - y_hat
        weight_grad_delta = [bias_grad_delta * xi for xi in x]
        return weight_grad_delta, bias_grad_delta

    def predict(self, X):
        if type(X[0]).__name__ == 'list':
            return [self._predict(x) for x in X]
        return self._predict(X)

    def _predict(self, x):
        return sum([w * xi for w, xi in zip(self.weight, x)]) + self.bias

class LogisticRegression(RegressionBase):

    def _getGradientDelta(self, x, y):
        '''
        z = wx+b
        y_hat = sigmoid(z)::dy_hat/dz = sigmoid(z) * (1-sigmoid(z)) = y_hat * (1-y_hat)
        P = II(p) #希望P值大
        p= y_hat^y  * (1-y_hat)^(1-y)
        logP = sum(logp)
        logL = -logp
        logL = -y * log(y_hat) - (1-y) * log(1-y_hat)
        dlogL/dy_hat = - y / y_hat + (1-y) / (1-y_hat)
        dy_hat/dz = y_hat * (1-y_hat)
        dz/dw = x
        dz/db = 1
        dlogL/dw = -(y - y_hat) * x
        dlogL/db = -(y - y_hat)
        -dlogL/dw
        -dlogL/db
        :param xi:
        :param yi:
        :return:
        '''
        y_hat = self._predict(x)
        bias_grad_delta = y - y_hat
        weight_grad_delta = [bias_grad_delta * xi for xi in x]
        return weight_grad_delta, bias_grad_delta

    def predict(self, X):
        if type(X[0]).__name__ == 'list':
            return [int(self._predict(x) > 0.5) for x in X]
        return int(self._predict(X) > 0.5)

    def _predict(self, x):
        z = sum([w * xi for w, xi in zip(self.weight, x)]) + self.bias
        return sigmoid(z)

def sigmoid(x):
    return 1. / (1 + exp(-x))

if __name__ == "__main__":
    # x = range(9)
    # z = [1, 3, 4, 5, 2, 9, 6, -3, -2]
    # noise = [0.1, 0.2, -0.3, 0.1, 0.1, -0.2, 0.3, 0.1, -0.1]
    # y = [xi * 0.5 + 2 * zi + 2 + noi for xi, zi, noi in zip(x, z, noise)]
    # x = np.array(x)
    # z = np.array(z)
    # x = np.vstack((x, z)).transpose().tolist()
    # print(x)
    # print(y)
    #
    # lg = LinearRegrssion(0.01)
    # lg.train(x, y, 300, sampleRate=1.0, method='stochastic')#
    # # print(lg.weight, lg.bias)
    # print(lg.predict(x))
    #
    # lg = LinearRegrssion(0.01)
    # lg.train(x, y, 200, sampleRate=1.0, method='stochastic')#
    # # print(lg.weight, lg.bias)
    # print(lg.predict(x))
    #
    # lg = LinearRegrssion(0.01)
    # lg.train(x, y, 200, sampleRate=0.8, method='stochastic')#
    # # print(lg.weight, lg.bias)
    # print(lg.predict(x))
    #
    # lg = LinearRegrssion(0.005)
    # lg.train(x, y, 300, sampleRate=1.0, method='batch')
    # # print(lg.weight, lg.bias)
    # print(lg.predict(x))



    x = range(15)
    z = [1, -3, 4, -5, 2, -9, 6, -3, -2, 0, 2, -3, -1, 0, 0.1]
    y = [xi * 0.05 + 0.2 * zi + 0.01 for xi, zi in zip(x, z)]
    y = [int(sigmoid(yi) > 0.5) for yi in y]
    x = np.array(x)
    z = np.array(z)
    x = np.vstack((x, z)).transpose().tolist()
    print(x)
    print(y)

    lg = LogisticRegression(0.01)
    lg.train(x, y, epochs=50, sampleRate=1.0, method='stochastic')
    print(lg.weight, lg.bias)
    print(lg.predict(x))

    lg = LogisticRegression(0.01)
    lg.train(x, y, epochs=50, sampleRate=0.8, method='stochastic')
    print(lg.weight, lg.bias)
    print(lg.predict(x))

    lg = LogisticRegression(0.01)
    lg.train(x, y, epochs=50, sampleRate=1.0, method='batch')
    print(lg.weight, lg.bias)
    print(lg.predict(x))