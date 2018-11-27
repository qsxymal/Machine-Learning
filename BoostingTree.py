from CART import RegressionTree
import matplotlib.pyplot as plt
from math import log
import numpy as np


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

class BoostingTreeBase():
    def __init__(self):
        '''
        _maxDepth:分类器数量
        _tolerance:分类容差
        _input_dim:输入数据的维度
        _classify_list:分类器list
        _alpha:分类器权值，#分类树需要，因为每个分类器的权值不一样，回归树权值都是1
        _error_rate_list:分类误差
        _y_train:输入数据，#回归树需要，因为每个分类器的最终结果y是更新的，需要保留最初的输入y，分类树y不变
        '''
        self._maxDepth = None
        self._tolerance = None
        self._input_dim = None
        self._classify_list = []
        self._alpha = []
        self._error_rate_list = []
        self._y_train = None

    def printClassify(self):
        """
        打印分类器的规则，即每个分类数的特征及特征值
        :return:
        """
        for i in range(len(self._classify_list)):
            if len(self._alpha) != 0:
                print(i, ': alpha =', self._alpha[i])
            else:
                print(i, ':')
            self._classify_list[i].printRule()

    def getDepth(self):
        """
        获取基本分类器数量
        :return:
        """
        return len(self._classify_list)

    def train(self, x_train, y_train, maxDepth=None, min_sample_num=None):
        """
        训练分类器
        :param x_train:
        :param y_train:
        :param maxDepth: 基本分类器的最大深度
        :param min_sample_num: 基本分类器的最小样本
        :return:
        """
        assert len(x_train) == len(y_train)
        self._y_train = y_train
        if type(x_train[0]).__name__ == 'list':
            self._input_dim = len(x_train[0])
        else:
            self._input_dim = 1
        sample_weight = [1.0 / len(x_train)] * len(x_train)
        print('train start:')
        self._getClassify(x_train, y_train, maxDepth, min_sample_num, sample_weight)
        print("The loss is ", self._getClassifyError(x_train, y_train))

    def predict(self, X):
        """
        预测，一个或多个输入都可以
        :param X:
        :return:
        """
        return self._getClassifyOut(X)

    def _getClassify(self, x, y, maxDepth, min_sample_num, sample_weight):
        """
        获取分类器的过程
        :param x:
        :param y:
        :param maxDepth:
        :param min_sample_num:
        :param sample_weight: 样本的权值，分类算法需要，回归不需要
        :return:
        """
        NotImplemented

    def _getClassifyOutWithOne(self, x):
        """
        单输入时，分类器的输出
        :param x:
        :return:
        """
        NotImplemented

    def _getClassifyOut(self, X):
        """
        单输入或多输入时分类器的输出
        :param X:
        :return:
        """
        if self._input_dim == 1:  # 如果是一维输入
            if type(X).__name__ == 'list':  # 如果的多样本预测
                return [self._getClassifyOutWithOne(x) for x in X]
            return self._getClassifyOutWithOne(X)
        else:
            # assert len(X[0]) == self._input_dim
            if type(X[0]).__name__ == 'list':  # 如果的多样本预测
                return [self._getClassifyOutWithOne(x) for x in X]
            return self._getClassifyOutWithOne(X)

    def _getClassifyError(self, x, y):
        """
        获取联合分类器的分类误差
        :param x:
        :param y:
        :return:
        """
        NotImplemented


class BoostingTreeClassify(BoostingTreeBase):
    def __init__(self, maxDepth=5, tolerance=0.1):
        super(BoostingTreeClassify, self).__init__()
        self._maxDepth = maxDepth
        self._tolerance = tolerance
        # self._input_dim = None
        # self._classify_list = []
        # self._alpha = []
        # self._error_rate_list = []

    # def printClassify(self):
    #     for c in self._classify_list:
    #         c.printRule()

    # def getDepth(self):
    #     return len(self._classify_list)

    # def train(self, x_train, y_train, maxDepth=None, min_sample_num=None):
    #     assert len(x_train) == len(y_train)
    #     self._y_train = y_train
    #     sample_weight = [1.0 / len(x)] * len(x)
    #     print('train start:')
    #     self._getClassify(x_train, y_train, maxDepth, min_sample_num, sample_weight)
    #     print("The loss is ", self._getClassifyError(x_train, y_train))

    # def predict(self, X):
    #     return [self._predict(x) for x in X]

    def _getClassify(self, x, y, maxDepth, min_sample_num, sample_weight):
        #选择分类器
        rt = RegressionTree(regression=False)
        if maxDepth is None:
            if min_sample_num is None:
                rt.train(x, y, sample_weight=sample_weight)
            else:
                rt.train(x, y, min_sample_num=min_sample_num, sample_weight=sample_weight)
        else:
            if min_sample_num is None:
                rt.train(x, y, maxDepth=maxDepth, sample_weight=sample_weight)
            else:
                rt.train(x, y, maxDepth=maxDepth, min_sample_num=min_sample_num, sample_weight=sample_weight)
        if rt.getDepth() is not None:
            #计算分类器误差及误差样本索引
            error_rate, error_index = self._getGmError(rt, x, y, sample_weight)
            # print('error_rate, error_index', error_rate, error_index)
            #更新分类器
            self._classify_list.append(rt)
            self._alpha.append(self._getClassifyWeights(error_rate))
            #计算分类器误差
            error = self._getClassifyError(x, self._y_train)
            self._error_rate_list.append(error)
            #若误差达不到要求，需要更新样本再继续迭代生成新的分类器
            if error > self._tolerance:
                if len(self._classify_list) < self._maxDepth:
                    #更新样本权重
                    w_list = self._getSampleWeights(error_rate, error_index, len(x))
                    sample_weight_update = [s * w for s, w in zip(sample_weight, w_list)]
                    # print('sum(sample_weight_update)',sum(sample_weight_update))
                    self._getClassify(x, y, maxDepth, min_sample_num, sample_weight_update)

    def _getClassifyOutWithOne(self, x):
        y_pred = 0
        # print('self._alpha',self._alpha)
        for i in range(len(self._classify_list)):
            classify_out = self._classify_list[i].predict(x)
            classify_out *= self._alpha[i]
            y_pred += classify_out
            # classify_out = self._classify_list[i].predict(x)
            # classify_out = [ci * self._alpha[i] for ci in classify_out]
            # y_pred = [yi_pred + ci for yi_pred, ci in zip(y_pred, classify_out)]
        return sign(y_pred)

    # def _getClassifyOut(self, x):
    #     if self._input_dim == 1:  # 如果是一维输入
    #         if type(X).__name__ == 'list':  # 如果的多样本预测
    #             return [self._getClassifyOutWithOne(x) for x in X]
    #         return self._getClassifyOutWithOne(x)
    #     else:
    #         # assert len(X[0]) == self._input_dim
    #         if type(X[0]).__name__ == 'list':  # 如果的多样本预测
    #             return [self._getClassifyOutWithOne(x) for x in X]
    #         return self._getClassifyOutWithOne(x)

    def _getClassifyError(self, x, y):
        error = 0
        y_pred = self._getClassifyOut(x)
        for i in range(len(y)):
            error += int(y_pred[i] != y[i])
        return error / len(y)

    #以下函数是非继承子类的
    def _getGmError(self, classify, x, y, sample_weight):
        """
        计算新的单分类器的分类误差，用来更新样本和分类器的权值
        :param classify:
        :param x:
        :param y:
        :param sample_weight:
        :return:
        """
        error_rate = 0
        error_index = []
        for i in range(len(x)):
            if y[i] != classify.predict(x[i]):
                error_rate += sample_weight[i]
                error_index.append(i)
        return error_rate, error_index

    def _getClassifyWeights(self, e):
        """
        获取分类的权值
        :param e:
        :return:
        """
        return 1. / 2 * log((1 - e) / e)

    def _getSampleWeights(self, e, error_index, x_length):
        '''
        获取样本的权值
        :param e:
        :param error_index:
        :param x_length:
        :return:
        '''
        w_list = []
        for i in range(x_length):
            if i in error_index:
                w = 1.0 / (2 * e)
            else:
                w = 1.0 / (2 * (1 - e))
            w_list.append(w)
        return w_list

class BoostingTreeRegression(BoostingTreeBase):
    def __init__(self, maxDepth=5, tolerance=0.1):
        super(BoostingTreeRegression, self).__init__()
        self._maxDepth = maxDepth
        self._tolerance = tolerance
        # self._input_dim = None
        # self._y_train = None
        # self._classify_list = []
        # self._error_rate_list = []

    # def printClassify(self):
    #     for c in self._classify_list:
    #         c.printRule()
    #
    # def getDepth(self):
    #     return len(self._classify_list)

    # def train(self, x_train, y_train, maxDepth=None, min_sample_num=None):
    #     assert len(x_train) == len(y_train)
    #     self._y_train = y_train
    #     sample_weight = [1.0 / len(x)] * len(x)
    #     print('train start:')
    #     self._getClassify(x_train, y_train, maxDepth, min_sample_num, sample_weight)
    #     print("The loss is ", self._getClassifyError(x_train, y_train))

    # def predict(self, X):
    #     return [self._predict(x) for x in X]

    def _getClassify(self, x, y, maxDepth, min_sample_num, sample_weight):
        rt = RegressionTree(regression=True)
        if maxDepth is None:
            if min_sample_num is None:
                rt.train(x, y)
            else:
                rt.train(x, y, min_sample_num=min_sample_num)
        else:
            if min_sample_num is None:
                rt.train(x, y, maxDepth=maxDepth)
            else:
                rt.train(x, y, maxDepth=maxDepth, min_sample_num=min_sample_num)
        # 更新分类器
        if rt.getDepth() is not None:  #如果分类器已经无法再分类了
            self._classify_list.append(rt)
            #计算新的分类器误差
            error = self._getClassifyError(x, self._y_train)
            self._error_rate_list.append(error)
            #若误差不满足，继续迭代
            if error > self._tolerance:
                if len(self._classify_list) < self._maxDepth:
                    #计算残差
                    residual = self._getResidual(x)
                    # print(residual)
                    self._getClassify(x, residual, maxDepth, min_sample_num, sample_weight=None)

    def _getClassifyOutWithOne(self, x):
        y_pred = 0
        for classify in self._classify_list:
            y_pred += classify.predict(x)
            # y_pred = [yi + ci for yi, ci in zip(y_pred, classify.predict(x))]
        return y_pred

    # def _getClassifyOut(self, x):
    #     if self._input_dim == 1:  # 如果是一维输入
    #         if type(X).__name__ == 'list':  # 如果的多样本预测
    #             return [self._getClassifyOutWithOne(x) for x in X]
    #         return self._getClassifyOutWithOne(x)
    #     else:
    #         # assert len(X[0]) == self._input_dim
    #         if type(X[0]).__name__ == 'list':  # 如果的多样本预测
    #             return [self._getClassifyOutWithOne(x) for x in X]
    #         return self._getClassifyOutWithOne(x)

    def _getClassifyError(self, x, y):
        error = 0
        y_pred = self._getClassifyOut(x)
        for i in range(len(y)):
            error += (y_pred[i] - y[i]) ** 2
        return error

    # 以下函数是非继承子类的
    def _getResidual(self, x):
        """
        获取当前联合分类器的残差，用来下一个分类器的输出
        :param x:
        :return:
        """
        y_pred = self._getClassifyOut(x)
        return [yi - yi_pred for yi, yi_pred in zip(self._y_train, y_pred)]

if __name__ == "__main__":
    # x = [[0, 1, 2], [2, 3, 5], [3, 4, 5], [4, 5, 6], [0, 1, 2], [2, 3, 4], [3, 4, 5], [2, 5, 6], [3, 5, 5]]
    # y = [2, 2, 1, 0, 1, 2, 0, 1, 2]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]
    # x = [[1, 5, 2], [2, 2, 1], [3, 1, 4], [4, 6, 3], [6, 8, 1], [6, 5, 5], [7, 9, 1], [8, 7, 3], [9, 8, 0], [10, 2, 4]]
    # y = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
    btr = BoostingTreeRegression(maxDepth=8, tolerance=0.2)
    btr.train(x, y, maxDepth=5, min_sample_num=2)
    btr.printClassify()
    print('Depth =', btr.getDepth())
    print('error_rate = ', btr._error_rate_list)
    print(btr.predict(x))
    plt.plot(btr._error_rate_list)
    plt.show()

    print('--------------------------------------------------------')

    x = [[1, 5], [2, 2], [3, 1], [4, 6], [6, 8], [6, 5], [7, 9], [8, 7], [9, 8], [10, 2]]
    # x = [[1, 5, 2], [2, 2, 1], [3, 1, 4], [4, 6, 3], [6, 8, 1], [6, 5, 5], [7, 9, 1], [8, 7, 3], [9, 8, 0], [10, 2, 4]]
    y = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
    btc = BoostingTreeClassify(maxDepth=8, tolerance=0.01)
    btc.train(x, y, maxDepth=5, min_sample_num=2)
    btc.printClassify()
    print('Depth =', btc.getDepth())
    print('error_rate = ', btc._error_rate_list)
    print(btc.predict(x[0]))
    plt.plot(btc._error_rate_list)
    plt.show()