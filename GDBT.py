#GDBT生成了一组树，然后每棵树用来预测样本所在的叶子，组合新的特征？？？？
from BoostingTree import BoostingTreeRegression
import matplotlib.pyplot as plt
from math import log, exp




def sigmoid(x):
    return 1.0 / (1 + exp(-x))


class GDBTRegression(BoostingTreeRegression):
    def __init__(self, maxDepth=8, tolerance=0.2, regression=True):
        super(GDBTRegression, self).__init__()
        self._maxDepth = maxDepth
        self._tolerance = tolerance
        self._init_val = None
        self._regression = regression

    def train(self, x_train, y_train, maxDepth=None, min_sample_num=None):
        assert len(x_train) == len(y_train)
        self._y_train = y_train
        #修改的地方----
        self._init_val = self._getInitVal(y)
        y_train_update = [yi - self._init_val for yi in y_train]
        #--------------
        if type(x_train[0]).__name__ == 'list':
            self._input_dim = len(x_train[0])
        else:
            self._input_dim = 1
        sample_weight = [1.0 / len(x_train)] * len(x_train)
        print('train start:')
        self._getClassify(x_train, y_train_update, maxDepth, min_sample_num, sample_weight)
        print("The loss is ", self._getClassifyError(x_train, y_train))

    def predict(self, X):
        if self._regression:
            return self._getClassifyOut(X)
        else:
            out = self._getClassifyOut(X)
            if type(out).__name__ == 'list':
                return [int(y > 0) for y in out]
            return int(out > 0)

    def _getClassifyOutWithOne(self, x):
        y_pred = self._init_val
        for classify in self._classify_list:
            y_pred += classify.predict(x)
        return y_pred

    def _getInitVal(self, y):
        if self._regression:
            return sum(y) / len(y)
        else:
            return log(sum(y) / (len(y) - sum(y)))

    def _getClassifyError(self, x, y):
        error = 0
        y_pred = self._getClassifyOut(x)
        if self._regression:
            for i in range(len(y)):
                error += (y_pred[i] - y[i]) ** 2
        else:
            # y_pred = [int(yi_pred>0) for yi_pred in y_pred]
            for i in range(len(y)):
                error += int(int(y_pred[i] > 0) != y[i])
            error /= len(y)
        return error


    def _getResidual(self, x):
        y_pred = self._getClassifyOut(x)
        if self._regression:
            residual = [yi - yi_pred for yi, yi_pred in zip(self._y_train, y_pred)]
        else:
            residual = [yi - sigmoid(yi_pred) for yi, yi_pred in zip(self._y_train, y_pred)]
        # residual = [sum(residual) / len(x)] * len(x)
        # print('residual', residual)
        return residual

if __name__ == "__main__":
    # x = [[0, 1, 2], [2, 3, 5], [3, 4, 5], [4, 5, 6], [0, 1, 2], [2, 3, 4], [3, 4, 5], [2, 5, 6], [3, 5, 5]]
    # y = [2, 2, 1, 0, 1, 2, 0, 1, 2]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]
    # x = [[1, 5, 2], [2, 2, 1], [3, 1, 4], [4, 6, 3], [6, 8, 1], [6, 5, 5], [7, 9, 1], [8, 7, 3], [9, 8, 0], [10, 2, 4]]
    # y = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
    gr = GDBTRegression(maxDepth=8, tolerance=0.2)
    gr.train(x, y, maxDepth=5, min_sample_num=2)
    gr.printClassify()
    print('Depth =', gr.getDepth())
    print('error_rate = ', gr._error_rate_list)
    print(gr.predict(x))
    plt.plot(gr._error_rate_list)
    plt.show()

    x = [[1, 5], [2, 2], [3, 1], [4, 6], [6, 8], [6, 5], [7, 9], [8, 7], [9, 8], [10, 2]]
    # x = [[1, 5, 2], [2, 2, 1], [3, 1, 4], [4, 6, 3], [6, 8, 1], [6, 5, 5], [7, 9, 1], [8, 7, 3], [9, 8, 0], [10, 2, 4]]
    y = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    btc = GDBTRegression(maxDepth=8, tolerance=0.01, regression=False)
    btc.train(x, y, maxDepth=5, min_sample_num=2)
    btc.printClassify()
    print('Depth =', btc.getDepth())
    print('error_rate = ', btc._error_rate_list)
    print(btc.predict(x[0]))
    plt.plot(btc._error_rate_list)
    plt.show()