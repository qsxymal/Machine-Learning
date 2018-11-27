from math import log, sqrt


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

class Classify(object):
    def __init__(self, feature, point, label, alpha, error_rate, error_index):
        """
        :param feature: 分类特征
        :param point: 左右子树的分界值
        :param label: 左子树的标签
        :param alpha: 分类器的权值
        :param error_rate: 分类器的错误率
        :param error_index: 分类器的误分样本索引
        """
        self.feature = feature
        self.point = point
        self.label = label
        self.alpha = alpha
        self.error_rate = error_rate
        self.error_index = error_index

    def print(self):
        print('alpha:', self.alpha, 'feature:', self.feature,
              'point:', self.point, 'label:', self.label,
              self.error_rate, self.error_index)

    def _predict(self, x):
        if type(x).__name__ == 'list':
            value = x[self.feature]
        else:
            value = x
        if value <= self.point:
            y = self.label
        else:
            y = -self.label
        y = y * self.alpha
        return y

    def predict(self, X):
        """
        :param X:
        :return: 分类器的分数
        """
        return [self._predict(x) for x in X]

#本算法的基本分类器是决策树桩（decision stump),即每个分类器都是由一个根节点直接连接两个叶子结点
class AdaBoost(object):
    def __init__(self, maxDepth=5):
        #成员变量既可以表示类的属性，同时也用来作为全局变量
        self.maxDepth = maxDepth
        self.classify_list = []#组合的分类器

    def printClassify(self):
        for classify in self.classify_list:
            classify.print()

    def train(self, x_train, y_train):
        weight = [1.0 / len(x_train)] * len(x_train)
        x = []
        for i in range(len(x_train)):
            x_train[i].append(y_train[i])
            x_train[i].append(weight[i])
            x.append(x_train[i])
        self._getClassify(x)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        y = 0
        for classify in self.classify_list:
            y += classify._predict(x)
        return sign(y)

    def _getSplitData(self, x, feature, point):
        """
        :param x:
        :param feature: 特征
        :param point: 分类点
        :return:
        """
        mat_left = []
        mat_right = []
        for data in x:
            if data[feature] <= point:
                mat_left.append(data)
            else:
                mat_right.append(data)
        return mat_left, mat_right

    def _getLabelWeights(self, x):
        """
        :param x: list，每个元素[样本，样本标签，样本权值]
        :return: 标签的权值和
        """
        label = {-1: 0, 1: 0}
        if len(x) == 1:
            label[x[0][-2]] = 1
            return label
        label[x[0][-2]] = x[0][-1]
        for i in range(1, len(x)):
            if x[i][-2] == x[0][-2]:
                label[x[0][-2]] += x[i][-1]
            else:
                if x[i][-2] in label.keys():
                    label[x[i][-2]] += x[i][-1]
                else:
                    label[x[i][-2]] = x[i][-1]
        return label

    def _getBestPoint(self, x, feature):
        """
                获得最佳的分类的信息：分类点，分类标签，误分率，误分样本标签
                :param x:
                :return:
                """
        best_point = -1
        best_label = -1
        min_error_index = []
        min_error_rate = 1.0
        max_point = max([x[i][feature] for i in range(len(x))])
        for i in range(0, len(x)):
            point  = x[i][feature]
            if point == max_point:
                continue
            # fp = [(c.feature, c.point) for c in self.classify_list]
            # print(fp)
            # if (feature, point) in fp:
            #     continue
            error_index = []
            # 分解数据
            mat_left, mat_right = self._getSplitData(x, feature, point)
            # 获取每组数据的标签权值和
            labels_left = self._getLabelWeights(mat_left)
            labels_right = self._getLabelWeights(mat_right)
            # 获取分组的最佳标签及错误率,左边标签都是label
            if labels_left[-1] + labels_right[1] > labels_left[1] + labels_right[-1]:
                label = -1
                error_rate = labels_left[1] + labels_right[-1]
            else:
                label = 1
                error_rate = labels_left[-1] + labels_right[1]
            # 获得误分样本标签
            for j in range(len(x)):
                if x[j][feature] <= point:
                    if x[j][-2] != label:
                        error_index.append(j)
                else:
                    if x[j][-2] == label:
                        error_index.append(j)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                min_error_index = error_index
                best_label = label
                best_point = point
        return best_point, best_label, min_error_rate, min_error_index

    def _getBestClassify(self, x):
        """
        获得最佳的分类的信息：分类点，分类标签，误分率，误分样本标签
        :param x:
        :return:
        """
        best_feature = -1
        best_point = -1
        best_label = -1
        min_error_index = []
        min_error_rate = 1.0

        for feature in range(len(x[0]) - 2):
            point, label, error_rate, error_index = self._getBestPoint(x, feature)
            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_feature = feature
                best_point = point
                best_label = label
                min_error_index = error_index
        return best_feature, best_point, best_label, min_error_rate, min_error_index

    def _getClassifyWeights(self, e):
        return 1. / 2 * log((1 - e) / e)

    def _getSampleWeights(self, e, error_index, x_length):
        z = 2 * sqrt(e * (1 - e))
        w_list = []
        for i in range(x_length):
            if i in error_index:
                w = 1.0 / (2 * e)
            else:
                w = 1.0 / (2 * (1 - e))
            w = w / z
            w_list.append(w)
        return w_list

    def _getOneClassify(self, x):
        """
        获取新的一个分类器
        :param x:
        :return:
        """
        # 选择分类器及分类误差
        feature, point, label, error_rate, error_index = self._getBestClassify(x)
        # 分类器权值
        alpha = self._getClassifyWeights(error_rate)
        # 更新分类器
        classify = Classify(feature, point, label, alpha, error_rate, error_index)
        return classify

    def _getClassify(self, x):
        #新的分类器
        classify = self._getOneClassify(x)
        #更新分类器（用全局变量，这样分类器不用传递，每次都会依次更新）
        self.classify_list.append(classify)
        #计算分类器误分个数
        y = [0] * len(x)
        for c in self.classify_list:
            y = [yi + ci for yi, ci in zip(y, c.predict(x))]
        y = [sign(yi) for yi in y]
        error_count = 0
        for i in range(len(x)):
            if y[i] != x[i][-2]:
                error_count += 1
        #仍存在误分，则需要更新样本权值
        if error_count > 0:
            if len(self.classify_list) < self.maxDepth:
                #样本更新率
                w_list = self._getSampleWeights(classify.error_rate, classify.error_index, len(x))
                #权值更新
                for i in range(len(x)):
                    x[i][-1] *= w_list[i]
                # print(w_list)
                # test = [xi[-1] for xi in x]
                # print(test)
                #进行迭代
                self._getClassify(x)



if __name__ == "__main__":
    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    # # y = [1, -1, 1, -1, 1, -1, 1, 1, -1, -1]



    #比较以下两组x，得到的分类器深度不一样，理论上来说，第二组肯定可以用第一组的分类器进行分类，但实际第二组分类器深度更深
    #原因就在于：上一轮优秀的分类器的误分样本，不是很好分类，所以造成后续分类难度加大
    #当前的分类器只能考虑现有样本分类出差最小，不能兼顾后续是否好分类
    x = [[1, 5], [2, 2], [3, 1], [4, 6], [6, 8], [6, 5], [7, 9], [8, 7], [9, 8], [10, 2]]
    # x = [[1, 5, 2], [2, 2, 1], [3, 1, 4], [4, 6, 3], [6, 8, 1], [6, 5, 5], [7, 9, 1], [8, 7, 3], [9, 8, 0], [10, 2, 4]]
    y = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
    ab = AdaBoost(maxDepth=8)
    ab.train(x, y)
    ab.printClassify()
    print(ab.predict(x))
