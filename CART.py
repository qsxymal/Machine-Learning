import numpy as np

class BTree(object):
    def __init__(self, score=None, index=[], usedFeature=[], flag=None):
        """
        :param feature: 特征
        :param point: 分类点
        :param score: 样本均值
        :param index: 子树样本的索引
        :param usedFeature:已经使用过的特征
        :param flag: 左or右子树
        """
        #当前节点的必要信息：特征，特征分类值,
        self.feature = None
        self.point = None
        self.score = score
        self.flag = flag#左右子树
        #向下传递相关的：子树的样本索引，子树的可用特征
        self.index = index
        self.usedFeature = usedFeature
        self.left = None
        self.right = None

    def _data_show(self):
        if self.index is not None:
            print('left:' if self.flag == 0 else 'right:' if self.flag == 1 else 'root',
                  'index:', self.index,
                  'feature:', self.feature, 'point:', self.point, 'usedFeature:', self.usedFeature,
                  'score:', self.score
                  )

    def preOrder(self):
        self._data_show()
        if self.left:
            self.left.preOrder()
        if self.right:
            self.right.preOrder()

    def getDepth(self):
        depth = 1
        depth1 = 0
        depth2 = 0
        if self.left:
            depth1 = self.left.getDepth()
        if self.right:
            depth2 = self.right.getDepth()
        depth += max(depth1, depth2)
        return depth

class RegressionTree(object):
    def __init__(self, regression=True):
        '''
        :param regression:True is regression, mse    False is classify, gini
        '''
        self._root = BTree()
        self._regression = regression
        self._input_dim = None
        self._sample_weights = None

    def printRule(self):
        if self._root.getDepth() != 1:
            self._root.preOrder()

    def getDepth(self):
        if self._root.getDepth() != 1:
            return self._root.getDepth()

    def getError(self, x, y):
        y_pred = self.predict(x)
        if y_pred is not None:
            if self._regression:
                return sum([(yi_pred - yi)**2 for yi_pred, yi in zip(y_pred, y)])
            else:
                return sum([yi_pred != yi for yi_pred, yi in zip(y_pred, y)])

    def train(self, dataSet, classLabel, maxDepth=5, min_sample_num=3, sample_weight=None):
        assert type(dataSet).__name__ == 'list'
        assert type(classLabel).__name__ == 'list'
        assert len(dataSet) == len(classLabel)
        self._sample_weights = sample_weight
        if type(dataSet[0]).__name__ == 'list':
            self._input_dim = len(dataSet[0])
        else:
            self._input_dim = 1
        data_set = np.array(dataSet)
        data_set = data_set.reshape(len(dataSet), -1)
        class_label = np.array(classLabel)
        class_label = np.reshape(class_label, (-1,))
        self._creatTree(data_set, class_label, maxDepth, min_sample_num)
        if self._root.getDepth() == 1:
            print("Warning: the data can not be splitted any more!")

    def predict(self, X):
        if self._root.getDepth() == 1:
            print("Warning: this function can not be used!")
        else:
            if self._input_dim == 1:#如果是一维输入
                if type(X).__name__ == 'list':#如果的多样本预测
                    return [self._predict(x) for x in X]
                return self._predict(X)
            else:
                # assert len(X[0]) == self._input_dim
                if type(X[0]).__name__ == 'list':#如果的多样本预测
                    return [self._predict(x) for x in X]
                return self._predict(X)


    def _predict(self, x):
        """
        对每个样本进行预测
        :param x:
        :return:
        """
        node = self._root
        while node.right and node.left:
            if 1 == self._input_dim:
                x_test = x
            else:
                x_test = x[node.feature]
            if x_test <= node.point:
                node = node.left
            else:
                node = node.right
        return node.score

    def _getGini(self, Label):
        label, count = np.unique(Label, return_counts=True)
        rate = count / sum(count)
        if len(label) == 1:
            return 0
        gini = 2 * rate[0] * rate[1]
        return gini

    def _getMse(self, Label):
        return np.sum(np.square(Label - np.mean(Label)))

    def _getSplitGini(self, dataSet, classLabel, feature, point, sample_index):
        '''
        :param dataSet: 待分类样本
        :param classLabel: 待分类样本的标签
        :param feature: 选择的特征
        :param point: 特征值
        :return:
        '''
        point_label = []
        index = dataSet[:, feature] <= point
        mat_left = classLabel[index]
        mat_right = classLabel[~index]
        point_label.append(mat_left)
        point_label.append(mat_right)

        index_1 = index.astype(np.int)
        index_1 = index_1.tolist()

        weights = 0
        weigts_left = 0
        weigts_right = 0
        if self._sample_weights is not None:
            for i in range(len(sample_index)):
                weights += self._sample_weights[i]
            for i in range(len(index_1)):
                if index_1[i] == 1:
                    weigts_left += self._sample_weights[i]
                else:
                    weigts_right += self._sample_weights[i]
            gini_left = weigts_left / weights * self._getGini(mat_left)
            gini_right = weigts_right / weights * self._getGini(mat_right)
        else:
            gini_left = mat_left.shape[0] / classLabel.shape[0] * self._getGini(mat_left)
            gini_right = mat_right.shape[0] / classLabel.shape[0] * self._getGini(mat_right)
        point_gini = gini_left + gini_right
        # print(self._sample_weights)
        # print(index_1)
        # print(weigts_left, weigts_right, weights, point_gini, point)
        return point_gini, point_label

    def _getSplitMse(self, dataSet, classLabel, feature, point):
        '''
        :param dataSet: 待分类样本
        :param classLabel: 待分类样本的标签
        :param feature: 选择的特征
        :param point: 特征值
        :return:
        '''
        point_mean = [0, 0]
        index = dataSet[:, feature] <= point
        mat_left = classLabel[index]
        mat_right = classLabel[~index]
        point_mean[0] = np.mean(mat_left)
        point_mean[1] = np.mean(mat_right)
        point_mse = np.sum(np.square(mat_left - point_mean[0])) + np.sum(np.square(mat_right - point_mean[1]))
        return point_mse, point_mean

    def _getBestPoint(self, dataSet, classLabel, feature, index):
        '''
        :param dataSet: 待分类样本
        :param classLabel: 待分类样本的标签
        :param feature: 选择的特征
        :param score: 样本的均方差
        :return:
        '''
        point_set = set(dataSet[:, feature])
        class_label_set = set(classLabel)
        #某特征下不可分条件：
            #1、特征值只有一个
            #2、样本标签都一样（冗余）
            #3、分类没有使方差降低（后手）
        #不可分条件下返回None
        if len(point_set) == 1 or len(class_label_set) == 1: #该特征下样本不可分
            return None
        if self._regression == True:
            min_point_mse = self._getMse(classLabel)
        else:
            min_point_mse = self._getGini(classLabel)
        best_point = -1
        min_point_mean = []
        for point in point_set:
            if point == max(point_set):#分类点point不能是最大值，这是由分类条件 <= 决定的(临界点值向下分类)
                continue
            if self._regression == True:
                point_mse, point_mean = self._getSplitMse(dataSet, classLabel, feature, point)
            else:
                point_mse, point_mean = self._getSplitGini(dataSet, classLabel, feature, point, index)
            if point_mse < min_point_mse:
                min_point_mse = point_mse
                min_point_mean = point_mean
                best_point = point
        if best_point == -1:
            return None
        return best_point, min_point_mse, min_point_mean

    def _getBestFeature(self, dataSet, classLabel, min_sample_num, index, usedFeature):
        '''
        :param dataSet: 总样本数据
        :param classLabel: 总样本的标签
        :param index: 样本的索引，得到待分类样本及其标签
        :param usedFeature: 已经用过的特征，不再作为待分类样本的分类特征
        :param score: 待分类样本的均方差
        :return:
        '''
        #没有可分的特征条件：
            #1、样本数量很少
            #2、样本标签一致
            #3、没有可用的特征
            #4、特征没有降低方差，通过flag实现（后手）
        if len(index) <= min_sample_num:
            # print("Only one data")
            return None
        data_set = dataSet[index]
        class_label = classLabel[index]
        if len(set(class_label)) == 1:
            return None
        num_feature = dataSet.shape[1]
        if len(usedFeature) == num_feature:
            return None

        if self._regression == True:
            min_feature_mse = self._getMse(class_label)
        else:
            min_feature_mse = self._getGini(class_label)

        min_feature_mean = []
        best_feature = -1
        best_point = -1
        flag = 0#用来记录是否一致是continue，即该样本在现有特征下不可分或已经没有特征了

        for i in range(num_feature):
            if i in usedFeature:#如果该特征已经使用过，则不再使用
                continue
            result = self._getBestPoint(data_set, class_label, i, index)
            if result == None: #该特征下没有差异
                continue
            min_point, min_point_mse, min_point_mean = result
            if min_point_mse < min_feature_mse:
                min_feature_mse = min_point_mse
                min_feature_mean = min_point_mean
                best_feature = i
                best_point = min_point
            flag = 1 #常伴随continue使用

        if flag == 0:
            # print("This data can not be splitted")
            return None
        #左子树的样本标签mask
        mask = data_set[:, best_feature] <= best_point
        return best_feature, best_point, min_feature_mean, mask

    def _creatTree(self, dataSet, classLabel, maxDepth, min_sample_num):
        #需要解决的问题
        #1、如何进行样本的更新，即分类后的样本
        #2、如何在下一次分类时，避免使用已经使用过的特征
        # 传递样本的绝对索引和已用过的特征，如何传递？
        # 通过父节点存储信息，逐渐向下传递实现

        index = range(dataSet.shape[0])
        self._root = BTree(index=index)
        record = [self._root]
        while record:
            node = record.pop(0)
            # if len(node.index) <= min_sample_num:#待分样本数量太少，不再分(冗余)
            #     if self._regression == False:
            #         node.score = self._getLabel(node.score)
            #     continue
            usedFeature = np.array(node.usedFeature)
            usedFeature = usedFeature.tolist()#因为usedFeature会更新，所以需要避免指向同一个存储位置
            result = self._getBestFeature(dataSet, classLabel, min_sample_num, node.index, usedFeature)
            if result == None: #待分样本不可分
                if self._regression == False:
                    node.score = self._getLabel(node.score)
                continue
            node.feature, node.point, score, index = result
            #更新需要传递的信息
            index_left = np.asarray(node.index)[index]
            index_right = np.asarray(node.index)[~index]
            usedFeature.append(node.feature)
            #生成子树或叶子
            node.left = BTree(score=score[0], index=index_left, usedFeature=usedFeature, flag=0)
            node.right = BTree(score=score[1], index=index_right, usedFeature=usedFeature, flag=1)

            #将子树或叶子存储到堆栈，进行下一次的迭代
            # 如果已经达到最大深度，，则新产生的节点不再放入堆栈中，将堆栈中现有的节点释放完则结束
            if self._root.getDepth() >= maxDepth:
                if self._regression == False:
                    node.left.score = self._getLabel(node.left.score)
                    node.right.score = self._getLabel(node.right.score)
                continue
            record.append(node.left)
            record.append(node.right)
            # print('detail')
            # print(node.index)
            # print(node.feature)
            # print(node.point)
            # print('index_left', index_left)
            # print('index_right', index_right)
            # print(node.left)
            # print(node.right)
        # print(self._root.getDepth())

    def _getLabel(self, labels):
        """
        样本数量多的标签作为该叶子的类别
        :param labels:
        :return:
        """
        label, count = np.unique(labels, return_counts=True)
        index_label = np.argmax(count).tolist() #数组转换成list，如果list只有一个元素，需要小心
        # index_label = list[set(index_label)][0]
        if type(index_label).__name__ == 'list':
            index_label = index_label[0]
        return label[index_label]

if __name__ == "__main__":
    x = np.array(
        [[0, 1, 2], [2, 3, 5], [3, 4, 5], [4, 5, 6], [0, 1, 2], [2, 3, 4], [3, 4, 5], [2, 5, 6], [3, 5, 5]]).tolist()
    # x = [0, 2, 3, 4, 0, 5, 4, 5, 5]
    # y = np.array([[2], [2], [1], [0], [1], [2], [0], [1], [2]]).tolist()
    y = [0.5, 0.0, 0.5, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0]
    rt = RegressionTree()
    rt.train(x, y, maxDepth=3, min_sample_num=2)
    print(rt.getError(x, y))
    rt.printRule()
    print('depth=', rt.getDepth())
    print(rt.predict(x))

    print('-------------------------------------------')

    x = np.array(
        [[0, 1, 2], [2, 3, 5], [3, 4, 5], [4, 5, 6], [0, 1, 2], [2, 5, 4], [6, 4, 5], [2, 5, 6], [3, 5, 5]]).tolist()
    # x = [0, 2, 3, 4, 0, 5, 4, 5, 5]
    y = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1]]).tolist()
    w = [1.0/len(x)] * len(x)
    rt = RegressionTree(regression=False)
    rt.train(x, y, maxDepth=4, min_sample_num=2, sample_weight=w)
    print(rt.getError(x, y))
    rt.printRule()
    print('depth=', rt.getDepth())
    print(rt.predict(x))


    # idnex,count = np.unique([0, 0],return_counts=True)
    # print(type(idnex), type(np.argmax(count).tolist()))

