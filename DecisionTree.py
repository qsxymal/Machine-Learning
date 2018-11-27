import numpy as np
from math import log

class DecisionTree(object):
    def __init__(self, k=1, rate=0.):
        self._k = k
        self._rate = rate
        self._myTree = None
        self._labels = []

    def train(self, x_train, y_train):
        """
        x_train:A list [特征值，类别]
        y_train:A list 特征
        """
        self._labels = y_train
        self._myTree = self._creatTree(x_train, y_train)
        return self._myTree

    def predict(self, x):
        return self._classify(x, self._myTree)

    def _calcEntropy(self, dataset):
        x_arr = np.array(dataset)
        labels_arr = x_arr[:, -1]
        labels, count = np.unique(labels_arr, return_counts=True)
        label_rate = count / np.sum(count)
        return -np.sum(label_rate * np.log2(label_rate))

    def _splitDateByFeature(self, dataset, axis):
        retDataSet = []
        dataset_arr = np.array(dataset)
        feature_arr = np.unique(dataset_arr[:, axis])
        feature_list = feature_arr.tolist()
        for i in range(len(feature_list)):
            mask = np.where(dataset_arr[:, axis] == feature_list[i])[0]
            newdataset = dataset_arr[mask]
            retdata = np.concatenate((newdataset[:, 0:axis], newdataset[:, axis+1:]), axis=1)
            retDataSet.append(retdata.tolist())
        return retDataSet, feature_list

    def _choseBestFeature(self, dataset):
        bestEntropy = self._calcEntropy(dataset)
        bestInfoGain = 0
        bestFeature = -1
        num_feature = len(dataset[0]) - 1

        for axis in range(num_feature):
            newEntropy = 0
            retDataSet, _ = self._splitDateByFeature(dataset, axis)
            for retdata in retDataSet:
                shannonEnt = self._calcEntropy(retdata)
                rate = float(len(retdata)) / len(dataset)
                newEntropy += rate * shannonEnt
            infoGain = bestEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = axis
        bestInfoGainRate = bestInfoGain / bestEntropy
        return bestFeature, bestInfoGainRate

    def _creatTree(self, dataset, featureLabels):
        #1、分类结果完全相同
        #2、分类的种类axis数量已经是0，即数据长度是1
        #3、样本数量很少了
        #4、优化效果很小了
        #返回类别
        if len(np.unique(np.array(dataset)[:, -1])) == 1:
            return dataset[0][-1]
        if len(dataset[0]) == 1 or len(dataset) <= self._k:
            labels, count = np.unique(np.array(dataset)[:, -1], return_counts=True)
            index = np.argmax(count)
            label = labels[index]
            return label
        bestFeature, bestInfoGainRate = self._choseBestFeature(dataset)
        if bestInfoGainRate <= self._rate:
            labels, count = np.unique(dataset[-1], return_counts=True)
            index = np.argmax(count)
            label = labels[index]
            return label

        bestFeatureLabel = featureLabels[bestFeature]
        myTree = {bestFeatureLabel: {}}
        if bestFeature == 0:
            newFeatureLabels = featureLabels[1:]
        elif bestFeature == len(featureLabels) - 1:
            newFeatureLabels = featureLabels[:-1]
        else:
            newFeatureLabels = featureLabels[0:bestFeature].extend(featureLabels[bestFeature+1:])
        retDataSet, featureList = self._splitDateByFeature(dataset, bestFeature)
        for retData, feature in zip(retDataSet, featureList):
            myTreeBottom = self._creatTree(retData, newFeatureLabels)
            myTree[bestFeatureLabel][feature] = myTreeBottom
        return myTree

    def _classify(self, x, inputTree):
        #myTree是dict，获取dict的可以和value，key就是特征，value也是一个dict
        firstKey = next(iter(inputTree))
        index = self._labels.index(firstKey)
        secondDict = inputTree[firstKey]
        for key in secondDict.keys():
            if x[index] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self._classify(x, secondDict[key])
                else:
                    classLabel = secondDict[key]

        return classLabel





# class NodeTree(object):
#     def __init__(self, value):
#         self.left = None
#         self.right = None
#         self.value = value

def createDataSet():  # 创造示例数据
    #    dataSet = [['长', '粗', '高', '男','1'],
    #               ['短', '粗', '高', '男','2'],
    #               ['短', '细', '高', '男','1'],
    #               ['长', '细', '矮', '女','1'],
    #               ['短', '细', '矮', '女','2'],
    #               ['短', '粗', '矮', '女','3'],
    #               ['长', '细', '高', '女','1'],
    #               ['长', '粗', '矮', '女','2'],
    #               ['长', '粗', '矮', '女','2'],
    #               ['长', '粗', '矮', '男','2']]
    #    n_out = 2
    dataSet = [['长', '粗', '高', '男'],
               ['短', '细', '高', '男'],
               ['短', '细', '矮', '女'],
               ['短', '粗', '高', '男'],
               ['短', '粗', '矮', '女'],
               ['长', '细', '矮', '男'],
               ['长', '细', '高', '女'],
               ['长', '粗', '矮', '女'],
               ['长', '粗', '矮', '女'],
               ['长', '粗', '矮', '男']
               ]

    labels = ['头发', '声音', '身高']  # 3个特征

    return dataSet, labels

if __name__ == "__main__":

    # a = np.array([[1,2,1],[2,2,2],[3,3,1],[1,3,2],[2,3,1]])
    # b = ['a', 'b']
    a, b = createDataSet()
    DT = DecisionTree()
    DT.train(a, b)
    for i in a:
        out = DT.predict(i)
        print(out)


