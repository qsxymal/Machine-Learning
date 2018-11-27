import numpy as np
from keras.datasets import imdb

class NaiveBayes(object):
    def __init__(self):
        self._vocabList = None #词库
        self._wordsVecRate = None #条件概率 每行表示一类
        self._labelsRate = None #先验概率

    def train(self, x_train, y_train):
        '''
        :param x_train: A list [num_test,len]
        :param y_train: A list [num_test]
        :return:
        '''
        self._vocabList = self._createVocabList(x_train)
        self._wordsVecRate, self._labelsRate = self._get_rate0fcondition(x_train, y_train)


    def predict(self, x):
        wordsVec = self._words2Vec(x)
        wordsVec = np.asarray(wordsVec)
        outRate = np.sum(np.log(self._wordsVecRate) * wordsVec, axis=1) + np.log(self._labelsRate)
        y_pred = np.argmax(outRate)
        return y_pred


    def _createVocabList(self, x_train):
        """
        生成输入文本的词库
        """
        vocabSet = set([])
        for x in x_train:
            vocabSet |= set(x)
        return list(vocabSet)

    def _words2Vec(self, document):
        """
        将文本转换成词向量
        """
        wordsVec = [0] * len(self._vocabList)
        for word in document:#遍历每一个单词
            if word in self._vocabList:
                wordsVec[self._vocabList.index(word)] += 1 #词袋模式
                # wordsVec[vocabList.index(word)] = 1  # 词集模式
        return wordsVec

    def _get_rate0fcondition(self, x_train, y_train):
        numtest = len(x_train)#样本总数
        numlabel = len(np.unique(y_train))#类别数量
        numwordsvec = len(self._vocabList)
        #1、将所有文本转换成词向量
        wordsVecList = []
        for x in x_train:
            wordsVec = self._words2Vec(x)
            wordsVecList.append(wordsVec)
        #2、分别计算每类文本的条件概率
        wordsVecCount = np.ones((numlabel, numwordsvec))
        for i in range(numtest):
            wordsVecCount[y_train[i]] += wordsVecList[i]
        wordsVecRate = wordsVecCount / (np.sum(wordsVecCount, keepdims=True) + numlabel)

        #3、计算先验概率
        labelsRate = [0] * numlabel
        label, count = np.unique(y_train, return_counts=True)
        for i in range(numlabel):
            labelsRate[label[i]] = count[i] / numlabel
        labelsRate = np.asarray(labelsRate)
        return wordsVecRate, labelsRate



def loadDataSet():
    '''返回 list'''
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid','quit']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def loadData():
    (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=100)
    p_train_data = train_data[0:10000]
    p_train_labels = train_labels[0:10000]
    val_data = train_data[20000:21000]
    val_labels = train_labels[20000:21000]
    x_train = p_train_data
    y_train = np.asarray(p_train_labels)
    x_train_arr = x_train.tolist()
    y_train_arr = y_train.tolist()

    x_val = val_data
    y_val = np.asarray(val_labels)
    x_val_arr = x_val.tolist()
    y_val_arr = y_val.tolist()
    return (x_train_arr, y_train_arr), (x_val_arr, y_val_arr)



if __name__ == "__main__":

    # x_train, y_train = loadDataSet()
    # NB = NaiveBayes()
    # NB.train(x_train, y_train)
    # for i in range(6):
    #     y_pred = NB.predict(x_train[i])
    #     print(y_pred)

    (wordLists, labels),(x_val,y_val) = loadData()
    NB = NaiveBayes()
    NB.train(wordLists, labels)
    print('-------------------train finished!!---------------------')

    y_pred_train = []
    print(type(wordLists))
    for testEntry in wordLists:
        label = NB.predict(testEntry)
        y_pred_train.append(label)
    train_acc = 1 - len(np.nonzero(np.array(y_pred_train)-np.array(labels))[0])/len(wordLists)
    print('train_acc=', train_acc) #0.5375

    y_pred = []
    for testEntry in x_val:
        label = NB.predict(testEntry)
        y_pred.append(label)
    acc = 1 - len(np.nonzero(np.array(y_pred)-np.array(y_val))[0])/len(x_val)
    print('test_acc=', acc) #0.521