import numpy as np


class Perception(object):
    def __init__(self, rate=0.05):
        self.rate = rate
        self.w = None
        self.b = None
        pass

    def _train(self, x_input, y_input):
        '''
        x_train:A numpy array [num_test,dim]
        y_train:A numpy array [num_test,]
        '''
        print('_train')
        count = 0
        for i in range(x_input.shape[0]):
            f = (np.matmul(x_input[i, :], np.transpose(self.w)) + self.b) * y_input[i]
            if f <= 0:
                self.w += self.rate * np.transpose(x_input[i, :]) * y_input[i]
                self.b += self.rate * y_input[i]
                C = np.sqrt(np.sum(np.square(self.w)))
                self.w /= C
                self.b /= C
                count += 1
        return count

    def train(self, x_input, y_input):
        assert x_input.shape[0] == y_input.shape[0]

        self.w = np.zeros((x_input.shape[1],))
        self.b = 0
        count = 1
        step = 0
        while count != 0 and step < 200:
            count = self._train(x_input, y_input)
            step += 1
            print('count', count)
            print('step:', step)

    def predict(self, x_pred):
        x_pred_arr = np.reshape(np.array(x_pred), (-1, ))
        assert x_pred_arr.shape[0] == 2
        return int(np.matmul(x_pred_arr, np.transpose(self.w)) + self.b > 0)


if __name__ == "__main__":
    x1 = np.random.rand(2, 30)*2 + np.array([[0], [4]])
    x2 = np.random.rand(2, 30)*2 + np.array([[4], [0]])
    x = np.hstack((x1, x2))
    x = np.transpose(x)
    # print(x)
    # print(x.shape)
    y1 = -np.ones((30,))
    y2 = np.ones((30,))
    y = np.hstack((y1, y2))
    y = np.transpose(y)
    # print(y.shape)
    # print(y)
    pp = Perception(rate=0.02)
    pp.train(x, y)
    x = [0,2]
    y = pp.predict(x)
    print(y)
    print(pp.w)
    print(pp.b)





