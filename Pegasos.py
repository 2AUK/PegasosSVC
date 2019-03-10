import numpy as np
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split


class SVC(object):

    def __init__(self, gamma, C):
        #Assume linear
        self.kernel = "linear"
        self.gamma = gamma
        self.C = C
        self.w = np.ndarray
        self.X_train = np.ndarray
        self.y_train = np.ndarray

    def predict(self, X_test):
        """Computes f(x) = wx (ignoring bias term for now) and then use a decision function (sign(), really)"""
        return np.asarray([np.sign(y) for y in self.w.dot(X_test.T)])
            
    def fit(self, T):
        """performs SGD for computing minimised w as described in Pegasos paper"""
        m, n_features = self.X_train.shape[0], self.X_train.shape[1]
        self.w = np.zeros(n_features)
        for i in range(T):
            eta = 1 / self.gamma * (i+1)
            j = np.random.choice(m, 1)[0]
            x, y = self.X_train[j], self.y_train[j]
            if y*(self.w.dot(x)) < 1:
                self.w = (1 - eta*self.gamma)*self.w + eta*y*x
            else:
                self.w = (1 - eta*self.gamma)*self.w
                

if __name__ == "__main__":
    svc = SVC(0.001, 100)
    all_data = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(all_data.data, all_data.target, test_size=0.3, random_state=109)
    svc.X_train = X_train
    svc.y_train = y_train
    svc.fit(1000)
    print(svc.predict(X_test))
