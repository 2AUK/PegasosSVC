import numpy as np
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm


class SVC(object):

    def __init__(self, C):
        #Assume linear
        self.kernel = "linear"
        self.C = C
        self.w = np.ndarray
        self.X_train = np.ndarray
        self.y_train = np.ndarray

    def predict(self, X_test):
        """Computes f(x) = wx (ignoring bias term for now) and then use a decision function (sign(), really)"""
        return np.sign(np.dot(self.w, X_test.T)).astype(int)
            
    def fit(self, T):
        """performs SGD for computing minimised w as described in Pegasos paper"""
        m, n_features = self.X_train.shape[0], self.X_train.shape[1]
        print(m)
        self.w = np.zeros(n_features, dtype=np.float64)
        for i in range(1, m*100):
            eta = 1. / self.C * float(i)
            j = np.random.randint(m)
            x, y = self.X_train[j], self.y_train[j]
            if y*(np.dot(self.w, x.T)) < 1:
                self.w = (1. - 1/i)*self.w + eta*y*x
            else:
                self.w = (1. - 1/i)*self.w
                

if __name__ == "__main__":
    svc = SVC(2)
    all_data = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(all_data.data, all_data.target, test_size=0.3)
    svc.X_train = X_train
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    svc.y_train = y_train
    #clf = svm.SVC(kernel='linear', max_iter=1000, C=0.5)
    #clf.fit(X_train, y_train)
    svc.fit(1000000)
    print(svc.predict(X_test))
    print(y_test)
