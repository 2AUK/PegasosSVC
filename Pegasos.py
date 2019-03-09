import numpy as np
import csv


class SVC(object):

    def __init__(self, gamma, C):
        #Assume linear
        self.kernel = "linear"
        self.gamma = gamma
        self.C = C
        self.w = np.ndarray
        self.X_train = np.ndarray
        self.X_test = np.ndarray
        self.y_test = np.ndarray
        self.y_pred = np.ndarray

    def read_file(input_file):
        with open(input_file, 'r') as ifile:
            pass
            

    def predict():
        """Computes f(x) = wx (ignoring bias term for now)"""
        pass

    def optimise_weights():
        """performs SGD for computing minimised w as described in Pegasos paper"""
        pass

    def preprocess_data():
        """splitting data in to testing and training tests"""
        pass




if __name__ == "__main__":
    svc = SVC(0.001, 100)
