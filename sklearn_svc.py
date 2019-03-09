import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

def main():
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 100]}
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=109)
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #print("Precision:", metrics.precision_score(y_test, y_pred))
    #print("Recall:", metrics.recall_score(y_test, y_pred))

main()
