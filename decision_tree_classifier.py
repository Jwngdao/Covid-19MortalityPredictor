import pandas as pd
import os
from sklearn import tree
import numpy as np
from array import *

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def import_data():
    dataset = pd.read_csv('COVID.csv')


    obj = tree.DecisionTreeClassifier()

    return dataset

def splitdataset(dataset):
    health = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
    doa = dataset.iloc[:, 18]

    X_train, X_test, y_train, y_test = train_test_split(
        health, doa, test_size=0.3, random_state=100)
    return health, doa, X_train, X_test, y_train, y_test



def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_using_entropy(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print(X_test)
    return y_pred





def cal_accuracy(y_test, y_pred):
    print(y_test)
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))

    print("Accuracy :",
          accuracy_score(y_test, y_pred) * 100)


def main():

    data = import_data()
    health,doa, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

    probs = clf_gini.predict_proba(X_test)
    probs = probs[:, 1]

    from sklearn.metrics import roc_curve, roc_auc_score

    auc = roc_auc_score(y_test, probs)

    print('AUROC=%.3f' % (auc))


if __name__=="__main__":
    main()