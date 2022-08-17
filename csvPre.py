import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict

data = pd.read_csv("COVID_10000.csv")
health = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
doa = data.iloc[:, 18]

new_data = pd.read_csv("patients_detail3.csv")
new_data1 = new_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
entry = open('COVID_10000.csv', 'a')
# enter_data = open('COVID_10000.csv','new_data')
d = []
l = []
k = []
r = []
n = []
a = []
def dtc():
    dt = DecisionTreeClassifier()
    dt.fit(health, doa)
    dt_pred = dt.predict(new_data)
    # print(dt_pred)
    for i in dt_pred:
        d.append(i)
    return d


def logisitic():
    scX = StandardScaler()
    health1 = scX.fit_transform(health)
    lr = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
    lr.fit(health1, doa.values.ravel())
    test2 = scX.transform(new_data)
    lr_pred = lr.predict(test2)
    # print(lr_pred)
    for i in lr_pred:
        l.append(i)
    return l


def knn():
    sc = StandardScaler()
    health1 = sc.fit_transform(health)
    test2 = sc.transform(new_data)
    knearest = KNeighborsClassifier()
    knearest.fit(health1, doa)
    knn_pred = knearest.predict(test2)
    # print(knn_pred)
    for i in knn_pred:
        k.append(i)
    return k


def rfc():
    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    rf.fit(health, np.ravel(doa))
    rf_pred = rf.predict(new_data)
    # print(rf_pred)
    for i in rf_pred:
        r.append(i)
    return r



def nb():
    sc = StandardScaler()
    health1 = sc.fit_transform(health)
    test2 = sc.transform(new_data)
    nb = GaussianNB()
    nb.fit(health1, doa.values.ravel())
    nb_pred = nb.predict(test2)
    # print(nb_pred)
    for i in nb_pred:
        n.append(i)
    return n

def arr_com():
    dtc()
    logisitic()
    knn()
    rfc()
    nb()
    a = list(zip(d, l, k, r, n))
    b = list(map(list, a))
    for i, j in zip(b, new_data1):
        # print(i)

        f_out = max(i, key=i.count)

        for x in j:
            entry.write(str(x))
            entry.write(',')

        entry.write(str(f_out))
        entry.write('\n')
        print(f_out)

    # return b


arr_com()
