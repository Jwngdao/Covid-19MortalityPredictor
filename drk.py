import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("COVID.csv")
X = data[['gender', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma',
          'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
          'renal_chronic', 'tobacco', 'contact_other_covid', 'covid_res', 'icu']]

Y = data[["DOA"]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=19)

def svm(x_train,y_train,x_test,y_test):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    clf = SVC(kernel='linear',degree=3, gamma='auto',probability=True)
    clf.fit(x_train, np.ravel(y_train,order='C'))
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion= confusion_matrix(y_test,y_pred)
    print("\nSVM Accuracy: ", accuracy*100)
    print("SVM Confusion Matrix:\n",confusion)

def knn():
    scaler = StandardScaler()
    scaler.fit(data.drop('DOA', axis=1))
    scaled_features = scaler.transform(data.drop('DOA', axis=1))
    scaled_data = pd.DataFrame(scaled_features, columns = data.drop('DOA', axis=1).columns)
    x = scaled_data
    y = data['DOA']
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3,random_state=100)
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(x_training_data, y_training_data)
    predictions = model.predict(x_test_data)
    print("Report(KNN):")
    print(classification_report(y_test_data, predictions))
    print("Confusion Matrix(KNN):")
    print(confusion_matrix(y_test_data, predictions))

    #tn,fp,fn,tp = confusion_matrix(y_test_data,predictions).ravel()


def dtc(x_train, y_train, x_test, y_test):
    dt=DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print("DTC Accuracy :",accuracy_score(y_test, y_pred) * 100)
    print("DTC Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
def logr(x_train, y_train, x_test,y_test):
    scX = StandardScaler()
    x_train = scX.fit_transform(x_train)
    clf = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
    clf.fit(x_train, y_train.values.ravel())
    x_test = scX.transform(x_test)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print("Report(LogR):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix(LogR):")
    print(confusionMatrix)


def main():
    dtc(x_train, y_train, x_test, y_test)
    logr(x_train, y_train, x_test, y_test)
    # svm(x_train, y_train, x_test, y_test)
    knn()

if  __name__=="__main__":
    main()


