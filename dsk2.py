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


    print(classification_report(y_test_data, predictions))
    print(confusion_matrix(y_test_data, predictions))

    error_rates = []
    for i in np.arange(1, 101):
        new_model = KNeighborsClassifier(n_neighbors = i)
        new_model.fit(x_training_data, y_training_data)
        new_predictions = new_model.predict(x_test_data)
        error_rates.append(np.mean(new_predictions != y_test_data))

    plt.figure(figsize=(16,12))
    plt.plot(error_rates)
    #tn,fp,fn,tp = confusion_matrix(y_test_data,predictions).ravel()


def dtc():
    dt=DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print("DTC Accuracy :",accuracy_score(y_test, y_pred) * 100)
    print("DTC Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
dtc()
knn()
svm(x_train, y_train, x_test, y_test)
