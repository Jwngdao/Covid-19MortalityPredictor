import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


data = pd.read_csv("COVID_10000.csv")
data=data.drop_duplicates(keep='first')
health = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
doa = data.iloc[:, 18]

a = []

def user_input():
    test = []

    entry = open('COVID_10000.csv', 'a')

    a = input("Gender:")
    test.append(a)
    entry.write(a)
    entry.write(",")

    b = input("Intubed:")
    test.append(b)
    entry.write(b)
    entry.write(",")

    c = input("pneumonia:")
    test.append(c)
    entry.write(c)
    entry.write(",")

    d = input("Age:")
    test.append(d)
    entry.write(d)
    entry.write(",")

    e = input("Pregnancy:")
    test.append(e)
    entry.write(e)
    entry.write(",")

    f = input("Diabetes:")
    test.append(f)
    entry.write(f)
    entry.write(",")

    g = input("COPD:")
    test.append(g)
    entry.write(g)
    entry.write(",")

    h = input("Asthma:")
    test.append(h)
    entry.write(h)
    entry.write(",")

    i = input("inmsupr:")
    test.append(i)
    entry.write(i)
    print(i)
    entry.write(",")

    j = input("Hypertension:")
    test.append(j)
    entry.write(j)
    entry.write(",")

    k = input("Other Diseases:")
    test.append(k)
    entry.write(k)
    entry.write(",")

    l = input("Cardiovascular:")
    test.append(l)
    entry.write(l)
    entry.write(",")

    m = input("Obesity:")
    test.append(m)
    entry.write(m)
    entry.write(",")

    n = input("Renal Chronic:")
    test.append(n)
    entry.write(n)
    entry.write(",")

    o = input("Tobacco:")
    test.append(o)
    entry.write(o)
    entry.write(",")

    p = input("Contact with other Covid patient:")
    test.append(p)
    entry.write(p)
    entry.write(",")

    q = input("Covid Res:")
    test.append(q)
    entry.write(q)
    entry.write(",")

    r = input("ICU:")
    test.append(r)
    entry.write(r)
    entry.write(",")

    test1 = []
    test1.append(test)

    outputs = []

    def dtc():
        dt = DecisionTreeClassifier()
        dt.fit(health, doa)
        dt_pred = dt.predict(test1)
        if dt_pred == 0:
            outputs.append(0)
        else:
            outputs.append(1)
        return outputs

    def logr():
        scX = StandardScaler()
        health1 = scX.fit_transform(health)
        lr = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
        lr.fit(health1, doa.values.ravel())
        test2 = scX.transform(test1)
        lr_pred = lr.predict(test2)
        if lr_pred == 0:
            outputs.append(0)
        else:
            outputs.append(1)
        return outputs

    def knn():
        sc = StandardScaler()
        health1 = sc.fit_transform(health)
        test2 = sc.transform(test1)
        knearest = KNeighborsClassifier()
        knearest.fit(health1, doa)
        knn_pred = knearest.predict(test2)
        if knn_pred == 0:
            outputs.append(0)
        else:
            outputs.append(1)
        return outputs

    def rfc():
        rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
        rf.fit(health, np.ravel(doa))
        rf_pred = rf.predict(test1)
        if rf_pred == 0:
            outputs.append(0)
        else:
            outputs.append(1)
        return outputs

    def nb():
        sc = StandardScaler()
        health1 = sc.fit_transform(health)
        test2 = sc.transform(test1)
        nb = GaussianNB()
        nb.fit(health1, doa.values.ravel())
        nb_pred = nb.predict(test2)
        if nb_pred == 0:
            outputs.append(0)
        else:
            outputs.append(1)
        return outputs

    dtc()
    logr()
    knn()
    rfc()
    nb()
    final_output = max(outputs, key=outputs.count)
    if final_output == 0:
        entry.write('0')
        print("Health Condition of Patient is Mild")
    else:
        entry.write('1')
        print("Health Condition of Patient is Serious")
    entry.write("\n")


def csv_input():
    new_data = pd.read_csv("patients_detail1.csv")
    new_data1 = new_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
    new_data_result=new_data.iloc[:, 18]
    print(new_data_result)
    entry = open('COVID_10000.csv', 'a')
    d = []
    l = []
    k = []
    r = []
    n = []

    def dtc():
        dt = DecisionTreeClassifier()
        dt.fit(health, doa)
        dt_pred = dt.predict(new_data1)
        for i in dt_pred:
            d.append(i)
        return d

    def logisitic():
        scX = StandardScaler()
        health1 = scX.fit_transform(health)
        lr = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
        lr.fit(health1, doa.values.ravel())
        test2 = scX.transform(new_data1)
        lr_pred = lr.predict(test2)
        for i in lr_pred:
            l.append(i)
        return l

    def knn():
        sc = StandardScaler()
        health1 = sc.fit_transform(health)
        test2 = sc.transform(new_data1)
        knearest = KNeighborsClassifier()
        knearest.fit(health1, doa)
        knn_pred = knearest.predict(test2)
        for i in knn_pred:
            k.append(i)
        return k

    def rfc():
        rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
        rf.fit(health, np.ravel(doa))
        rf_pred = rf.predict(new_data1)
        for i in rf_pred:
            r.append(i)
        return r

    def nb():
        sc = StandardScaler()
        health1 = sc.fit_transform(health)
        test2 = sc.transform(new_data1)
        nb = GaussianNB()
        nb.fit(health1, doa.values.ravel())
        nb_pred = nb.predict(test2)
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
        y=1
        final_pred=[]
        for i, j in zip(b, new_data1):
            f_out = max(i, key=i.count)
            for x in j:
                entry.write(str(x))
                entry.write(',')
            entry.write(str(f_out))

            final_pred.append(f_out)

            entry.write('\n')
            if f_out==0:
                print("Health Condition of Patient No.",y," is Mild")

            else:
                print("Health Condition of Patient No.",y," is Serious")

            y+=1
        print("______________")
        print("______________")
        print("______________")
        print("______________")
        print("______________")
        print("Accuracy of Decision Tree Classifier:",accuracy_score(new_data_result,d) * 100,"%")
        print("              ____                ")
        print("Accuracy of Logistic Regression:",accuracy_score(new_data_result,l) * 100,"%")

        print("              ____                ")
        print("Accuracy of K-Nearest Neighbour :",accuracy_score(new_data_result,k) * 100,"%")

        print("              ____                ")
        print("Accuracy of Random Forest Classifier :",accuracy_score(new_data_result,r) * 100,"%")

        print("              ____                ")
        print("Accuracy of Naive Bayes:",accuracy_score(new_data_result,n) * 100,"%")

        print("              ____                ")
        print("__________________")
        print("Accuracy of the Final Output:",accuracy_score(new_data_result,final_pred) * 100,"%")
        print("__________________")

        print("              ____                ")

        # return b
    arr_com()


print("WELCOME TO COVID'19 Mortality prediction with Machine Learning Techniques")
print("Report is given by: ")
print("1.User input \n2.CSV file")

print("Enter your choice : ")
x = input()
if x=='1':
    print("Results : ")
    user_input()
elif x=='2':
    print("Results: ")
    csv_input()
else:
    print("Invalid input!!!!")