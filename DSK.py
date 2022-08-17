import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

data = pd.read_csv("COVID_10000.csv")
# data.info()
# data.head()
# data.hist(bins=20, figsize=(19,10))
# plt.show()
X = data[['gender', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma',
          'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
          'renal_chronic', 'tobacco', 'contact_other_covid', 'covid_res', 'icu']]

Y = data[["DOA"]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=19)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# #SVM
# clf = SVC(kernel='linear',degree=3, gamma='auto',probability=True)
# clf.fit(x_train,np.ravel(y_train,order='C'))
# y_pred = clf.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# confusion= confusion_matrix(y_test,y_pred)
# #print("\nAccuracy: ", accuracy*100)
# print("Support Vector Machine")
# print("Confusion Matrix:\n",confusion)
# print(classification_report(y_test, y_pred))
# SVMprobs = clf.predict_proba(x_test)
# SVMprobs =SVMprobs[:,1]
# SVM_auc = roc_auc_score(y_test, SVMprobs)
# svm_fpr, svm_tpr, _ = roc_curve(y_test, SVMprobs)
# plt.plot(svm_fpr, svm_tpr, marker='.', label='Support Vector Machine (AUROC = %0.3f)' % SVM_auc)

#probs = clf.predict_proba(x_test)
#probs = probs[:,1]

#KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data.drop('DOA', axis=1))
scaled_features = scaler.transform(data.drop('DOA', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = data.drop('DOA', axis=1).columns)

from sklearn.model_selection import train_test_split
x = scaled_data
y = data['DOA']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)


from sklearn.metrics import confusion_matrix
print("K-Nearest Neighbors")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
KNN_probs = model.predict_proba(x_test)
KNN_probs =KNN_probs[:,1]
KNN_auc = roc_auc_score(y_test, KNN_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, KNN_probs)
plt.plot(knn_fpr, knn_tpr, marker='.', label='K-Nearest Neighbors (AUROC = %0.3f)' % KNN_auc)

#
# error_rates = []
# for i in np.arange(1, 101):
#     new_model = KNeighborsClassifier(n_neighbors = i)
#     new_model.fit(x_train, y_train)
#     new_predictions = new_model.predict(x_test)
#     error_rates.append(np.mean(new_predictions != y_test))

#DTC
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
    #print(X_test)
#print(y_test)
print("DECISION TREE")
print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
DTprobs = clf_gini.predict_proba(X_test)
DTprobs =DTprobs[:,1]
DT_auc = roc_auc_score(y_test, DTprobs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, DTprobs)
plt.plot(dt_fpr, dt_tpr, linestyle='--', label='DECISION TREE (AUROC = %0.3f)' % DT_auc)
#print("Accuracy :",accuracy_score(y_test, y_pred) * 100)
#LOGISIC
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
scX = StandardScaler()
x_train = scX.fit_transform(x_train)
clf_l = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
clf_l.fit(x_train, y_train.values.ravel())
x_test = scX.transform(x_test)
y_pred = clf_l.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
print("LOGISTIC REGRESSION")
print("Confusion Matrix:")
print(confusionMatrix)
print("Report:")
print(classification_report(y_test, y_pred))
LGRprob = clf_l.predict_proba(x_test)
LGRprob = LGRprob[:,1]
lgr_roc_auc = roc_auc_score(y_test , LGRprob)
lgr_fpr, lgr_tpr, _ = roc_curve(y_test, LGRprob)
plt.plot(lgr_fpr, lgr_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % lgr_roc_auc)

# #Random forest
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
rf.fit(x_train, y_train)
rf_pred = rf.predict(X_test)
print("RANDOM FOREST")
print("Confusion Matrix: \n",confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
RFprobs = rf.predict_proba(X_test)
RFprobs =RFprobs[:,1]
RF_auc = roc_auc_score(y_test, RFprobs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, RFprobs)
plt.plot(rf_fpr, rf_tpr, linestyle='-.', label='Random Forest (AUROC = %0.3f)' % RF_auc)

# #naive base
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
nb = GaussianNB()
nb.fit(x_train, y_train.values.ravel())
x_test = sc.transform(x_test)
nb_pred = nb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
print("NAIVE BAYES")
print("Confusion Matrix:")
print(confusionMatrix)
print("Report:")
print(classification_report(y_test, y_pred))
NBprob = nb.predict_proba(x_test)
NBprob = NBprob[:,1]
NB_roc_auc = roc_auc_score(y_test , NBprob)
nb_fpr, nb_tpr, _ = roc_curve(y_test, NBprob)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % NB_roc_auc)







# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() #
# Show plot
plt.show()









