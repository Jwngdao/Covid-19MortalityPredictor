## svm: support vector machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix

data = pd.read_csv('COVID.csv')
X = data[['gender', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma',
          'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
          'renal_chronic', 'tobacco', 'contact_other_covid', 'covid_res', 'icu']]

Y = data[["DOA"]]


## split the data into training and testing


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(x_train)9
# print(y_train)
# print(x_test)
# print(y_test)

## svm classifier and finding accuracy
clf = SVC(kernel='linear',degree=3, gamma='auto',probability=True)
clf.fit(x_train, np.ravel(y_train,order='C'))
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusion= confusion_matrix(y_test,y_pred)
print("\nAccuracy: ", accuracy*100)
print("Confusion Matrix:\n",confusion)



# probs = clf.predict_proba(x_test)
# probs = probs[:,1]
#
# from sklearn.metrics import roc_curve, roc_auc_score
#
# auc= roc_auc_score(y_test,probs)
#
# print('AUROC=%.3f'%(auc))