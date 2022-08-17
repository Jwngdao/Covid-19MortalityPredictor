# Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

data = pd.read_csv("COVID.csv")
X = data[['gender', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma',
          'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
          'renal_chronic', 'tobacco', 'contact_other_covid', 'covid_res', 'icu']]

Y = data[["DOA"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

scX = StandardScaler()
x_train = scX.fit_transform(x_train)

clf = LogisticRegression(solver="liblinear", C=0.05, multi_class="ovr", random_state=0)
clf.fit(x_train, y_train.values.ravel())

x_test = scX.transform(x_test)

y_pred = clf.predict(x_test)
# print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

confusionMatrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(confusionMatrix)
#
# print("Report:")
# print(classification_report(y_test, y_pred))

# #roc curve
# LGRprob = clf.predict_proba(x_test)
# LGRprob = LGRprob[:,1]
# lgr_roc_auc = roc_auc_score(y_test , LGRprob)
# lgr_fpr, lgr_tpr, _ = roc_curve(y_test, LGRprob)
# plt.plot(lgr_fpr, lgr_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % lgr_roc_auc)
#
# plt.title('ROC Plot')
# # Axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # Show legend
# plt.legend() #
# # Show plot
# plt.show()

# INPUT
new_data = pd.read_csv("patients_detail3.csv")
y_pred = clf.predict(new_data)
print(new_data)
print(y_pred)
