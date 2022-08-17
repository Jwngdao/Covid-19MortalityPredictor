import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('COVID_10000.csv', index_col = 0)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('DOA', axis=1))
scaled_features = scaler.transform(df.drop('DOA', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = df.drop('DOA', axis=1).columns)

from sklearn.model_selection import train_test_split
x = scaled_data
y = df['DOA']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3,random_state=100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))
#
# error_rates = []
# for i in np.arange(1, 101):
#     new_model = KNeighborsClassifier(n_neighbors = i)
#     new_model.fit(x_training_data, y_training_data)
#     new_predictions = new_model.predict(x_test_data)
#     error_rates.append(np.mean(new_predictions != y_test_data))
#
# plt.figure(figsize=(16,12))
# plt.plot(error_rates)
import sklearn
tn,fp,fn,tp = confusion_matrix(y_test_data,predictions).ravel()
# print("True Negatives: ",tn)
# print("False Positives: ",fp)
# print("False Negatives: ",fn)
# print("True Positives: ",tp)
# values = [tn,fp,fn,tp]
# label = '00','01','10','11'
# plt.pie(values,labels=label,autopct='%1.1f%%',radius=1500,frame=True)
# plt.title('confusion matrix')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()



