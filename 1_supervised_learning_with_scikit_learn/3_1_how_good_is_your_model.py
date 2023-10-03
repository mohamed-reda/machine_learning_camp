"""
Accuracy= (TP + TN) / (TP + TN + FP + FN)


Precision (توقعت قد ايه صح مقابل التوقعات اللي قولت صح بس هي خطا):
Precision= TP / (TP + FP)

Recall: (توقعت قد ايه صح مقابل اللي كان صح بس معرفتوش)
Recall= TP / (TP + FN)

F1 Score: harmonic mean of precision and recall
F1 Score= 2 ∗ ( (precision ∗ recall) / (precision + recall) )
"""
import numpy as np
# import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
# print(type(X), type(y))
X_bmi = X[:, 3]
# print(X_bmi)
X_bmi = X_bmi.reshape(-1, 1)

# Confusion matrix in scikit-learn
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Classification report in scikit-learn
# print(f"confusion_matrix: {confusion_matrix(y_test, y_pred)[:3]}")
# confusion_matrix: [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0]]
count = 0
# print(np.nonzero(confusion_matrix(y_test, y_pred)))
print(classification_report(y_test, y_pred))
# for num in confusion_matrix(y_test, y_pred):
#     count += 1
#     # print(num.size)
#     if num.imag != 0:
#         print(f'the index of {count} has : {num}')

# Logistic regression for binary classification
