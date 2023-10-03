# Introduction to regression

import pandas as pd
from pandas.compat import numpy

diabetes_df = pd.read_csv("diabetes_clean.csv")
print(diabetes_df)
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
print(type(X), type(y))
print(X.shape)
print(diabetes_df.shape)

X_bmi = X[:, 3]
# print(X_bmi)
X_bmi = X_bmi.reshape(-1, 1)
# y = X_bmi.reshape(-1, 1)
# print(y.shape, X_bmi.shape)

# Plotting glucose vs. body mass index
import matplotlib.pyplot as plt
#
# plt.scatter(X_bmi, y)

# plt.ylabel("Blood Glucose (mg/dl)")
# plt.xlabel("Body Mass Index")
# plt.show()

# Fitting a regression model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
# show the dots:
plt.scatter(X_bmi, y)
# show the line of the linear reg:
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
