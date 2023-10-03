import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
print(type(X), type(y))
X_bmi = X[:, 3]
# print(X_bmi)
X_bmi = X_bmi.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test))
print(scores)
