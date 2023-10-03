import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression

# music_dummies = pd.read_csv("music_clean.csv")
# music_dummies = pd.get_dummies(music_dummies, drop_first=True)
music_df = pd.read_csv('music_clean.csv')
pd.set_option('display.max_columns', None)
# print(f'music_df columns: {music_df.head()[:]}')
# print(f'music_df columns: {music_df.columns}')
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)

music_dummies = pd.concat([music_df, music_dummies], axis=1)
# print(f'music_dummies columns: {music_dummies.columns}')
music_dummies = music_dummies.drop("genre", axis=1)
# print(f'music_dummies columns: {music_dummies.columns}')
# print(f'music_dummies columns: {music_dummies.head()[:]}')
print(f'music_dummies shape: {music_dummies.shape}')

X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
# print(linreg_cv)

# print(np.sqrt(linreg_cv))
# [nan nan nan nan nan]

print(np.sqrt(-linreg_cv))

print(music_dummies.isna().sum().sort_values())
