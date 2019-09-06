# -*- coding: utf-8 -*-
# blog link : https://www.cnblogs.com/pinard/p/6016029.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# solve：RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

data = pd.read_csv('/Users/withheart/Documents/studys/ljp-blogs/datas/CCPP/ccpp.csv')
# blog code:  from sklearn.cross_validation import train_test_split
# however cross_validation has been deprecated
from sklearn.model_selection import train_test_split
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.intercept_)
print(linreg.coef_)

#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()