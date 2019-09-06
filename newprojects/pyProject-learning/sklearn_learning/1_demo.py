from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.data)
# print(iris.target)

from sklearn.preprocessing import StandardScaler
print("standard scaler:")
print(StandardScaler().fit_transform(iris.data))

from sklearn.preprocessing import MinMaxScaler
print("min max scaler")
print(MinMaxScaler().fit_transform(iris.data))

from sklearn.preprocessing import Normalizer
print("normalizer")
print(Normalizer().fit_transform(iris.data))

from sklearn.preprocessing import Binarizer
print("binarizer")
print(Binarizer(threshold=3).fit_transform(iris.data))


import pandas as pd





