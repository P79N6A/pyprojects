#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print(iris)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))




