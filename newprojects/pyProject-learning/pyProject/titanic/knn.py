#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:28:14 2018

@author: withheart
"""

# scipy
#import scipy
#print('scipy: {}'.format(scipy.__version__))
#import numpy
# matplotlib
import matplotlib
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings("ignore")
#print('matplotlib: {}'.format(matplotlib.__version__))
# numpy
import numpy as np # linear algebra
#print('numpy: {}'.format(np.__version__))
# pandas
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#print('pandas: {}'.format(pd.__version__))
import seaborn as sns
#print('seaborn: {}'.format(sns.__version__))
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#print('matplotlib: {}'.format(matplotlib.__version__))

# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

from sklearn.metrics import accuracy_score
# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas import get_dummies
from sklearn.cross_validation import train_test_split


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# ---------------analysis data set-------------------
# explore the data set
# print(train.shape) #the shape of the data set,eg:(891, 12)
# print(train.size) # the size of the train set,that is columns*rows; eg:10692
# print(train.isnull().sum()) #how many NA elements in every column
# remove rows that have NA's
#train = train.dropna()
# print(train.info()) # for getting some information about the dataset
# print(train['Age'].unique()) #see number of unique item 
# print(train['Pclass'].value_counts()) #stat the unique item of selected
# print(train.head(5)) 
# train.sample(5) # to pop up 5 random rows from the data se
# print(train.describe()) #to give a statistical summary about the dataset
# print(train.groupby('Pclass').count())
# print(train.columns) #to print dataset columns
# print(train[train['Age']>7.2])
# ---------------analysis data set-------------------

#----------------clean data-------------------
def set_missing_ages(df):
    Mr_age_mean = (df[df.Name.str.contains('Mr. ')]['Age'].mean())
    Mrs_age_mean = (df[df.Name.str.contains('Mrs. ')]['Age'].mean())
    Miss_age_mean = (df[df.Name.str.contains('Miss. ')]['Age'].mean())
    Master_age_mean = (df[df.Name.str.contains('Master. ')]['Age'].mean())
    df.loc[(df['Name'].str.contains('Dr. ')) & df.Age.isnull(),'Age'] = Mr_age_mean
    df.loc[(df['Name'].str.contains('Mr. ')) & df.Age.isnull(),'Age'] = Mr_age_mean
    df.loc[(df['Name'].str.contains('Mrs. ')) & df.Age.isnull(),'Age'] = Mrs_age_mean
    df.loc[(df['Name'].str.contains('Ms. ')) & df.Age.isnull(),'Age'] = Mrs_age_mean
    df.loc[(df['Name'].str.contains('Miss. ')) & df.Age.isnull(),'Age'] = Miss_age_mean
    df.loc[(df['Name'].str.contains('Master. ')) & df.Age.isnull(),'Age'] = Master_age_mean
    return df

def fill_age_with_rfr(data):
    from sklearn.ensemble import RandomForestRegressor
    age_df = data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
    age_df_notnull = age_df.loc[(data['Age'].notnull())]
    age_df_isnull = age_df.loc[(data['Age'].isnull())]
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]
    rfr = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_df_isnull.values[:,1:])
    data.loc[data['Age'].isnull(),['Age']] = predictAges
    return data

def normalize_age(df):
    df['Age'][df.Age <= 16] = 0
    df['Age'][(df.Age > 16) & (df.Age <= 32)] = 1
    df['Age'][(df.Age > 32) & (df.Age <= 48)] = 2
    df['Age'][(df.Age > 48) & (df.Age <= 64)] = 3
    df['Age'][df.Age > 64 ] = 4
    return df

def normalize_fare(df):
    df.loc[(df.Fare.isnull()),'Fare'] = 0
    df['Fare'][df.Fare <= 7.91] = 0
    df['Fare'][(df.Fare > 7.91) & (df.Fare <= 14.454)] = 1
    df['Fare'][(df.Fare > 14.454) & (df.Fare <= 31)] = 2
    df['Fare'][df.Fare > 31] = 3
    return df

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 1
    df.loc[(df.Cabin.isnull()),'Cabin'] = 0
    return df

def one_hot_encode(origian_data,df):
    dummies_Cabin = pd.get_dummies(df['Cabin'],prefix = 'Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'],prefix = 'Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'],prefix = 'Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'],prefix = 'Pclass')
    dummies_SibSp = pd.get_dummies(df['SibSp'],prefix = 'SibSp')
    #dummies_Parch = pd.get_dummies(df['Parch'],prefix = 'Parch')
    df = pd.concat([origian_data,dummies_Cabin,dummies_Sex, dummies_Embarked, dummies_Pclass,dummies_SibSp], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df
#----------------clean data-------------------

def handle_test():
    df_test = set_missing_ages(test)
    df_test = normalize_age(df_test)
    df_test = normalize_fare(df_test)
    df_test = set_cabin_type(df_test)
    df_test = one_hot_encode(test,df_test)
    test_x = df_test.loc[:,'Age':]
    return test_x

#-----------------prepare feature & targets-----------------
df_train = set_missing_ages(train)
df_train = normalize_age(df_train)
df_train = normalize_fare(df_train)
df_train = set_cabin_type(df_train)
df_train = one_hot_encode(train,df_train)
train_x = df_train.loc[:,'Age':]
train_y = df_train['Survived']

# Splitting the dataset into the Training set and Test set
# =============================================================================
# 测试时使用
# from sklearn.cross_validation import train_test_split
# train_x,test_x ,train_y,test_y = train_test_split(train_x,train_y,test_size = 0.2)
# =============================================================================


#-----------------prepare feature & targets-----------------


#----------------KNN---------------------
def knn(train_x,train_y,test_x,test_y):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 6)
    knn.fit(train_x,train_y)
    y_pred = knn.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('knn accuracy is', accuracy_score(test_y,y_pred))
#----------------KNN---------------------

#----------------lr---------------------
def lr(train_x,train_y,test_x,test_y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(tol=1e-6)
    lr.fit(train_x, train_y)
    y_pred = lr.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('lr accuracy is', accuracy_score(test_y,y_pred))
#----------------lr---------------------


#----------------rf---------------------
def rf(train_x,train_y,test_x,test_y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    rf = RandomForestClassifier(n_estimators = 150,min_samples_leaf = 3,max_depth = 6,oob_score = True)
    rf.fit(train_x,train_y)
    y_pred = rf.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('rf accuracy is', accuracy_score(test_y,y_pred))
    joblib.dump(rf,'rf1.pkl')
#----------------rf---------------------


#----------------svm--------------------
def svn(train_x,train_y,test_x,test_y):
    from sklearn import svm
    svc = svm.SVC()
    svc = svm.SVC(C=1,max_iter=250)
    svc.fit(train_x,train_y)
    y_pred = svc.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('svm accuracy is', accuracy_score(test_y,y_pred))

#----------------svm--------------------


#----------------gbdt-------------------
def gbdt(train_x,train_y,test_x,test_y):
    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(learning_rate = 0.7,max_depth=6,n_estimators=100,min_samples_leaf=2)
    gbdt.fit(train_x,train_y)
    y_pred = gbdt.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('gbdt accuracy is', accuracy_score(test_y,y_pred))
#----------------gbdt-------------------


#----------------xgboost----------------
def xgboost(train_x,train_y,test_x,test_y):
    import xgboost as xgb
    xgb = xgb.XGBClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6)
    xgb.fit(train_x,train_y)
    y_pred = xgb.predict(test_x)
    print(classification_report(test_y,y_pred))
    print(confusion_matrix(test_y,y_pred))
    print('gbdt accuracy is', accuracy_score(test_y,y_pred))

#----------------xgboost----------------



#----------------voting----------------
from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(tol=1e-6)

import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,mun_round=5)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)

vote = VotingClassifier(estimators=[('knn',knn),('lr',lr),('xgb',xgb),('rf',rf),('gbdt',gbdt)])
vote.fit(train_x,train_y)

test_x = handle_test()
y_pred = vote.predict(test_x)

result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':y_pred.astype(np.int32)})
result.to_csv("vote+knn.csv", index=False)
#----------------voting----------------

#----------------stacking--------------
# =============================================================================
# '''模型融合中使用到的各个单模型'''
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# import xgboost as xgb
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# 
# clfs = [LogisticRegression(C=0.1,max_iter=100),
#         xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
#         RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
#         GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]
# 
# # 创建n_folds
# from sklearn.cross_validation import StratifiedKFold
# n_folds = 5
# skf = list(StratifiedKFold(train_y, n_folds))
# 
# # 创建零矩阵
# dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
# dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)))
# 
# # 建立模型
# for j, clf in enumerate(clfs):
#     '''依次训练各个单模型'''
#     # print(j, clf)
#     dataset_blend_test_j = np.zeros((test_x.shape[0], len(skf)))
#     for i, (train, test) in enumerate(skf):
#         '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
#         # print("Fold", i)
#         X_train, y_train, X_test, y_test = train_x[train], train_y[train], test_x[test], test_y[test]
#         clf.fit(X_train, y_train)
#         y_submission = clf.predict_proba(X_test)[:, 1]
#         dataset_blend_train[test, j] = y_submission
#         dataset_blend_test_j[:, i] = clf.predict_proba(test_x)[:, 1]
#     '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
#     dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
# 
# # 用建立第二层模型
# clf2 = LogisticRegression(C=0.1,max_iter=100)
# clf2.fit(dataset_blend_train, train_y)
# y_pred = clf2.predict_proba(dataset_blend_test)[:, 1]
# 
# print(classification_report(test_y,y_pred))
# print(confusion_matrix(test_y,y_pred))
# print('stacking accuracy is', accuracy_score(test_y,y_pred))
# =============================================================================
#----------------stacking--------------
#result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':y_pred.astype(np.int32)})
#result.to_csv("knn.csv", index=False)



