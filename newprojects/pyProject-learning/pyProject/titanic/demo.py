#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:56:47 2018

@author: withheart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv('input/test.csv')
print(train_data.head())
print(train_data.info())
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')

def fill_embark_with_mean(data):
    data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values
    return data
    
def fill_cabin_with_unknow(data):
    data['Cabin'] = data.Cabin.fillna('U0')
    return data
    
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

train_data = fill_embark_with_mean(train_data)
train_data = fill_cabin_with_unknow(train_data)
train_data = fill_age_with_rfr(train_data)
print("--------------------------")
print(train_data.info())

print(train_data.groupby(['Sex','Survived'])['Survived'].count())
#train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
#train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
print(train_data.groupby(['Pclass','Survived'])['Survived'].count())
print(train_data['Age'].describe())