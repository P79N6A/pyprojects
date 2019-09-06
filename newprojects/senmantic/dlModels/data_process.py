# -*- coding: utf-8 -*-
import random
random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec
import config


def filter_char(arr):
    res = []
    for c in arr:
        if c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return " ".join(res)


def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)

data = pd.read_csv(config.train_data_path)
data.content = data.content.map(lambda x:filter_char(x))
data.content = data.content.map(lambda x:get_char(x))
data.to_csv("preprocess/train_char.csv",index=None)

line_sent = []
for s in data["content"]:
    line_sent.append(s)
w2v_model = Word2Vec(line_sent,size=100,min_count=1,workers=4,iter=15)
w2v_model.wv.save_word2vec_format("w2v/train_char.vector",binary=True)

validation = pd.read_csv(config.validate_data_path)
validation.content = validation.content.map(lambda x:filter_char(x))
validation.content = validation.content.map(lambda x:get_char(x))
validation.to_csv("preprocess/validation_char.csv",index=None)

# test = pd.read_csv(config.test_data_path)
# test.content = test.content.map(lambda x:filter_char(x))
# test.content = test.content.map(lambda x:get_char(x))
# test.to_csv("preprocess/test_char.csv",index=None)