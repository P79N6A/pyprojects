#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba


# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df


# 分词
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs


# 不分词，直接char试下
def get_char(contents):
    res = []
    for c in contents:
        res.append(c)
    return res
