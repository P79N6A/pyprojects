import json
from collections import namedtuple
from third_utils import read_vocab
from tensorflow.contrib import learn

import numpy as np
import jieba
import pandas as pd


UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs


def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list), max_len), np.int32)
    for i, t in enumerate(tokens_list):
        t = t + (max_len - len(t)) * [EOS_ID]
        ret[i] = t
    return ret


def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c, UNK_ID))
        return ids

    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t, UNK_ID) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS_ID] + ids[:max_tokens] + [EOS_ID]
    return tokens


class DataItem(namedtuple("DataItem", ('content', 'length', 'label', 'id'))):
    pass


class DataSet(object):
    def __init__(self, datas, labels, batch_size=32, reverse=False, split_word=True, max_len=1200):
        self.reverse = reverse
        self.split_word = split_word
        self.batch_size = batch_size
        self.max_len = max_len
        self.labels = labels
        self.datas = datas

        self.tag_l2i = {"1": 0, "0": 1, "-1": 2, "-2": 3}
        self.tag_i2l = {v: k for k, v in self.tag_l2i.items()}

        self._raw_data = []
        self.items = []
        self._preprocess()

    def get_label(self, label,  normalize=False):
        one_hot_labels = np.zeros(len(self.tag_l2i), dtype=np.float32)
        one_hot_labels[self.tag_l2i[str(label)]] = 1
        return one_hot_labels

    def _preprocess(self):
        print("# Start to preprocessing data...")
        vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_len)
        contents = np.array(list(vocab_processor.fit_transform(self.datas)))
        vocab_size = len(vocab_processor.vocabulary_)
        print("vocab_size:" + str(vocab_size))
        for i,content in enumerate(contents):
            # content = _tokenize(content, self.w2i, self.max_len, self.reverse, self.split_word)
            label = self.get_label(self.labels[i], self.tag_l2i)
            self._raw_data.append(
                DataItem(content=content, label=np.asarray(label), length=len(content), id=i))

        self.num_batches = len(self._raw_data) // self.batch_size
        self.data_size = len(self._raw_data)
        print("# Got %d data items with %d batches" % (self.data_size, self.num_batches))

    def _shuffle(self):
        # code from https://github.com/fastai/fastai/blob/3f2079f7bc07ef84a750f6417f68b7b9fdc9525a/fastai/text.py#L125
        idxs = np.random.permutation(self.data_size)
        sz = self.batch_size * 50
        ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=lambda x: self._raw_data[x].length, reverse=True) for s in ck_idx])
        sz = self.batch_size
        ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self._raw_data[ck[0]].length for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

    def process_batch(self, batch):
        contents = [item.content for item in batch]
        lengths = [item.length for item in batch]
        # contents = _padding(contents, max(lengths))
        lengths = np.asarray(lengths)
        targets = np.asarray([item.label for item in batch])
        ids = [item.id for item in batch]
        return contents, lengths, targets, ids

    def get_next(self, shuffle=True):
        if shuffle:
            idxs = self._shuffle()
        else:
            idxs = range(self.data_size)

        batch = []
        for i in idxs:
            item = self._raw_data[i]
            if len(batch) >= self.batch_size:
                yield self.process_batch(batch)
                batch = [item]
            else:
                batch.append(item)
        if len(batch) > 0:
            yield self.process_batch(batch)
