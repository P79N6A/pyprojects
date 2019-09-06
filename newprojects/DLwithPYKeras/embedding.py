from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

max_len = 100  # 100个单词后的评论截断
train_sample = 200
validate_sample = 10000
max_words = 10000  # 只保留数据集中最常见的10000个单词

texts = []
labels = []


# 2. 分词
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(seq, maxlen=max_len)

labels = np.asarray(labels)

print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.rand(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:train_sample]
x_label = labels[:train_sample]
x_validate = data[train_sample:train_sample + validate_sample]
y_validate = data[train_sample:train_sample + validate_sample]


# 3. embedding 预处理
import os
glove_dir = ''

embedding_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

embedding_dim = 100
embedding_matrix = np.zeros(shape=(max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[word] = embedding_vector

# 定义模型
from keras.models import Sequential
from keras.layers import Embedding,Dense,Flatten

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 模型中嵌入glove
model.layers[0].set_weight([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,x_label,epochs=10,batch_size=32,validation_data=(x_validate,y_validate))

model.save('pre_trained_glove_model.h5')












