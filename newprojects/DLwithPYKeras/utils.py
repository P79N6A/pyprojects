import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    result = np.zeros((len(sequences),dimension))
    for i,seq in enumerate(sequences):
        result[i,seq] = 1
    return result



from keras.applications import VGG16
# weights:初始化的权重检查点
# include_top:是否包含顶层的密集分类器,imagenet的分类器为1000个,所以应用于具体问题时通常会将这一部分替换
conv_base = VGG16(weights='imagenet',include_top=False)

from keras.preprocessing.image import ImageDataGenerator

base_dir = ''
train_dir = ''
validate_dir = ''
test_dir = ''

dategen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(dir,sample_cnt):
    features = np.zeros(shape=(sample_cnt,4,4,512))
    labels = np.zeros(shape=(sample_cnt))
    generator = dategen.flow_from_directory(
        dir,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size > sample_cnt:
            break
    return features,labels

train_features,train_labels = extract_features(train_dir,2000)
validate_features,validate_labels = extract_features(validate_dir,1000)
test_features,test_labels = extract_features(test_dir,1000)

train_features = np.reshape(train_features,(2000,4*4*512))
validate_features = np.reshape(validate_features,(1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))



from keras import models
from keras import layers

conv_base.trainable = False  # 预训练模型不参与权重更新
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


from keras.preprocessing.text import Tokenizer
samples = ['','']

# 设置一个分词器，只考虑最常见的前1000个词汇
tokenizer = Tokenizer(num_words=1000)

# 构建单词索引
tokenizer.fit_on_texts(samples)

# 将字符串转化为整数索引组成的列表
seq = tokenizer.texts_to_sequences(samples)

one_hot = tokenizer.texts_to_matrix(samples,mode='binary')

# 找回单词索引
word_index = tokenizer.word_index
print("Found %s unique token." % len(word_index))

import numpy as np


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size > max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

