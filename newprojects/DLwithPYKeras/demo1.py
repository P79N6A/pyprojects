from keras import losses
from keras import metrics

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_dim=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsp',loss='binary_crossenntropy',metrics=['acc'])
history = model.fit([],[],batch_size=128,epochs=5)

# 绘制 acc 图
acc = history.history['acc']
val_acc = history.history['val_acc']

import matplotlib.pyplot as plt
plt.plot(5,acc,'bo',label='Training acc')







