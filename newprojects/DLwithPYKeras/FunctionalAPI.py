from keras import Input
from keras import layers
from keras import Model

input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(1,activation='sigmoid')(x)

model = Model(input_tensor,output_tensor)

model.summary()


# 多模态输入demo
from keras.models import Model
from keras import layers
from keras import Input

max_text_vocabulary = 10000
max_question_vocabulary = 10000
max_answer_vocabulary = 500

text_input = Input(shape=(None,),dtype='int32', name='text')
text_embedded = layers.Embedding(max_text_vocabulary,64)(text_input)
text_encode = layers.LSTM(64)(text_embedded)

question_input = Input(shape=(None,),dtype='int32',name='question')
question_embedded = layers.Embedding(max_question_vocabulary,64)(question_input)
question_encode = layers.LSTM(64)(question_embedded)

connected = layers.Concatenate([text_encode,question_encode],axis=-1)

answer = layers.Dense(max_answer_vocabulary,activation='softmax')(connected)

model = Model([text_encode,question_encode],answer)
model.compile(optimizer='rmsprop',loss='loss_crossentropy',metrics=['acc'])

# 塞输入
text = []
question = []
answer = []

model.fit([text,question],answer,epochs=10,batch_size=128)


# 多输出模型demo
from keras import layers
from keras import Input
from keras import models

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,),dtype='int32',name='posts')
posts_embedded = layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(posts_embedded)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128,activation='relu')(x)

age_prediction = layers.Dense(1,name='age')(x)

income_prediction = layers.Dense(1,name='income',activation='softmax')(x)

gender_prediction = layers.Dense(1,name='gender',activation='sigmoid')(x)

model = models.Model(posts_input,[age_prediction,income_prediction,gender_prediction])

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

# model.compile(optimizer='rmsprop',loss={
#     'age':'mse',
#     'income':'categorical_crossentropy',
#     'gendier':'binary_crossentropy'
# })

posts = []
age_target,income_target,gender_target = []
model.fit(posts,[age_target,income_target,gender_target],epochs=10,batch_size=128)


# 共享lstm权重demo
from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32)  # 共享lstm的权重

left_input = Input(shape=(None,128))
left_output = lstm(left_input)

right_input = Input(shape=(None,128))
right_output = lstm(right_input)

merged = layers.Concatenate([left_output,right_input],axis=-1)
prediction = layers.Dense(1,activation='sigmoid')(merged)

model = Model([left_input,right_input],prediction)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

train_left,train_right,targets = []
model.fit([train_left,train_right],targets,epochs=10,batch_size=128)


# 模型的回调：
# 1. early stop
# 2. checkpoint
# 3. tensorboard

import keras

callback_list = [
    keras.callbacks.EarlyStopping(monitor='acc',patience=1),
    keras.callbacks.ModelCheckpoint(filepath='my_model.h5',monitor='val_loss',save_best_only=True)
]

model.fit([train_left,train_right],targets,epochs=10,batch_size=128,callbacks=callback_list)


