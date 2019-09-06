import keras
from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPooling1D,Dense

def cnn(seq_num,embedding_dim,max_len):
    model = Sequential()
    model.add(Embedding(seq_num,embedding_dim,input_length=max_len))
    model.add(Conv1D(64,3,activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(64,3,activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dense(4,activation='softmax'))
    return model

def train():
    model = cnn(seq_num=100,embedding_dim=20,max_len=100)
    model.compile(optimizer='adma',loss='',matrix=[])
    model.fit()