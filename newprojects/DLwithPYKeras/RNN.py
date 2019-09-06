from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,LSTM,Dense

model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))

model.summary()


modelLSTM = Sequential()
modelLSTM.add(Embedding(10000,32))
modelLSTM.add(LSTM(32))
modelLSTM.add(Dense(1, activation='sigmoid'))


from keras import layers
modelCNN1D = Sequential()
modelCNN1D.add(layers.Conv1D(32,5,input_shape=(None,100),activation='relu'))
modelCNN1D.add(layers.MaxPooling1D(5))
modelCNN1D.add(layers.Conv1D(32,5,activation='relu'))
modelCNN1D.add(layers.GRU(64,dropout=0.2,recurrent_dropout=0.5))
modelCNN1D.add(Dense(1))

modelCNN1D.summary()