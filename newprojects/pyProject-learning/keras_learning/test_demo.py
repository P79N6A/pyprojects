from keras.models import Sequential
from keras.layers.core import Dense,Activation

model = Sequential()
model.add(Dense(units=32,input_shape=(784,)))
model.add(Activation('relu'))
print(model.summary())