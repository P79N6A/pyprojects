from keras.datasets import mnist
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


# x_train = x_train.astype('float32')
# y_train = y_train.astype('float32')
# x_test = x_test.astype('float32')
# y_test = y_test.astype('float32')

# normalize
x_train = x_train/255
x_test = x_test/255

# one hot encode
from keras.utils import np_utils
y_train = np_utils.to_categorical(num_classes=10, y=y_train)
y_test = np_utils.to_categorical(num_classes=10,y=y_test)


def LeNet5():
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model


def train_model():
    model = LeNet5()
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.2)
    return model

model = train_model()
model.save('lenet5.h5')
loss,accuracy = model.evaluate(x_test,y_test)
print("loss:" + loss)
print("accuracy:" + accuracy)
