"""
This CNN digit recognition model was built using MNIST dataset from Keras library.
The architecture of the model was chosen according to the results, shown in:
https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

"""

from keras.layers import (Dense, Conv2D, MaxPool2D,
                          Flatten, Dropout, BatchNormalization)
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing the x datasets
x_train = x_train / 255
x_test = x_test / 255

# categorizing the y datasets
y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

# reshaping to include the channel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=5, strides=2,
          padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, strides=2,
          padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_cat_train, epochs=20)
