
# coding: utf-8

# In[1]:

from __future__ import division

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt


# In[2]:

#parameter settings
batch_size = 128
nb_classes = 100
nb_epoch = 12

img_rows, img_cols = 32, 32
nb_filters = 32
nb_pool = 2
nb_conv = 3


# In[3]:

#load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# In[4]:

##reshape data
x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
x_test =x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# In[5]:

##define model
#TODO -- play around with this

model = Sequential()

##input layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, 
                       border_mode='valid', 
                       input_shape=(3, img_rows, img_cols)))
model.add(Activation('relu'))
#convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
#pooling layer
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

#convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
#convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
#pooling layer
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

#fully connected layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[6]:

#take validation data from training set automatically
#use validation_data=(x_test, y_test) for explicit setting fo validation data
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
         verbose=1, validation_data=(x_test, y_test))


# In[ ]:



