
# coding: utf-8

# In[ ]:

import theano


# In[1]:

import os

print os.environ["THEANO_FLAGS"]


# In[1]:

from __future__ import division

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt


# In[24]:

##set backend to theano
K.set_image_dim_ordering('th')


# In[28]:

#parameter settings
batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3


# In[13]:

#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[15]:

##reshape data
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test =x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


# In[19]:

i = 5
plt.imshow(x_train[i, 0], interpolation='nearest')
plt.show()


# In[26]:

##define model
#TODO -- play around with this

model = Sequential()

##input layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, 
                       border_mode='valid', 
                       input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
#convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
#convolutional layer
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

#fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[29]:

#take validation data from training set automatically
#use validation_data=(x_test, y_test) for explicit setting fo validation data
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
         verbose=1, validation_split=0.2)


# In[ ]:

score = model.evaluate(x_test, y_test, verbose=1)
print 'test score={}'.format(score[0])
print 'test accuracy={}'.format(score[1])
print
print model.predict_classes(x_test[:10])
print y_test[:10]

