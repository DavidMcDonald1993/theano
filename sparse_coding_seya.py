
# coding: utf-8

# In[ ]:

# Download a natural image patches dataset
get_ipython().magic(u'matplotlib inline')
import os
import numpy as np
import theano
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop
from seya.layers.coding import SparseCoding
from keras.datasets import mnist


# In[7]:

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[8]:

x_train.shape


# In[9]:

model = Sequential()
model.add(
    SparseCoding(
        input_dim=256,
        output_dim=49, # we are learning 49 filters,
        n_steps = 100, # remember the self.n_steps in the scan loop?
        truncate_gradient=1, # no backpropagation through time today now,
                             # just regular sparse coding
        W_regularizer=l2(.00005),
        return_reconstruction=True # we will output Ax which approximates the input
    )
)

rmsp = RMSprop(lr=.1)
model.compile(loss='mse', optimizer=rmsp) # RMSprop for Maximization as well


# In[ ]:



