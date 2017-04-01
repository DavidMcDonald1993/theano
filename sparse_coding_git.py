
# coding: utf-8

# ## Sparse coding
# using code from https://github.com/EderSantana/blog/blob/master/2015-08-02%20sparse%20coding%20with%20keras.ipynb
# 
# # Download a natural image patches dataset
# %matplotlib inline
# import os
# import numpy as np
# import theano
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
# 
# from keras.models import Sequential
# from keras.regularizers import l2
# from keras.optimizers import RMSprop
# from seya.layers.coding import SparseCoding
# 
# S = loadmat('patches.mat')['data'].T.astype(floatX)
# print S.shape

# In[ ]:

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

