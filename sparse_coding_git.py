
# coding: utf-8

# ## Sparse coding
# using code from https://github.com/EderSantana/blog/blob/master/2015-08-02%20sparse%20coding%20with%20keras.ipynb

# In[1]:

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


# In[2]:

S = loadmat('patches.mat')['data'].T.astype(floatX)
print S.shape


# In[ ]:



