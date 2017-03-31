
# coding: utf-8

# ## Personal implementation of sparse coding layer
# 
# source: https://github.com/EderSantana/blog/blob/master/2015-08-02%20sparse%20coding%20with%20keras.ipynb

# In[3]:

get_ipython().magic(u'matplotlib inline')
import os
import numpy as np
import theano
from scipy.io import loadmat
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop


# In[4]:

class SparseCoding(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform',
                 activation='linear',
                 truncate_gradient=-1,
                 gamma=.1, # 
                 n_steps=10,
                 batch_size=100,
                 return_reconstruction=False,
                 W_regularizer=l2(.01),
                 activity_regularizer=None):
            
            super(SparseCoding, self).__init__()
            self.init = init
            
            self.A = self.init((self.output_dim, self.input_dim)) 
            # contrary to a regular neural net layer, here 
            # the output needs to have the same dimension as
            # as the input we are modeling. Other layers would
            # have self.init((self.input_dim, self.output_dim))
            # as the dimensions of its adaptive coefficients.
    
    def get_output(self, train=False):
        s = self.get_input(train) # input data to be modeled
        initial_x = alloc_zeros_matrix(self.batch_size, self.output_dim) 
        # initialize sparse codes with zeros.
        # Again note that the coefficients here got 
        # output_dim as its last dimension because this 
        # a generative model.
        outputs, updates = theano.scan(
                self._step, # function operated in the main loop
                sequences=[], # iterable input sequences, we don't need this here
                outputs_info=[initial_states, ]*3 + [None, ], # initial states, 
                # I'll explain why we have 4 initial states.
                non_sequences=[inputs, prior], # this is kept the same for the entire for loop
                n_steps=self.n_steps, # since sequences is empty, scan needs this 
                                      # information to know when to stop
                truncate_gradient=self.truncate_gradient) # how much backpropagation 
                                                          # through time/iteration you need.
        if self.return_reconstruction:
                return outputs[-1][-1] # return the approximation of the input
        else:
                return outputs[0][-1] # return the sparse codes


# In[ ]:



