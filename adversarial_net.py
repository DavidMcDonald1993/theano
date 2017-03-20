
# coding: utf-8

# In[1]:

from __future__ import division

from time import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist, cifar100


# In[2]:

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 100     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 100    # size of generated output vector

d_input_size = 100  # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'

minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1


# In[3]:

def get_distribution_sampler(mu, sigma):
#     return lambda n : K.random_normal(mean=mu, std=sigma, shape=(1, n))
    return lambda m, n : np.random.normal(mu, sigma, size=(m, n)).astype("float32")


# In[4]:

def get_generator_input_sampler():
#     return lambda m, n: K.random_uniform(shape=(m, n))
    return lambda m, n: np.random.rand(m, n).astype("float32")


# In[5]:

#samplers for data distribution
gi_sampler = get_generator_input_sampler()
d_sampler = get_distribution_sampler(data_mean, data_stddev)


# In[6]:

class Generator:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape = (input_size,), activation = "elu"))
        self.model.add(Dense(hidden_size, activation = "elu"))
#         self.model.add(Dense(hidden_size, activation = "elu"))
        self.model.add(Dense(output_size, activation = "elu"))


# In[7]:

class Discriminator:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape = (input_size,), activation = "elu"))
        self.model.add(Dense(hidden_size, activation = "elu"))
        self.model.add(Dense(output_size, activation = "sigmoid"))
    
    def compileModel(**kwargs):
        self.model.compile(**kwargs)


# In[8]:

class GeneratorAndDiscriminator:
    
    def __init__(self, generator, discriminator):
        self.model = Sequential()
        self.model.add(generator)
        
        discriminator.trainable = False
        self.model.add(discriminator)
        


# In[21]:

def print_progress(epoch, epochs, start_time):
    
    bar_length = 80
    
    progress_bar = "[" + "=" * int(bar_length * epoch / epochs) + ">" + "-" * int(bar_length * (epochs - epoch) / epochs) + "]"
    
    secs = np.ceil((time() - start_time) * epochs / epoch)
    hours = np.floor(secs / 3600)
    secs -= hours * 3600
    mins = np.floor(secs / 60)
    secs -= mins * 60
    
    sys.stdout.write("\r" + progress_bar + " Epoch {}/{} ETA:{}h {}m {}s".format(epoch, epochs, hours, mins, secs))
    sys.stdout.flush()


# In[22]:

##dicriminator
# dis_in = Input(shape=(d_input_size,))
# dis_1 = Dense(d_hidden_size, activation = "elu")(dis_in)
# dis_2 = Dense(d_hidden_size, activation = "elu")(dis_1)
# dis_3 = Dense(d_output_size, activation = "sigmoid")(dis_2)
# D = Model(dis_in, dis_3)

D = Discriminator(d_input_size, d_hidden_size, d_output_size)

#compile D
D.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[23]:

##generator
# gen_in = Input(shape=(g_input_size,))
# gen_1 = Dense(g_hidden_size , activation = "elu")(gen_in)
# gen_2 = Dense(g_hidden_size , activation = "elu")(gen_1)
# gen_3 = Dense(g_output_size , activation = "sigmoid")(gen_2)
# G = Model(gen_in, gen_3)

G = Generator(g_input_size, g_hidden_size, g_output_size)

#compile G
G.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


# In[24]:

##generator and discriminator

GD = GeneratorAndDiscriminator(G.model, D.model)
GD.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'\n\'\'\'\nmain loop\n\'\'\'\n\nstart_time = time()\n\nfor epoch in range(1, num_epochs+1):\n    \n    for d_step in range(d_steps):\n        \n        #train D on real data\n        d_real_data = d_sampler(100, d_input_size)\n        d_real_targets = np.ones(100)\n        \n        #train D on fake data\n        d_gen_data = gi_sampler(100, g_input_size)\n        d_fake_data = G.model.predict(d_gen_data)\n        d_fake_targets = np.zeros(100)\n        \n        #train model\n        d_data = np.append(d_real_data, d_fake_data, axis=0)\n        d_targets = np.append(d_real_targets, d_fake_targets)\n\n        #fit discriminator\n        D.model.trainable = True\n        D.model.fit(d_data, d_targets, shuffle=True, nb_epoch=10,\n              batch_size=minibatch_size, validation_split=0.0, verbose=0)\n    \n#     print "Completed training of D for epoch {}".format(epoch)\n        \n    for g_step in range(g_steps):\n        \n        #generate data from noise\n        gen_input = gi_sampler(100, g_input_size)\n        \n        #target\n        target = np.ones(100)\n        \n        #fit generator\n        D.model.trainable = False\n        GD.model.fit(gen_input, target, shuffle=True, nb_epoch=10,\n              batch_size=minibatch_size, validation_split=0.0, verbose=0)\n        \n        \n#     print "Completed training of G for epoch {}".format(epoch)\n    \n#     print "Epoch {} complete".format(epoch)\n    print_progress(epoch, num_epochs, start_time)\nprint "\\nDONE"')


# In[ ]:

gen_input = gi_sampler(1000, g_input_size)
forgery = G.model.predict(gen_input)
# print D.model.predict(forgery)
print np.mean(forgery)


# In[46]:

d_real_data = d_sampler(100, d_input_size)


# In[47]:

d_real_data


# In[ ]:



