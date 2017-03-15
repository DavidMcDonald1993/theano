
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist, cifar100


# In[2]:

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1


# In[3]:

def get_distribution_sampler(mu, sigma):
#     return lambda n : K.random_normal(mean=mu, std=sigma, shape=(1, n))
    return lambda n : np.random.normal(mean=mu, std=sigma, shape=(1, n))


# In[4]:

def get_generator_input_sampler():
#     return lambda m, n: K.random_uniform(shape=(m, n))
    return lambda m, n: np.random_uniform(shape=(m, n))


# In[5]:

gi_sampler = get_generator_input_sampler()
d_sampler = get_distribution_sampler(data_mean, data_stddev)


# In[6]:

##generator
gen_in = Input(batch_shape=(minibatch_size, g_input_size))
gen_1 = Dense(g_hidden_size , activation = "elu")(gen_in)
gen_2 = Dense(g_hidden_size , activation = "elu")(gen_1)
gen_3 = Dense(g_hidden_size , activation = "elu")(gen_2)
G = Model(gen_in, gen_3)

#compile G
G.compile(optimizer='rmsprop', loss='binary_crossentropy')


# In[7]:

##dicriminator
dis_in = Input(batch_shape=(minibatch_size, d_input_size))
dis_1 = Dense(d_hidden_size, activation = "elu")(dis_in)
dis_2 = Dense(d_hidden_size, activation = "elu")(dis_1)
dis_3 = Dense(d_hidden_size, activation = "elu")(dis_2)
D = Model(dis_in, dis_3)

#compile D
D.compile(optimizer='rmsprop', loss='binary_crossentropy')


# In[8]:

d_sampler(d_input_size)


# In[50]:

'''
main loop
'''

for epoch in range(num_epochs):
    
    for d_step in range(d_steps):
        
        #train D on real data
        d_real_data = d_sampler(d_input_size)
        d_real_targets = np.ones(d_input_size)
        
        #train D on fake data
        d_gen_data = gi_sampler(minibatch_size, d_input_size)
        d_fake_data = G(d_gen_data)
        d_fake_targets = np.zeros(d_input_size)
        
        #train model
        d_data = np.append(d_real_data, d_fake_data)
        d_targets = np.append(d_real_targets, d_fake_targets)
        
        D.fit(d_data, d_targets, shuffle=True, nb_epoch=1,
              batch_size=minibatch_size, validation_split=0.1)
        
    for g_step in range(g_steps):
        break

