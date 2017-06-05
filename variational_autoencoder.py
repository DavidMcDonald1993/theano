
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Reshape, Flatten, RepeatVector
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist, cifar100


# In[110]:

# use the sampled latent points to generate new latent points z in the latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(latent_dim,), 
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# custom loss function
def vae_loss(x, x_decoded_mean):
    
    # cross entropy term (reconstruction error)
#     xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = original_dim * objectives.mean_squared_error(x, x_decoded_mean)
    
    # KL-divergence term (regularization term)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    
    # sum both terms
    return xent_loss + kl_loss

# apply all layers in a list
def h_apply(h_l, x):
    y = x
    for h in h_l:
        y = h(y)
    return y


# In[111]:

# train vae model on mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

if len(x_train.shape) < 4:
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)
    
image_dim = x_train[0].shape
original_dim = np.prod(image_dim)
latent_dim = 2
# intermediate_dims = np.array([2048, 1024, 512, 256])
intermediate_dims = np.array([256])

nb_epoch = 10
batch_size = 100
epsilon_std = 1.0e-0

num_filters = 3

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# In[112]:

# encoder network to map real inputs to latent space
x = Input(shape=(original_dim,))

# h = [Reshape(target_shape=image_dim)]
# h.append(MaxPooling2D(pool_size=(2,2)))
# h.append(Conv2D(activation="relu", filters=num_filters, padding="same", kernel_size=(5, 5)))
# h.append(MaxPooling2D(pool_size=(2, 2)))
# h.append(Conv2D(activation="relu", filters=num_filters, padding="same", kernel_size=(5, 5)))
# h.append(MaxPooling2D(pool_size=(2, 2)))
# h.append(Flatten())
# h.append(Dense(256, activation="relu"))

h = [Dense(256, activation="tanh")]

# compute latent mean and variance         
z_mean = Dense(units=latent_dim)(h_apply(h, x))
z_log_var = Dense(units=latent_dim)(h_apply(h, x))


# In[113]:

# generate latent points (lambda function layer)
z = Lambda(function=sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# z = Dense(2)(h_apply(h, x))


# In[114]:

# map sampled latent points back to reconstructed points
# we instantiate these layers separately so as to reuse them later

# decoder_h = Dense(intermediate_dim, activation='relu')
# h_decoded = decoder_h(z)

h_decoded = [Dense(256, activation="tanh")]
h_decoded.append(Dense(original_dim, activation="sigmoid"))

# h_decoded.append(Dense(units=num_filters * image_dim[1] / 4 * image_dim[2] / 4, activation="relu"))
# h_decoded.append(Reshape(target_shape=(num_filters, image_dim[1] / 4, image_dim[2] / 4)))
# h_decoded.append(UpSampling2D(size=(2, 2)))
# h_decoded.append(Conv2D(activation="relu", filters=num_filters, padding="same", kernel_size=(5, 5)))
# h_decoded.append(UpSampling2D(size=(2, 2)))
# h_decoded.append(Conv2D(activation="sigmoid", filters=image_dim[0], padding="same", kernel_size=(5, 5)))

# decoder_mean = Dense(original_dim, activation='sigmoid')(h_decoded[-1])
x_decoded_mean = h_apply(h_decoded, z)
# x_decoded_mean = Flatten()(x_decoded_mean)


# In[115]:

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# compile model
vae.compile(optimizer="rmsprop", loss=vae_loss)


# In[91]:

# train model
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=50,
        batch_size=100,
        validation_split=0.0, verbose=True)


# In[92]:

# build a model to project inputs on the latent space
# encoder = Model(x, z_mean)
# encoder = Model(x, z_log_var)
encoder = Model(x, z)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# In[93]:

plt.figure(figsize=(15, 15))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()


# In[94]:

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = h_apply(h_decoded, decoder_input)
generator = Model(decoder_input, _x_decoded_mean)


# In[95]:

# from PIL import Image

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
disp_size = np.array(image_dim[1:])
digit_size = image_dim[-1]
disp_size[-2] = n * digit_size
disp_size[-1] = n * digit_size

figure = np.zeros(tuple(disp_size))

# we will sample n points within [-m, m] standard deviations
m = 3 * epsilon_std
grid_x = np.linspace(-m, m, n)
grid_y = np.linspace(-m, m, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded.reshape(image_dim[1:])
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)

# figure = np.uint8(figure * 255)
# if figure.ndim > 2:
# #     figure = np.swapaxes(figure, 0, figure.ndim -1)
#     for i in range(figure.ndim - 1):
#         figure = np.swapaxes(figure, i, i+1)
# Image.fromarray(figure)
# img.show()


# In[109]:

# generate sample according to prior
z_sample = np.random.normal(loc=0, scale=1, size=(1,2))
print z_sample
generated_digit = generator.predict(z_sample).reshape(28, 28)
plt.imshow(generated_digit)


# In[46]:

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# In[89]:

# G = nx.karate_club_graph()
# G = nx.read_gml("dolphins_labelled.gml")
G = nx.read_gml("embedded_polbooks.gml")
# G = nx.read_gml("embedded_yeast_union.gml")
# G = nx.read_gpickle("embedded_yeast_union.gpickle")
# G = nx.read_gpickle("embedded_yeast_reactome.gpickle")
# G = nx.read_gpickle("hierarchical_benchmark.gpickle")
G = max(nx.connected_component_subgraphs(G), key=len)


# In[90]:

A = np.array(nx.adj_matrix(G).todense())

S1 = A
S2 = cosine_similarity(A)

S = S1 + 0 * S2
S /= np.max(S)


# In[91]:

num_samples, original_dim = A.shape
batch_size = 1
latent_dim = 2
intermediate_dims = [256]
epsilon_std = 1


# In[92]:

x = Input(batch_shape=(batch_size, original_dim))

h = []
for i in range(len(intermediate_dims)):
    h.append(Dense(intermediate_dims[i], activation='relu'))

z_mean = Dense(latent_dim)(h_apply(h, x))
z_log_var = Dense(latent_dim)(h_apply(h, x))


# In[93]:

#generate latent points (lambda function layer)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# In[94]:

h_decoded = []
for i in range(len(intermediate_dims)):
    h_decoded.append(Dense(intermediate_dims[-1-i], activation='relu'))

decoder_mean = Dense(original_dim, activation='sigmoid')
x_decoded_mean = decoder_mean(h_apply(h_decoded, z))


# In[95]:

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# compile VAE model
vae.compile(optimizer="adam", loss=vae_loss)


# In[102]:

# train model
vae.fit(S, S,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_split=0, verbose=True)


# In[103]:

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
S_encoded = encoder.predict(S, batch_size=batch_size)


# In[106]:

polbooks_colour_map = {"c" : "r", "l" : "b", "n" : "g"}


# In[108]:

plt.figure(figsize=(15, 15))


# for label, i, j in zip(G.nodes(), S_encoded[:, 0], S_encoded[:, 1]):
#     plt.annotate(
#         label,
#         xy=(i, j), xytext=(-20, 20),
#         textcoords='offset points', ha='right', va='bottom',
#         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# nodes = np.array(G.nodes())
# for n1, n2 in G.edges():
#     i, = np.where(nodes == n1)
#     j, = np.where(nodes == n2)
#     plt.plot(S_encoded[(i, j), 0], S_encoded[(i, j), 1], c="k", 
#              linewidth = 3 * np.exp(- np.linalg.norm(S_encoded[i] - S_encoded[j]) ** 2 / (2 * 0.5 ** 2) ))
    
plt.scatter(S_encoded[:, 0], 
        S_encoded[:, 1], 
#             c=["b" if v=="Mr. Hi" else "r" for v in nx.get_node_attributes(G, "club").values()],
#             c=nx.get_node_attributes(G, "value").values(),
            c = [polbooks_colour_map[k] for k in nx.get_node_attributes(G,"value").values()],
#             c=nx.get_node_attributes(G, "group").values(), 
#             c="k",
            s=100)


# In[109]:

from sklearn.manifold import MDS


# In[110]:

mds = MDS(dissimilarity="precomputed", n_components=2, metric=True, max_iter=1000)


# In[114]:

D = nx.floyd_warshall_numpy(G)


# In[116]:

D_t = mds.fit_transform(D)


# In[117]:

plt.figure(figsize=(15, 15))
plt.scatter(D_t[:, 0], 
            D_t[:, 1], 
#             c=["b" if v=="Mr. Hi" else "r" for v in nx.get_node_attributes(G, "club").values()],
#             c=nx.get_node_attributes(G, "value").values(), 
            c = [polbooks_colour_map[k] for k in nx.get_node_attributes(G,"value").values()],
#             c=nx.get_node_attributes(G, "group").values(), 
#             c="k",
            s=100)
# nodes = np.array(G.nodes())
# for n1, n2 in G.edges():
#     i, = np.where(nodes == n1)
#     j, = np.where(nodes == n2)
#     plt.plot(S_t[(i, j), 0], S_t[(i, j), 1], c="k", 
#              linewidth = 1e-1 * np.exp(- np.linalg.norm(S_t[i] - S_t[j]) ** 2 / (2 * 0.2 ** 2) ))


# In[ ]:



