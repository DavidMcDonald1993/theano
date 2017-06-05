
# coding: utf-8

# In[120]:

import numpy as np
import scipy as sp
# from keras.datasets import mnist

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import MiniBatchDictionaryLearning


# In[37]:

(train_x, train_y), (test_x, test_y) = mnist.load_data()

num_codes = 10000
input_dim = train_x.shape[1] * train_x.shape[2]


# In[38]:

train_x = train_x.reshape(train_x.shape[0], input_dim).astype(np.float32)


# In[40]:

train_x /= 255


# In[122]:

dico = MiniBatchDictionaryLearning(n_components=10, alpha=0.01, n_iter=1000)


# In[123]:

D = dico.fit(X).components_


# In[128]:

D


# In[129]:

sparse_coder = SparseCoder(dictionary=D, transform_alpha=0.01)


# In[130]:

X_new = sparse_coder.fit_transform(X)


# In[133]:

X_new.shape


# In[134]:

D.shape


# In[135]:

X.shape


# In[139]:

np.dot(X_new, D)


# In[140]:

X


# In[9]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[10]:

plt.imshow(D[0].reshape(28, 28))


# In[37]:

plt.imshow(D[1].reshape(28, 28))


# In[1]:

from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve


# In[9]:

X, t = make_s_curve(n_samples=100, noise=0.1)


# In[21]:

X, y = make_blobs(n_samples=1000, centers=3, cluster_std=2.5)


# In[5]:

get_ipython().magic(u'matplotlib inline')


# In[6]:

import matplotlib.pyplot as plt


# In[22]:

plt.scatter(X[:,0], X[:,1])


# In[11]:

t


# In[12]:

X.shape


# In[19]:

from sklearn.mixture import GaussianMixture


# In[69]:

gm = GaussianMixture(n_components=3, max_iter=1000, covariance_type="tied")


# In[70]:

gm.fit(X, y)


# In[71]:

y_pred = gm.predict(X)


# In[72]:

y_pred


# In[73]:

from sklearn.metrics import normalized_mutual_info_score


# In[74]:

normalized_mutual_info_score(y, y_pred)


# In[75]:

from sklearn.datasets import load_iris


# In[77]:

X, y = load_iris(return_X_y=True)


# In[79]:

X, y


# In[80]:

iris_gm = GaussianMixture(n_components=3)


# In[95]:

iris_gm.fit(X)


# In[96]:

iris_gm.predict(X)


# In[97]:

normalized_mutual_info_score(y, iris_gm.predict(X))


# In[90]:

m = np.mean(X, axis=0)


# In[91]:

std = np.std(X, axis=0)


# In[93]:

X = (X - m) / std 


# In[98]:

np.mean(X, axis=0)


# In[101]:

from sklearn.decomposition import PCA


# In[115]:

pca = PCA()


# In[116]:

pca.fit(X)


# In[117]:

Y = pca.transform(X)


# In[118]:

Y


# In[119]:

pca.explained_variance_ratio_


# In[113]:

plt.scatter(Y[y==0,0], Y[y==0,1], c="r")
plt.scatter(Y[y==1,0], Y[y==1,1], c="g")
plt.scatter(Y[y==2,0], Y[y==2,1], c="b")


# In[114]:

plt.scatter(X[y==0,0], X[y==0,1], c="r")
plt.scatter(X[y==1,0], X[y==1,1], c="g")
plt.scatter(X[y==2,0], X[y==2,1], c="b")


# In[141]:

from sklearn.cluster import KMeans


# In[142]:

kmeans = KMeans(n_clusters=3)


# In[144]:

Y = kmeans.fit_predict(X)


# In[145]:

normalized_mutual_info_score(y, Y)


# In[146]:

Y


# In[147]:

import networkx as nx


# In[148]:

G = nx.karate_club_graph()


# In[150]:

degree = nx.degree(G)


# In[154]:

sorted(degree, key=degree.get)[::-1]


# In[153]:

degree


# In[164]:

X, y = make_s_curve(n_samples=1000)


# In[157]:

from sklearn.manifold import LocallyLinearEmbedding


# In[158]:

lle = LocallyLinearEmbedding()


# In[165]:

Y = lle.fit_transform(X)


# In[168]:

plt.scatter(Y[:,0], Y[:,1])


# In[169]:

X.shape


# In[170]:

from sklearn.datasets import load_digits


# In[196]:

X, y = load_digits(return_X_y=True)


# In[176]:

from sklearn.manifold import MDS


# In[177]:

mds = MDS()


# In[200]:

Y = mds.fit_transform(X)


# In[179]:

Y.shape


# In[180]:

y


# In[201]:

for i in range(10):
    plt.scatter(Y[y==i,0], Y[y==i,1], c=np.random.rand(3))


# In[198]:

X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)


# In[188]:

len(np.mean(X, axis=1))


# In[190]:

X.shape


# In[199]:

X[0]


# In[197]:

np.std(X, axis=0)


# In[6]:

####LaBNE
def LaBNE(network):
    
    # dictionary of node degrees
    degrees = nx.degree(network)
    
    # mean node degree
    m = np.mean(degrees.values()) / 2
    
    # estimated scaling exponent of the network 
    gamma = scaling_exponent(degrees)
    
    # beta 
    beta = 1 / (gamma - 1)
    
    # N
    N = len(network)
    
    # R
    R = 2 * np.log(N) - 2 * np.log( 2 * (1 - np.exp(- np.log(N) * (1 - beta))) / (np.pi * m * (1 - beta)) )
    
    # network laplacian
    L = np.array(nx.laplacian_matrix(network).todense())
    
    # eigen-decomposition
    l, U = np.linalg.eigh(L)
    
    # embedding
    Y = U[:,(1, 2)]
    
    # sort nodes decreasingly by degree
    sorted_nodes = np.array(sorted(degrees, key=degrees.get)[::-1])
    
    # argsort and label nodes
    i = sorted_nodes.argsort() + 1
    
    # radial co-ordinates
    r = 2 * beta * np.log(i) + 2 * (1 - beta) * np.log(N)
    
    # compute theta 
    theta = np.arctan(Y[:,1] / Y[:,0])
    
    # return (r, theta)
    return r, theta


# In[7]:

def scaling_exponent(degrees):
    
    ##sort nodes by degree
    degrees = np.array([sum(1 for v in degrees.values() if v == i) for i in range(max(degrees.values()))], 
                       dtype=np.float32)
    
    #probabilities
    probabilities = degrees / np.sum(degrees)
    
    return powerlaw.Fit(probabilities).power_law.alpha
    


# In[19]:

get_ipython().magic(u'matplotlib inline')

import networkx as nx
import numpy as np
import powerlaw
import matplotlib.pyplot as plt


# In[20]:

G = nx.karate_club_graph()


# In[21]:

r, theta = LaBNE(G)


# In[22]:

plt.polar(theta, r, ".", c="k")


# In[23]:

x = r * np.cos(theta)
y = r * np.sin(theta)


# In[26]:

plt.scatter(x, y)
for label, i, j in zip(G.nodes(), x, y):
#     if label not in H.nodes():
#         continue
    plt.annotate(
        label,
        xy=(i, j), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


# In[ ]:



