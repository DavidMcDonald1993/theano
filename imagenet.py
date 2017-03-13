
# coding: utf-8

# In[1]:

##mapping
mapping = {}
for line in open("mapping.txt", 'rb'):
    l = line.rstrip().split("\t")
    mapping.update({l[0]: l[1]})


# In[4]:

import urllib, cStringIO
from PIL import Image
import gzip
import numpy as np
import matplotlib.pyplot as plt

d = 128 

size = d, d, 3

i = 0
for line in gzip.open('imagenet.tgz', 'rb'):
    i += 1
    if i % 10000 != 0:
        continue
    if i != 1:
        l = line.rstrip().split('\t')
        str = l[0]
        m = str.split("_")
        str = mapping[m[0]]
        url = l[1]
        try:        
            file = cStringIO.StringIO(urllib.urlopen(url).read())
            img = Image.open(file)
            img.thumbnail(size, Image.ANTIALIAS)
#             img.show()
            xpad = (size[0] - img.size[0]) / 2
            ypad = (size[1] - img.size[1]) / 2
            padImg = np.pad(np.array(img), [(ypad, ypad), (xpad, xpad), (0, 0)], mode='constant')
            A = np.zeros(size)
            A[:padImg.shape[0], :padImg.shape[1], :padImg.shape[2]] = padImg
            print '{} {} {}'.format(i, str, A.shape)
        except Exception:
            print 'Exception'
            pass
print i


# In[48]:

A = np.array([[1,2],[3,4], [3,4]])
print A.shape[1]
np.pad(A, [(2, 2), (1,1)], mode='constant')

