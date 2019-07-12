#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np


# In[3]:


filename = "./audio/1.mp3"

y, sr = librosa.load(filename, sr=44100, mono=False)


# In[4]:


print(type(y), y.shape)


# In[5]:


print(y[0])


# In[6]:


print(y[1])


# In[7]:


check = (y[0] == y[1])


# In[8]:


for i, b in enumerate(check):
    if b == False:
        print(i, y[0][i], y[1][i])


# In[20]:


y_mono, sr_mono = librosa.load(filename, sr=44100, mono=True)

y, sr = librosa.load(filename, sr=44100, mono=False)
y_mono_maked = np.mean(y, axis=0)


# In[21]:


print(y_mono)


# In[22]:


print(y_mono_maked)


# In[24]:


np.all(y_mono == y_mono_maked)


# In[ ]:




