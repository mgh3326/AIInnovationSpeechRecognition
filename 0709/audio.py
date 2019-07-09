#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import numpy as np

# In[3]:


filename = "./voice.wav"

y, sr = librosa.load(filename)

# In[4]:


# get_ipython().system('ffprobe -i voice.wav')


# In[41]:


16000 * 16

# In[38]:


# get_ipython().system('file "./voice.wav"')


# In[5]:


print(y)
print(y.shape)
print(sr)

# In[6]:


import IPython.display as ipd

ipd.Audio(filename)  # load a local WAV file

# In[7]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)

# In[8]:


import numpy

sr = 22050  # sample rate
T = 2.0  # seconds
t = numpy.linspace(0, T, int(T * sr), endpoint=False)  # time variable
x = 0.5 * numpy.sin(2 * numpy.pi * 440 * t)  # pure sine wave at 440 Hz

ipd.Audio(x, rate=sr)  # load a NumPy array

# In[9]:


D = librosa.stft(y)
print(D)

D_mag = np.abs(D)
print(D_mag)
print(D_mag.shape)

magnitude, phase = librosa.magphase(D)

print(magnitude)
print(magnitude.shape)

print(magnitude - D_mag)

# In[10]:


import matplotlib.pyplot as plt

librosa.display.specshow(librosa.amplitude_to_db(magnitude,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# In[11]:


mel_s = librosa.feature.melspectrogram(y=y, sr=sr)
print(mel_s.shape)

import matplotlib.pyplot as plt

librosa.display.specshow(librosa.amplitude_to_db(mel_s,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# In[12]:


67451 / 512

# In[13]:


mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

print(mfccs.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# In[14]:


import matplotlib.pyplot as plt

librosa.display.specshow(librosa.amplitude_to_db(mfccs,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# In[ ]:
