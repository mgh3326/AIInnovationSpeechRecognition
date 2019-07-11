#!/usr/bin/env python
# coding: utf-8

# In[4]:


import librosa
import librosa.display
import numpy as np

import matplotlib.pyplot as plt


# In[5]:


filename_wav = "./audio/p257_143__clean.wav"
filename_noise = "./audio/p257_143__background.wav"
filename_mix = "./audio/p257_143__mixed_conditioned0_2dB.wav"


# In[6]:




# In[7]:




# In[8]:




# In[9]:




# In[10]:




# In[11]:


y, sr = librosa.load(filename_wav, sr=16000)

D = librosa.stft(y, sr)
magnitude, phase = librosa.magphase(D)

librosa.display.specshow(librosa.amplitude_to_db(magnitude,
                                                  ref=np.max),
                          y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


# In[12]:


y, sr = librosa.load(filename_noise, sr=16000)

D = librosa.stft(y, sr)
magnitude, phase = librosa.magphase(D)

librosa.display.specshow(librosa.amplitude_to_db(magnitude,
                                                  ref=np.max),
                          y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


# In[13]:


y, sr = librosa.load(filename_mix, sr=16000)

D = librosa.stft(y, sr)
magnitude, phase = librosa.magphase(D)

librosa.display.specshow(librosa.amplitude_to_db(magnitude,
                                                  ref=np.max),
                          y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


# In[14]:


y, sr = librosa.load(filename_wav, sr=16000)

power_wav = np.mean(np.power(y, 2))
print(power_wav, y.shape)

y, sr = librosa.load(filename_noise, sr=16000)

power_noise = np.mean(np.power(y, 2))
print(power_noise, y.shape)


# In[15]:


dB = 10 * (np.log10(power_wav) - np.log10(power_noise))
print(dB)


# In[16]:


y_mix, sr_mix = librosa.load(filename_mix, sr=16000)
y_noise, sr_noise = librosa.load(filename_noise, sr=16000)

estimated_wav = y_mix - y_noise

librosa.output.write_wav("./estimated.wav", estimated_wav, sr=sr_mix)


# In[17]:


y_wav, sr_wav = librosa.load(filename_wav, sr=16000)
y_est, sr_est = librosa.load("./estimated.wav", sr=16000)

print(y_wav - y_est)

for i in (y_wav - y_est):
    if i != 0.:
        print(i)


# In[18]:




# In[19]:


y_est, sr_est = librosa.load("./estimated.wav", sr=16000)

D = librosa.stft(y_est, sr_est)
magnitude, phase = librosa.magphase(D)

librosa.display.specshow(librosa.amplitude_to_db(magnitude,
                                                  ref=np.max),
                          y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


# In[ ]:




