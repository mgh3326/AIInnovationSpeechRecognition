# coding: utf-8

# In[2]:


import os
import os.path

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from torchvision.datasets.utils import makedir_exist_ok
# import torchaudio

import pandas as pd
import numpy as np

import librosa

import random

# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# In[4]:


# parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 64

# In[5]:


a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

c = np.hstack([a, b])
print(c)


# In[6]:


class UrbanSound8K(Dataset):
  training_file = 'training.pt'
  test_file = 'test.pt'
  classes = [
    (0, 'air_conditioner'),
    (1, 'car_horn'),
    (2, 'children_playing'),
    (3, 'dog_bark'),
    (4, 'drilling'),
    (5, 'engine_idling'),
    (6, 'gun_shot'),
    (7, 'jackhammer'),
    (8, 'siren'),
    (9, 'street_music')
  ]

  def __init__(self, root, train=True, fold_list=None,
               transform=None, target_transform=None, require_preprocess=True):

    super(UrbanSound8K, self).__init__()
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train

    if require_preprocess:
      self.preprocess(fold_list)

    if self.train:
      data_file = self.training_file
    else:
      data_file = self.test_file

    if self._check_exists(os.path.join(self.processed_folder, data_file)):
      self._read_metadata(fold_list)
      self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
    else:
      raise RuntimeError('Dataset not found. (%s)' %
                         os.path.join(self.processed_folder, data_file))

  def __getitem__(self, index):

    sound, target = self.data[index], int(self.targets[index])

    if self.transform is not None:
      sound = self.transform(sound)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return sound, target

  def __len__(self):
    return len(self.data)

  @property
  def audio_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'audio')

  @property
  def metadata_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'metadata')

  @property
  def processed_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'processed')

  @property
  def class_to_idx(self):
    return {_class: i for i, (cid, _class) in enumerate(self.classes)}

  @property
  def idx_to_class(self):
    return {i: _class for i, (cid, _class) in enumerate(self.classes)}

  def _check_exists(self, file):
    return os.path.exists(file)

  def _read_metadata(self, fold_list):
    csv_path = os.path.join(self.metadata_folder, self.__class__.__name__ + ".csv")
    print(csv_path)

    self.metadata = {}
    file_names = []
    labels = []
    folders = []

    if os.path.exists(csv_path):
      csvData = pd.read_csv(csv_path)
      # print(csvData)

      for i in range(0, len(csvData)):
        if csvData.iloc[i, 5] in fold_list:
          file_names.append(csvData.iloc[i, 0])
          labels.append(csvData.iloc[i, 6])
          folders.append(csvData.iloc[i, 5])

      self.metadata['file_names'] = file_names
      self.metadata['labels'] = labels
      self.metadata['folders'] = folders
    else:
      raise RuntimeError('Metadata(csv format) not found.')

  def preprocess(self, fold_list):

    makedir_exist_ok(self.audio_folder)
    makedir_exist_ok(self.processed_folder)

    self._read_metadata(fold_list)

    # pre-process
    file_names = self.metadata['file_names']
    labels = self.metadata['labels']
    folders = self.metadata['folders']

    data = []
    targets = []

    start = time.time()
    for idx, (file_name, label, folder) in enumerate(zip(file_names, labels, folders)):
      wav_file_path = os.path.join(self.audio_folder,
                                   "fold{}".format(folder),
                                   file_name)

      sound, sr = librosa.load(wav_file_path, mono=True, res_type='kaiser_fast')
      # print("sound : ", sound, sr)
      #             # 4초 임시데이터 생성
      #             tempSound = torch.zeros(4*8000) # 4sec. * 8KHz
      #             if len(sound) < 4*8000:
      #                 tempSound[:len(sound)] = torch.FloatTensor(sound[:])
      #             else:
      #                 tempSound[:] = torch.FloatTensor(sound[:4*8000])

      #             sound = tempSound
      target = label

      #             frame_length = 0.025                        # 25(ms)
      #             frame_stride = 0.010                        # 10(ms)
      #             n_fft = int(round(sr*frame_length))        # 200 (sample)
      #             hop_length = int(round(sr*frame_stride))    # 80 (sample)

      X, sample_rate = sound, sr

      # stft = np.abs(librosa.stft(X))
      # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
      # mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
      # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
      # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

      melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=40)
      # print("mel: ", melspec)
      # print("mel_size: ", melspec.shape)

      # logspec = np.log(melspec)
      logspec = librosa.power_to_db(melspec, ref=np.max)
      log_mel = np.mean(logspec.T, axis=0)

      # S = np.hstack([mfccs, chroma, mel, contrast])
      S = np.hstack([log_mel])

      # S = np.mean(librosa.feature.melspectrogram(y=sound, n_mels=40, n_fft=n_fft, hop_length=hop_length).T, axis=0)
      # mfccs = np.mean(librosa.feature.mfcc(y=sound.numpy(), sr=sr, n_mfcc=40).T, axis=0)

      # print(S.shape, S)
      # print(mfccs.shape, mfccs)
      # break

      S = torch.FloatTensor(S)

      data.append(S)
      targets.append(target)

      end = time.time()
      if idx % 100 == 0:
        print("(%s) %04d/%04d processed. (%.4f (sec.))" % (
          "train" if self.train else "test", idx + 1, len(file_names), (end - start)))

    print("len(data)", len(data))
    print("data[:10] :", data[:10])

    print("torch.stack(data).shape: ", torch.stack(data).shape)
    print("torch.stack(data): ", torch.stack(data))

    if self.train:
      training_set = (
        torch.stack(data),
        torch.LongTensor(targets)
      )

      with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
        torch.save(training_set, f)
    else:
      test_set = (
        torch.stack(data),
        torch.LongTensor(targets)
      )

      with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
        torch.save(test_set, f)

    print('Done!')


# In[11]:


# urbansound8k dataset
start = time.time()
urbansound8k_train = UrbanSound8K(root='../../',
                                  train=True,
                                  fold_list=[1],
                                  require_preprocess=True)
end = time.time()
print("Prepare (Train Data Set): %.4f (sec.)" % (end - start))

# urbansound8k dataset
start = time.time()
urbansound8k_test = UrbanSound8K(root='../../',
                                 train=False,
                                 fold_list=[10],
                                 require_preprocess=True)
end = time.time()
print("Prepare (Test Data Set): %.4f (sec.)" % (end - start))

train_loader = torch.utils.data.DataLoader(dataset=urbansound8k_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

# In[12]:


# Linear modeling
linear1 = torch.nn.Linear(40, 256, bias=True)  # 193
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 256, bias=True)
linear4 = torch.nn.Linear(256, 256, bias=True)
linear5 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

drop_prob = 0.5
dropout = torch.nn.Dropout(p=drop_prob)

bn1 = torch.nn.BatchNorm1d(256)
bn2 = torch.nn.BatchNorm1d(256)
bn3 = torch.nn.BatchNorm1d(256)
bn4 = torch.nn.BatchNorm1d(256)

# In[14]:


# Initialization
# torch.nn.init.normal_(linear.weight)

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)

# In[15]:


# model
model = torch.nn.Sequential(linear1, bn1, relu, dropout,
                            linear2, bn2, relu, dropout,
                            linear3, bn3, relu, dropout,
                            linear4, bn4, relu, dropout,
                            linear5).to(device)

# In[16]:


# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

# In[17]:


# model.reset_parameters()

training_epochs = 10

total_batch = len(train_loader)
print(total_batch)

model.train()  # set the model to train mode (dropout=True)
for epoch in range(training_epochs):
  avg_cost = 0

  costs = []
  for iter_id, (X, Y) in enumerate(train_loader):
    start = time.time()

    X = X.view(-1, 40).to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch

    costs.append(cost.item())
    mean_costs = np.mean(np.array(costs))

    end = time.time()
    e_time_sec = (end - start)
  #         print('Epoch:', '%04d' % (epoch + 1),
  #               'Iter:', '%03d/%03d' % (iter_id + 1, total_batch),
  #               'cost =', '{:.9f}'.format(mean_costs),
  #               'Time:', '%.4f (sec.)' % e_time_sec)

  print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

  with torch.no_grad():
    model.eval()  # set the model to evaluation mode (dropout=False)

    X_train = urbansound8k_train.data.view(-1, 40).to(device)
    Y_train = urbansound8k_train.targets.to(device)

    prediction = model(X_train)

    correct_prediction = torch.argmax(prediction, 1) == Y_train
    accuracy = correct_prediction.float().mean()
    print('(Train) Accuracy:', accuracy.item())

    ############################################

    X_test = urbansound8k_test.data.view(-1, 40).to(device)
    Y_test = urbansound8k_test.targets.to(device)

    prediction = model(X_test)

    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('(Test ) Accuracy:', accuracy.item())

print('Learning finished')

# In[19]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if not title:
    if normalize:
      title = 'Normalized confusion matrix'
    else:
      title = 'Confusion matrix, without normalization'

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  # Only use the labels that appear in the data
  print(type(unique_labels(y_true, y_pred)), unique_labels(y_true, y_pred))
  classes = classes[unique_labels(y_true, y_pred)]
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title=title,
         ylabel='True label',
         xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, format(cm[i, j], fmt),
              ha="center", va="center",
              color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax


np.set_printoptions(precision=2)

# In[21]:


# In[24]:


# Test the model using test sets
with torch.no_grad():
  model.eval()  # set the model to evaluation mode (dropout=False)

  X_test = urbansound8k_test.data.view(-1, 40).to(device)
  Y_test = urbansound8k_test.targets.to(device)

  prediction = model(X_test)

  correct_prediction = torch.argmax(prediction, 1) == Y_test
  accuracy = correct_prediction.float().mean()
  print('Accuracy:', accuracy.item())

  # Get one and predict
  r = random.randint(0, len(urbansound8k_test) - 1)
  print('Random One: ', r)
  X_single_data = urbansound8k_test.data[r:r + 1].view(-1, 40).to(device)
  Y_single_data = urbansound8k_test.targets[r:r + 1].to(device)

  print('Label: ', Y_single_data.item(),
        urbansound8k_test.idx_to_class[Y_single_data.item()])

  single_prediction = model(X_single_data)
  print('Prediction: ', torch.argmax(single_prediction, 1).item(),
        urbansound8k_test.idx_to_class[torch.argmax(single_prediction, 1).item()])

  folder = urbansound8k_test.metadata['folders'][r]
  file_name = urbansound8k_test.metadata['file_names'][r]
  wav_file_path = os.path.join(urbansound8k_test.audio_folder,
                               "fold{}".format(folder),
                               file_name)
  print(wav_file_path)

  # import IPython.display as ipd
  # ipd.Audio(wav_file_path) # load a local WAV file

  y_pred = torch.argmax(prediction, 1).cpu().numpy()
  y_true = Y_test.cpu().numpy()

  print(y_pred.shape, type(y_pred), y_pred[0], type(y_pred[0]))
  print(y_true.shape, type(y_true), y_true[0], type(y_true[0]))

  labels = [cid for cid, _class in urbansound8k_test.classes]
  print(labels)

  from sklearn.metrics import confusion_matrix

  print(confusion_matrix(y_true, y_pred, labels=labels))

  # Plot non-normalized confusion matrix
  class_names = np.array([_class for cid, _class in urbansound8k_test.classes])
  plot_confusion_matrix(y_true, y_pred, classes=class_names,
                        title='Confusion matrix, without normalization')

print(wav_file_path)

# In[22]:


# In[23]:


# In[27]:


import librosa
import time

file_name = "../../UrbanSound8K/audio/fold1/119455-5-0-0.wav"

audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
print(audio.shape, sample_rate)

mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

print(mfccs.shape, mfccs)

mfccsscaled = np.mean(mfccs.T, axis=0)

print(mfccsscaled.shape, mfccsscaled)
