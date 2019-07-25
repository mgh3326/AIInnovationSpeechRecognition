#!/usr/bin/env python
# coding: utf-8

# In[174]:


text="""A barber is a person. 
a barber is good person. 
a barber is huge person. 
he Knew A Secret! 
The Secret He Kept is huge secret. 
Huge secret. 
His barber kept his word. 
a barber kept his word. 
His barber kept his secret. 
But keeping and keeping such a huge secret to himself was driving the barber crazy. 
the barber went up a huge mountain."""


# In[175]:


# 목표 1: Word Seq --> Word-ID Seq

## 세부목표 1: 문장을 토큰화
## 세부목표 2: 단어 Set 및 아이디 부여
## 세부목표 3: Word Seq --> Word-ID Seq


# In[176]:


print(text.replace(".", " .").replace("!", " !"))


# In[177]:


token = text.replace(".", " .").split('\n')
print(token)


# In[178]:


text_ = text.replace(".", " .").replace("!", " !")
text_ = text_.strip()


# In[179]:


text_sent = text_.lower().split('\n')
# print(text_sent)


# In[180]:


sent = []

for s in text_sent:
    word_seq = s.split()
    sent.append(word_seq)
    # print(word_seq)

print(sent)


# In[181]:


count_dict = {}

for s in sent:
    for w in s:
        if w not in count_dict:
            count_dict[w] = 1
        else:
            count_dict[w] += 1

print(count_dict)


# In[182]:


print(count_dict.items())


# vocab_sorted=sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
# print(vocab_sorted)

# In[183]:


vocab_sorted=sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)


# In[184]:


vocab_dict = {}

for idx, (word, word_count) in enumerate(vocab_sorted):
    # print(word, word_count)
    vocab_dict[word] = idx+1

print(vocab_dict)


# In[185]:


a = [0]*27
a[12] = 1
print(a)


# In[186]:


b = [0]*27
b[vocab_dict['keeping']] = 1
print(b)


# In[187]:


a == b


# In[188]:


sent_idx = []

for s in sent:
    s_id = []
    for w in s:
        w_id = vocab_dict[w]
        s_id.append(w_id)
        print("s_id :", w, w_id, s_id)
    sent_idx.append(s_id)
    print()

# print(sent_idx)
    


# In[189]:


for s, s_idx in zip(sent, sent_idx):
    print(s)
    print(s_idx)
    print()


# In[190]:


lengths = []
for s in sent_idx:
    len_sent = len(s)
    lengths.append(len_sent)
    
print(lengths)
    


# In[191]:


max_length = max(lengths)
print(max_length)


# In[192]:


max_length = 0
for s in sent_idx:
    len_sent = len(s)
    if len_sent > max_length:
        max_length = len_sent
    
print(max_length)


# In[193]:


sent_idx_zeropadded = []

for s in sent_idx:
    # print(s)
    temp = [0]*max_length
    for idx, w_id in enumerate(s):
        temp[idx] = s[idx]
    # print(temp)
    sent_idx_zeropadded.append(temp)
    # break

print(sent_idx_zeropadded)
        


# In[194]:


import numpy as np

data_X = np.asarray(sent_idx_zeropadded, dtype=np.float32)


# In[195]:


print(data_X.shape)
print(type(data_X[0]))
print(data_X[0])


# In[196]:


print(data_X)
data_Y= np.roll(data_X, shift=-1, axis=1)
for row in range(0, len(data_Y)):
    # print(row)
    data_Y[row, len(data_Y[row])-1] = 0 
print(data_Y)


# In[197]:


from keras.utils import to_categorical

print(vocab_dict)
num_classes = len(vocab_dict) + 1
print(num_classes)
print(data_X)
data_XX = to_categorical(data_X, num_classes)


# In[198]:


print(data_XX)


# In[199]:


data_XX = to_categorical(data_X, num_classes)
data_YY = to_categorical(data_Y, num_classes)


# In[200]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop

hidden_units = 100
batch_size = 2
epochs = 100
learning_rate = 0.01

x_train = data_XX
y_train = data_YY

x_test = data_XX
y_test = data_YY

print("x_train.shape[1:] : ", x_train.shape[1:])

model = Sequential()
model.add(SimpleRNN(hidden_units,
                    return_sequences=True,
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)

print('SimpleRNN test loss    : ', scores[0])
print('SimpleRNN test accuracy: ', scores[1])


# In[201]:


pred_x_test = model.predict(x_test)

print(pred_x_test[0][0])
print(np.argmax(pred_x_test[0][0]))

print(y_test[0][0])
print(np.argmax(y_test[0][0]))

print("PRED:")
print(np.argmax(pred_x_test, axis=-1))

print("REF :")
print(np.argmax(y_test, axis=-1))


# In[202]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras import initializers
from keras.optimizers import RMSprop

hidden_units = 100
batch_size = 2
epochs = 100
learning_rate = 0.01

x_train = data_XX
y_train = data_YY

x_test = data_XX
y_test = data_YY

print("x_train.shape[1:] : ", x_train.shape[1:])

model = Sequential()
model.add(LSTM(hidden_units,
               return_sequences=True,
               input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)

print('LSTM test loss    : ', scores[0])
print('LSTM test accuracy: ', scores[1])


# In[203]:


pred_x_test = model.predict(x_test)

print(pred_x_test[0][0])
print(np.argmax(pred_x_test[0][0]))

print(y_test[0][0])
print(np.argmax(y_test[0][0]))

print("PRED:")
print(np.argmax(pred_x_test, axis=-1))

print("REF :")
print(np.argmax(y_test, axis=-1))


# In[204]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding
from keras import initializers
from keras.optimizers import RMSprop

word_embedding_size = 100
vocab_size = len(vocab_dict) + 1

hidden_units = 100
batch_size = 2
epochs = 100
learning_rate = 0.01

x_train = data_X
y_train = data_YY

x_test = data_X
y_test = data_YY

print("x_train.shape[1:] : ", x_train.shape[1:])

model = Sequential()
model.add(Embedding(vocab_size, word_embedding_size, input_length=x_train.shape[1]))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

model.add(LSTM(hidden_units,
               return_sequences=True,
               input_shape=x_train.shape[1:]))
model.add(LSTM(hidden_units,
               return_sequences=True))
               # input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)

print('LSTM test loss    : ', scores[0])
print('LSTM test accuracy: ', scores[1])

model.summary()


# In[205]:


pred_x_test = model.predict(x_test)

print(pred_x_test[0][0])
print(np.argmax(pred_x_test[0][0]))

print(y_test[0][0])
print(np.argmax(y_test[0][0]))

print("PRED:")
print(np.argmax(pred_x_test, axis=-1))

print("REF :")
print(np.argmax(y_test, axis=-1))


# In[206]:


print(model.layers[0])
emebdding_layer = model.layers[0]


# In[207]:


embeddings = model.layers[0].get_weights()[0]

words_embeddings = {w:embeddings[idx] for w, idx in vocab_dict.items()}
print(words_embeddings)

print(words_embeddings['barber'])


# In[208]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding, Bidirectional
from keras import initializers
from keras.optimizers import RMSprop

word_embedding_size = 100
vocab_size = len(vocab_dict) + 1

hidden_units = 100
batch_size = 2
epochs = 100
learning_rate = 0.01

x_train = data_X
y_train = data_YY

x_test = data_X
y_test = data_YY

print("x_train.shape[1:] : ", x_train.shape[1:])

model = Sequential()
model.add(Embedding(vocab_size, word_embedding_size, input_length=x_train.shape[1]))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

model.add(Bidirectional(LSTM(hidden_units,
               return_sequences=True,
               input_shape=x_train.shape[1:])))

model.add(Bidirectional(LSTM(hidden_units,
               return_sequences=True)))
               # input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)

print('LSTM test loss    : ', scores[0])
print('LSTM test accuracy: ', scores[1])

model.summary()


# In[209]:


pred_x_test = model.predict(x_test)

print(pred_x_test[0][0])
print(np.argmax(pred_x_test[0][0]))

print(y_test[0][0])
print(np.argmax(y_test[0][0]))

print("PRED:")
print(np.argmax(pred_x_test, axis=-1))

print("REF :")
print(np.argmax(y_test, axis=-1))


# In[ ]:




