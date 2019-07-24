#!/usr/bin/env python
# coding: utf-8

# In[19]:


text = """A barber is a person. 
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

# In[20]:


# 목표 1: Word Seq --> Word-ID Seq

## 세부목표 1: 문장을 토큰화
## 세부목표 2: 단어 Set 및 아이디 부여
## 세부목표 3: Word Seq --> Word-ID Seq


# In[21]:


## 세부 목표 1:
text_token = text.split()
print(text_token)

# In[12]:


import nltk

nltk.download('stopwords')
nltk.download('punkt')

# In[25]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

sent_text = sent_tokenize(text)
print("sent_text:", sent_text)

vocab = Counter()  # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.

sentences = []

for i in sent_text:
  sentence = word_tokenize(i)  # 단어 토큰화를 수행합니다.
  print("sentence : ", sentence)
  result = []

  for word in sentence:
    word = word.lower()  # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.

    result.append(word)
    # vocab[word] = vocab[word] + 1 #각 단어의 빈도를 Count 합니다.
    vocab[word] += 1
  sentences.append(result)

print("sentences: ", sentences)
print("vocab: ", vocab)

# In[26]:


vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
print("vocab_sorted: ", vocab_sorted)

# In[27]:


word_to_index = {}

for idx, (word, frequency) in enumerate(vocab_sorted):
  word_to_index[word] = idx

print(word_to_index)

# In[28]:


sentence_id = []

for idx, sent in enumerate(sentences):
  sent_id = []
  for word in sent:
    word_id = word_to_index[word]
    sent_id.append(word_id)
  print("sent   : ", sent)
  print("sent_id: ", sent_id)

# In[ ]:
