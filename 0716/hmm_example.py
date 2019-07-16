#!/usr/bin/env python
# coding: utf-8

# In[1]:

import hmm

# In[15]:


states = ('hot', 'cold')
symbols = ('1', '2', '3')

start_prob = {
    'hot': 0.8,
    'cold': 0.2
}

trans_prob = {
    'hot': {'hot': 0.6, 'cold': 0.4},
    'cold': {'hot': 0.4, 'cold': 0.6}
}

emit_prob = {
    'hot': {'1': 0.2, '2': 0.4, '3': 0.4},
    'cold': {'1': 0.5, '2': 0.4, '3': 0.1}
}

# In[16]:


model = hmm.Model(states, symbols, start_prob, trans_prob, emit_prob)
sequence = ['3', '1', '3']
# sequence = ['2', '2', '2']

# model._start_prob['hot'] = 0.5
# model._start_prob['cold'] = 0.5

print(model.evaluate(sequence))  # Likelihood 계산
print(model.decode(sequence))  # 최적상태열 추정

# In[36]:


print("=" * 70)
print("(TRAIN )")

# sequences = [
#     (['hot', 'cold', 'hot'], ['3', '1', '3']),
#     (['cold', 'cold', 'cold'], ['1', '1', '1']),
#     (['cold', 'hot', 'hot'], ['1', '2', '2'])
# ]

sequences = [
    (['hot', 'cold', 'hot'], ['3', '1', '3']),
    (['hot', 'hot', 'hot'], ['3', '3', '3']),
    (['cold', 'cold', 'cold'], ['1', '1', '1']),
    (['cold', 'hot', 'hot'], ['1', '3', '3'])
]

model = hmm.train(sequences, delta=1e-10, smoothing=1e-10)

print("=" * 70)
print("(PREDICT)")

# In[37]:


# model._start_prob['hot'] = 0.5
# model._start_prob['cold'] = 0.5


print("=" * 70)
sequence = ['3', '1', '3']
print(model.evaluate(sequence))  # Likelihood 계산
print(model.decode(sequence))  # 최적상태열 추정

print("=" * 70)
sequence = ['1', '1', '1']
print(model.evaluate(sequence))  # Likelihood 계산
print(model.decode(sequence))  # 최적상태열 추정

print("=" * 70)
print(model.start_prob('hot'))
print(model.start_prob('cold'))

# In[ ]:


# In[ ]:
