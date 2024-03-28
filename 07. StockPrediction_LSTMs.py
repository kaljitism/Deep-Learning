#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[2]:


data = pd.read_csv('GOOG.csv')
data.head()


# In[3]:


x = data.iloc[:, 1:2].values


# In[4]:


sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)


# In[5]:


x[:10]


# In[6]:


len(x)


# In[7]:


x_train = []
y_train = []
for i in range(25, 1259):
    x_train.append(x[i-25:i, 0])
    y_train.append(x[i, 0])


# In[8]:


x_train[0]


# In[9]:


y_train[0]


# In[10]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[11]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[12]:


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


# In[13]:


model.compile(optimizer='adam',
             loss='mean_squared_error')


# In[14]:


history = model.fit(x_train, y_train, epochs=10, batch_size=128)


# In[15]:


predictions = model.predict(x_train)


# In[16]:


plt.plot(range(len(x_train)), y_train, c='g')
plt.plot(range(len(x_train)), predictions, c='r')


# In[ ]:




