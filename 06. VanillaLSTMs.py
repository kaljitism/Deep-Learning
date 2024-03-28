#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


# Training Data
x = []
for i in range(100):
    x.append([[i+j] for j in range(5)])


# In[3]:


x[:10]


# In[4]:


# Testing Data
y = []
for i in range(100):
    y.append(i+5)


# In[5]:


y[:10]


# In[6]:


# Converting into arrays
x, y = np.array(x), np.array(y)


# In[7]:


# Normalization
x, y = x/100, y/100


# In[8]:


x.shape, y.shape


# In[9]:


# Train Test Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[10]:


# Modelling
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(1, batch_input_shape=(None, 5, 1), return_sequences=True),
    tf.keras.layers.LSTM(1, return_sequences=False)
])


# In[11]:


model.compile(loss='mse',
             optimizer='adam',
             metrics=['accuracy'])


# In[12]:


model.summary()


# In[13]:


# Trining
history = model.fit(x_train, y_train, epochs=400, validation_data=(x_test, y_test))


# In[14]:


predictions = model.predict(x_test)


# In[15]:


predictions[:10]


# In[16]:


y_test[:10]


# In[17]:


# Prediction
plt.scatter(range(len(y_test)), predictions, c='r')
plt.scatter(range(len(y_test)), y_test, c='g')
plt.show()


# In[18]:


# Optimization
plt.plot(history.history['loss'])
plt.title('Loss')
plt.show()

