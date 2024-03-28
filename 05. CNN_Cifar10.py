#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


# In[3]:


class_names = ['airplane', 
               'automobile', 
               'bird', 
               'cat', 
               'dear', 
               'dog', 
               'frog', 
               'horse', 
               'ship', 
               'truck']


# In[4]:


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[5]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[6]:


model.summary()


# In[7]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[8]:


history = model.fit(train_images, train_labels, epochs=10,
                   validation_data=(test_images, test_labels))


# In[9]:


plt.plot(history.history['acc'], label='Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower_right')
plt.show()


# In[10]:


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


# In[11]:


print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)

