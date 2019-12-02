# Dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data Prep
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = x * 2 + 1

# Initialize the w and b
w = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# Prediction
pred = tf.add(tf.multiply(x, w), b)

# Cost
mse = tf.reduce_mean(tf.square(y - pred))

# Optimization
gd = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Initializing Variable
init = tf.global_variables_initializer()

# Epochs
epochs = 10

loss_record = []

# Session
with tf.Session() as sess:
    sess.run(init)
    # Training - Minimizing the cost function
    for i in range(epochs):
        sess.run(gd.minimize(loss=mse))
        loss_record.append(sess.run(mse))
        print("Loss for Epoch {} is: {}".format(i, sess.run(mse)))
    print("\nWeights and Bias = ", sess.run(w), sess.run(b))


plt.plot(range(epochs), loss_record)
plt.show()

