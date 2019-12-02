from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
x_train = tf.reshape(x_train, shape=(-1, 784))
x_test = tf.reshape(x_test, shape=(-1, 784))
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

weights = tf.Variable(tf.random.normal(shape=(784, 10), dtype=tf.float32))
bias = tf.Variable(tf.random.normal(shape=(10,), dtype=tf.float32))

def logistic_regression(x):
    lr = tf.add(tf.matmul(x, weights), bias)
    return lr

def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, 10)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    preds = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
    preds = tf.equal(y_true, preds)
    return tf.reduce_mean(tf.cast(preds))

def gradient_descent(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss_val = cross_entropy(y, y_pred)
    return tape.gradient(loss_val, [weights, bias])

n_batches = 10000
learning_rate = 0.01
batch_size = 32

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(x_train.shape[0]).batch(batch_size)

optimizer = tf.train.AdamOptimizer(learning_rate)
for batch_num, (batch_xs, batch_ys) in enumerate(dataset.take(n_batches), 1):
    gradients = gradient_descent(batch_xs, batch_ys)
    optimizer.apply_gradients(zip(gradients, weights, bias))
    y_pred = logistic_regression(batch_xs)
    loss = cross_entropy(batch_ys, y_pred)
    acc = accuracy(batch_ys, y_pred)
    print("Batch number: {}, loss: {}, accuracy: {}".format(batch_num, loss, acc))

