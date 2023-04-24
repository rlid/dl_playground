import tensorflow as tf
import numpy as np

num_samples = 1000

negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],
         [0.5, 1]],
    size=num_samples
)

positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],
         [0.5, 1]],
    size=num_samples
)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((
    np.zeros((num_samples, 1), dtype=np.float32),
    np.ones((num_samples, 1), dtype=np.float32)
))

import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:, 0])
plt.show()

input_dim = 2
output_dim = 1

W = tf.Variable(tf.random.uniform((input_dim, output_dim)))
b = tf.Variable(tf.random.normal((output_dim,)))
learning_rate = 1e-1


def model(inputs):
    return tf.matmul(inputs, W) + b


def one_step(inputs):
    with tf.GradientTape() as tape:
        predicts = model(inputs)
        loss = tf.reduce_mean(tf.square(predicts - labels))
    dW, db = tape.gradient(loss, [W, b])
    W.assign_sub(dW * learning_rate)
    b.assign_sub(db * learning_rate)
    return loss


num_epoches = 50

for i in range(num_epoches):
    print(one_step(inputs))

predicts = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:, 0] > 0.5)
x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, '-r')
plt.show()
