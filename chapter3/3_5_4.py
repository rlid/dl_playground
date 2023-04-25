import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

num_neg_samples = 1000
num_pos_samples = 100

negative_samples = np.random.multivariate_normal(
    mean=[0, 5],
    cov=[[10, 5],
         [5, 10]],
    size=num_neg_samples
)

positive_samples = np.random.multivariate_normal(
    mean=[2, 0],
    cov=[[1, 0.5],
         [0.5, 1]],
    size=num_pos_samples
)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((
    np.zeros((num_neg_samples, 1), dtype=np.float32),
    np.ones((num_pos_samples, 1), dtype=np.float32)
))


# plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:, 0])
# plt.show()

input_dim = 2
output_dim = 1


class ModelLinear:
    def __init__(self, input_dim, output_dim):
        self.W = tf.Variable(tf.random.uniform((input_dim, output_dim)))
        self.b = tf.Variable(tf.random.uniform((output_dim,)))
        # self.W = tf.Variable(tf.ones((input_dim, output_dim)))
        # self.b = tf.Variable(tf.ones((output_dim,)))

    def __call__(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

    @property
    def weights(self):
        return [self.W, self.b]


class ModelLogistic(ModelLinear):
    def __call__(self, inputs):
        return tf.math.sigmoid(super(ModelLogistic, self).__call__(inputs))


learning_rate = 1e-2


model1 = ModelLinear(input_dim, output_dim)
model2 = ModelLogistic(input_dim, output_dim)


def one_step(inputs):
    with tf.GradientTape() as tape1:
        predicts1 = model1(inputs)
        loss1 = tf.reduce_mean(tf.square(predicts1 - labels))
    d_weights1 = tape1.gradient(loss1, model1.weights)

    with tf.GradientTape() as tape2:
        predicts2 = model2(inputs)
        loss2 = -tf.reduce_mean(
            tf.math.log(predicts2) * labels +
            tf.math.log(1 - predicts2) * (1 - labels)
        )
    d_weights2 = tape2.gradient(loss2, model2.weights)

    for (w, dw) in zip(model1.weights + model2.weights, d_weights1 + d_weights2):
        w.assign_sub(dw * learning_rate)
    return np.sum((predicts1 > 0.5) == labels), np.sum((predicts2 > 0.5) == labels)


num_epoches = 1000

for i in range(num_epoches):
    if i % 10 == 0:
        print(f'Epoch {i}')
    print(one_step(inputs))

predicts1 = model1(inputs)
predicts2 = model2(inputs)

plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:, 0] > 0.5)
x = np.linspace(-10, 10, 100)
(W1, b1) = model1.weights
(W2, b2) = model2.weights
y1 = -W1[0] / W1[1] * x + (0.5 - b1) / W1[1]
y2 = -W2[0] / W2[1] * x + (0.0 - b2) / W2[1]
plt.plot(x, y1, '-r')
plt.plot(x, y2, '-g')
plt.show()

print(model1.weights)
print(model2.weights)
