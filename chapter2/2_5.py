import math

import keras.losses
import tensorflow as tf


class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, maxval=0.1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        next_index = self.index + self.batch_size
        batch = (self.images[self.index:next_index], self.labels[self.index:next_index])
        self.index = next_index
        return batch


def one_training_step(model, images, labels):
    with tf.GradientTape() as tape:
        predicts = model(images)
        loss = keras.losses.sparse_categorical_crossentropy(labels, predicts)
        average_loss = tf.reduce_mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


learning_rate = 1e-3


def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)


def fit(model, images, labels, num_epoches, batch_size=128):
    for i in range(num_epoches):
        print(f'Epoch {i}')
        bg = BatchGenerator(images, labels, batch_size)
        for j in range(bg.num_batches):
            batch_images, batch_labels = bg.next()
            loss = one_training_step(model, batch_images, batch_labels)
            if j % 100 == 0:
                print(f'loss[{j}] = {loss}')


from keras.datasets import mnist

(raw_train_images, train_labels), (raw_test_images, test_labels) = mnist.load_data()

train_images = raw_train_images.reshape(60000, 28 * 28)
train_images = train_images.astype('float32') / 255
test_images = raw_test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

fit(model, train_images, train_labels, num_epoches=10, batch_size=128)

test_predicts = model(test_images).numpy().argmax(axis=1)
matches = test_predicts == test_labels
print(matches.mean())
