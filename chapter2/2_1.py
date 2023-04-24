import numpy as np
import tensorflow as tf

x = tf.Variable(0.0)
with tf.GradientTape() as tape:
    y = x * x + 2 * x + 1
grad_y_x = tape.gradient(y, x)

print(grad_y_x)


from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images_raw, train_labels), (test_images_raw, test_labels) = mnist.load_data()
train_images = train_images_raw.reshape(60000, 28 * 28)
train_images =  train_images.astype('float32') / 255
test_images = test_images_raw.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, batch_size=128, epochs=5)

loss, accuracy = model.evaluate(test_images, test_labels)
print(f'loss: {loss}, accuracy: {accuracy}')

test_predicts = model.predict(test_images).argmax(axis=1)

error_indices = np.arange(10000)[test_labels != test_predicts]
for error_i in error_indices[:5]:
    plt.imshow(test_images_raw[error_i], cmap=plt.cm.binary)
    plt.title(f'index={error_i}, label={test_labels[error_i]}, predict={test_predicts[error_i]}')
    plt.show()
    print(f'index={error_i}, label={test_labels[error_i]}, predict={test_predicts[error_i]}')
