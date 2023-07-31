"""Classifying MNIST images by a convolutional network"""
import gzip
import os

import numpy as np

import autograd as ag

random_state = np.random.RandomState(0)


def read_mnist_labels(fn):
  with gzip.open(fn, 'rb') as f:
    content = f.read()
    num_images = int.from_bytes(content[4:8], byteorder='big')
    labels = np.zeros((num_images, 10), dtype=np.float32)
    indices = np.frombuffer(content[8:], dtype=np.uint8)
    labels[np.arange(num_images), indices] += 1

  return labels


def read_mnist_images(fn):
  with gzip.open(fn, 'rb') as f:
    content = f.read()
    num_images = int.from_bytes(content[4:8], byteorder='big')
    height = int.from_bytes(content[8:12], byteorder='big')
    width = int.from_bytes(content[12:16], byteorder='big')
    images = np.frombuffer(
        content[16:], dtype=np.uint8,
    ).reshape((num_images, height, width))
  images = images.astype(np.float32) / 255.
  return images


class ConvNet(ag.layers.Layer):

  def __init__(self):
    super(ConvNet, self).__init__()
    self._conv2d_1_1 = ag.layers.Conv2D(
        filters=16,
        strides=(1, 1),
        kernel_size=(3, 3),
        use_bias=False,
        padding="SAME",
        activation=None,
        kernel_initializer="truncated_normal",
    )
    self._conv2d_1_2 = ag.layers.Conv2D(
        filters=32,
        strides=(2, 2),
        kernel_size=(3, 3),
        use_bias=False,
        padding="SAME",
        activation=None,
        kernel_initializer="truncated_normal",
    )
    self._conv2d_2_1 = ag.layers.Conv2D(
        filters=64,
        strides=(1, 1),
        kernel_size=(3, 3),
        use_bias=False,
        padding="SAME",
        activation=None,
        kernel_initializer="truncated_normal",
    )
    self._conv2d_2_2 = ag.layers.Conv2D(
        filters=128,
        strides=(2, 2),
        kernel_size=(3, 3),
        use_bias=False,
        padding="SAME",
        activation=None,
        kernel_initializer="truncated_normal",
    )
    self._conv2d_3 = ag.layers.Conv2D(
        filters=256,
        strides=(2, 2),
        kernel_size=(3, 3),
        use_bias=False,
        padding="SAME",
        activation=None,
        kernel_initializer="truncated_normal",
    )
    self._bn_1_1 = ag.layers.BatchNormalization(momentum=0.99, epsilon=0.0001)
    self._bn_1_2 = ag.layers.BatchNormalization(momentum=0.99, epsilon=0.0001)
    self._bn_2_1 = ag.layers.BatchNormalization(momentum=0.99, epsilon=0.0001)
    self._bn_2_2 = ag.layers.BatchNormalization(momentum=0.99, epsilon=0.0001)
    self._bn_3 = ag.layers.BatchNormalization(momentum=0.99, epsilon=0.0001)
    self._dense = ag.layers.Dense(10, use_bias=True)

  def __call__(self, inputs, training=False):
    outputs_1_1 = ag.relu(
        self._bn_1_1(self._conv2d_1_1(inputs), training=training),
    )
    outputs_1_2 = ag.relu(
        self._bn_1_2(self._conv2d_1_2(outputs_1_1), training=training),
    )
    outputs_2_1 = ag.relu(
        self._bn_2_1(self._conv2d_2_1(outputs_1_2), training=training),
    )
    outputs_2_2 = ag.relu(
        self._bn_2_2(self._conv2d_2_2(outputs_2_1), training=training),
    )
    outputs_3 = ag.relu(
        self._bn_3(self._conv2d_3(outputs_2_2), training=training),
    )
    pool = ag.reduce_mean(outputs_3, [1, 2])
    logits = self._dense(pool)
    return logits


def minibatch_generator(labels, images, batch_size):
  while True:
    which = random_state.choice(train_images.shape[0], batch_size, False)
    yield labels[which], images[which]


if __name__ == "__main__":
  graph = ag.get_default_graph()

  batch_size = 50

  # build graph
  convnet = ConvNet()
  inputs = ag.placeholder(shape=(batch_size, 28, 28, 1))
  labels = ag.placeholder(shape=(batch_size, 10))
  logits = convnet(inputs, True)
  preds = convnet(inputs, False)

  losses = ag.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = ag.reduce_mean(losses)
  variables = convnet.variables

  # optimizer
  adam = ag.optimizers.AdamOptimizer(
      alpha=0.001, beta1=.9, beta2=.999, epsilon=1e-8,
  )
  grads_and_vars = adam.compute_gradients(loss, variables)

  # data
  path = "/home/chaoji/data/mnist"
  train_images = read_mnist_images(
      os.path.join(path, "train-images-idx3-ubyte.gz"),
  )
  train_labels = read_mnist_labels(
      os.path.join(path, "train-labels-idx1-ubyte.gz"),
  )
  test_images = read_mnist_images(
      os.path.join(path, "t10k-images-idx3-ubyte.gz"),
  )
  test_labels = read_mnist_labels(
      os.path.join(path, "t10k-labels-idx1-ubyte.gz"),
  )
  train_images = train_images[..., np.newaxis]
  test_images = test_images[..., np.newaxis]
  train_data_generator = minibatch_generator(
      train_labels, train_images, batch_size,
  )

  # training loops
  iterations = 1000
  for i in np.arange(iterations):
    batch_labels, batch_images = next(train_data_generator)
    inputs.set_value(batch_images)
    labels.set_value(batch_labels)

    if i % 100 == 0:
      print(f"step: {i}, loss: {loss.eval()}")

    adam.apply_gradients(grads_and_vars, reset_runtime=True)

    if i % 100 == 0:
      assert len(
          graph.runtime._values,
      ) == 0 and len(graph.runtime._placeholder_values) == 0
      inputs.set_value(test_images)
      print((preds.eval().argmax(axis=1) == test_labels.argmax(axis=1)).mean())
      graph.runtime.reset()
