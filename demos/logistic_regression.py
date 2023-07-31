import gzip
import os

import matplotlib.pyplot as plt
import numpy as np

import autograd as ag


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
    ).reshape((num_images, height * width))
  images = images.astype(np.float32) / 255.
  return images


def minibatch_generator(labels, images, batch_size):
  while True:
    which = np.random.choice(train_images.shape[0], batch_size, False)
    yield labels[which], images[which]


if __name__ == "__main__":
  graph = ag.get_default_graph()

  batch_size = 100

  # build the graph
  inputs = ag.placeholder(shape=(batch_size, 784))
  labels = ag.placeholder(shape=(batch_size, 10))

  kernel_initializer = ag.initializers.TruncatedNormalInitializer(
      mean=0.0, stddev=0.01,
  )
  dense = ag.layers.Dense(
      10, use_bias=True, activation=None, kernel_initializer=kernel_initializer,
  )

  logits = dense(inputs)

  loss = ag.reduce_mean(
      ag.softmax_cross_entropy_with_logits(labels=labels, logits=logits),
  )

  l2norm = (
      ag.reduce_sum(ag.square(dense.variables[0].weight)) +
      ag.reduce_sum(ag.square(dense.variables[1].weight))
  )

  reg = 1e-3
  loss = loss + reg * l2norm

  # optimizer
  variables = dense.variables
  gd = ag.optimizers.GradientDescentOptimizer(alpha=0.5)

  grads_and_vars = gd.compute_gradients(loss, variables)

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

    gd.apply_gradients(grads_and_vars, reset_runtime=True)

    if i % 100 == 0:
      assert len(
          graph.runtime._values,
      ) == 0 and len(graph.runtime._placeholder_values) == 0
      inputs.set_value(test_images)
      print(
          "test accuracy:", (
              logits.eval().argmax(axis=1) == test_labels.argmax(axis=1)
          ).mean(),
      )
      graph.runtime.reset()
