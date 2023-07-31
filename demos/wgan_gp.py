"""Generating MNIST images of digits using WGAN with gradient penalty.
https://arxiv.org/abs/1704.00028
"""
import gzip
import os
import sys

import numpy as np

import autograd as ag

random_state = np.random.RandomState(None)


def read_mnist_images(fn):
  with gzip.open(fn, 'rb') as f:
    content = f.read()
    num_images = int.from_bytes(content[4:8], byteorder='big')
    height = int.from_bytes(content[8:12], byteorder='big')
    width = int.from_bytes(content[12:16], byteorder='big')
    images = np.frombuffer(
        content[16:],
        dtype=np.uint8,
    ).reshape((num_images, height, width))
  return images


class Generator(ag.layers.Layer):

  def __init__(self):
    super(Generator, self).__init__()
    self._dense = ag.layers.Dense(4 * 4 * 256, use_bias=True, activation="relu")
    self._tconv0 = ag.layers.Conv2DTranspose(
        filters=128,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="relu",
    )
    self._tconv1 = ag.layers.Conv2DTranspose(
        filters=64,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="relu",
    )
    self._tconv2 = ag.layers.Conv2DTranspose(
        filters=1,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="tanh",
    )

  def __call__(self, inputs, training=False):
    outputs0 = self._dense(inputs)
    reshaped = ag.reshape(outputs0, [-1, 4, 4, 256])
    tconv0 = self._tconv0(reshaped)
    tconv0 = tconv0[:, :-1, :-1]
    tconv1 = self._tconv1(tconv0)
    tconv2 = self._tconv2(tconv1)
    return tconv2


class Discriminator(ag.layers.Layer):

  def __init__(self):
    super(Discriminator, self).__init__()
    self._conv0 = ag.layers.Conv2D(
        filters=64,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="leaky_relu",
    )
    self._conv1 = ag.layers.Conv2D(
        filters=128,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="leaky_relu",
    )
    self._conv2 = ag.layers.Conv2D(
        filters=256,
        strides=(2, 2),
        kernel_size=(5, 5),
        use_bias=True,
        padding="SAME",
        activation="leaky_relu",
    )
    self._dense0 = ag.layers.Dense(1, use_bias=True)

  def __call__(self, images):
    outputs = self._conv0(images)
    outputs = self._conv1(outputs)
    outputs = self._conv2(outputs)
    reshaped = ag.reshape(outputs, (50, -1))
    logits = self._dense0(reshaped)
    return logits


def minibatch_generator(images, batch_size):
  while True:
    yield images[
        random_state.choice(
            images.shape[0],
            batch_size,
            False,
        )
    ].astype("float32")


if __name__ == "__main__":
  noise_dim = 128
  batch_size = 50

  # build the graph
  noises = ag.random_normal([batch_size, noise_dim])
  real_images = ag.placeholder(shape=[batch_size, 28, 28, 1])
  epsilon = ag.random_uniform([batch_size, 1, 1, 1])

  generator = Generator()
  discriminator = Discriminator()

  fake_images = generator(noises)

  fake_logits = discriminator(fake_images)
  real_logits = discriminator(real_images)

  raw_discriminator_loss = ag.reduce_mean(
      fake_logits,
  ) - ag.reduce_mean(real_logits)

  images_hat = real_images * epsilon + fake_images * (1 - epsilon)
  logits_hat = discriminator(images_hat)
  grad_images_hat = ag.backprop([logits_hat], [images_hat])[0]
  gp_loss = ag.reduce_mean(
      ag.square(
          ag.sqrt(ag.reduce_sum(ag.square(grad_images_hat), axis=[1, 2, 3])) -
          1,
      ),
  )

  discriminator_loss = raw_discriminator_loss + 10 * gp_loss
  generator_loss = -ag.reduce_mean(fake_logits)

  # optimizer and backprop graph
  optimizer_d = ag.optimizers.AdamOptimizer(
      alpha=0.0001,
      beta1=0.0,
      beta2=0.9,
      epsilon=1e-07,
  )
  optimizer_g = ag.optimizers.AdamOptimizer(
      alpha=0.0001,
      beta1=0.0,
      beta2=0.9,
      epsilon=1e-07,
  )

  grads_and_vars_d = optimizer_d.compute_gradients(
      discriminator_loss,
      discriminator.variables,
  )
  grads_and_vars_g = optimizer_g.compute_gradients(
      generator_loss,
      generator.variables,
  )

  # data
  path = "/home/chaoji/data/mnist"
  train_images = read_mnist_images(
      os.path.join(path, "train-images-idx3-ubyte.gz"),
  )
  train_images = train_images.reshape(
      train_images.shape[0],
      28,
      28,
      1,
  ).astype('float32')
  train_images = (
      train_images - 127.5
  ) / 127.5  # Normalize the images to [-1, 1]

  data_generator = minibatch_generator(train_images, batch_size)

  # training loops
  for i in np.arange(15001):
    for j in np.arange(5):
      real_images.set_value(next(data_generator))
      optimizer_d.apply_gradients(grads_and_vars_d, reset_runtime=True)
    optimizer_g.apply_gradients(grads_and_vars_g, reset_runtime=True)

    if i % 100 == 0:
      print("i", i)
      real_images.set_value(next(data_generator))

      print("discriminator_loss:", discriminator_loss.eval())
      print("generator_loss:", generator_loss.eval())
      sys.stdout.flush()
      print()

    if i % 200 == 0:
      generator.save_variable_weights(f"gp_weights/weights_{i}")
