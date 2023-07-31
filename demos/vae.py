"""Generating MNIST images of digits using Variantional Autoencoder."""
import gzip
import os
import sys

import numpy as np

import autograd as ag


def read_mnist_images(fn):
  with gzip.open(fn, 'rb') as f:
    content = f.read()
    num_images = int.from_bytes(content[4:8], byteorder='big')
    height = int.from_bytes(content[8:12], byteorder='big')
    width = int.from_bytes(content[12:16], byteorder='big')
    images = np.frombuffer(
        content[16:], dtype=np.uint8,
    ).reshape((num_images, height, width))

  return images


class Encoder(ag.layers.Layer):

  def __init__(self, latent_dim):
    super(Encoder, self).__init__()
    self._conv0 = ag.layers.Conv2D(
        filters=32,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding="VALID",
        use_bias=True,
        activation="relu",
    )
    self._conv1 = ag.layers.Conv2D(
        filters=64,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding="VALID",
        use_bias=True,
        activation="relu",
    )
    self._dense = ag.layers.Dense(latent_dim * 2, use_bias=True)

  def __call__(self, inputs):
    outputs = self._conv0(inputs)
    outputs = self._conv1(outputs)
    outputs = ag.reshape(outputs, (batch_size, -1))
    outputs = self._dense(outputs)
    return outputs


class Decoder(ag.layers.Layer):

  def __init__(self):
    super(Decoder, self).__init__()
    self._dense = ag.layers.Dense(7 * 7 * 32, use_bias=True, activation="relu")
    self._tconv0 = ag.layers.Conv2DTranspose(
        filters=64,
        strides=(2, 2),
        kernel_size=(3, 3),
        use_bias=True,
        padding="SAME",
        activation="relu",
    )
    self._tconv1 = ag.layers.Conv2DTranspose(
        filters=32,
        strides=(2, 2),
        kernel_size=(3, 3),
        use_bias=True,
        padding="SAME",
        activation="relu",
    )
    self._tconv2 = ag.layers.Conv2DTranspose(
        filters=1,
        strides=(1, 1),
        kernel_size=(3, 3),
        use_bias=True,
        padding="SAME",
    )

  def __call__(self, inputs):
    outputs = self._dense(inputs)
    reshaped = ag.reshape(outputs, [-1, 7, 7, 32])
    outputs = self._tconv0(reshaped)
    outputs = self._tconv1(outputs)
    outputs = self._tconv2(outputs)
    return outputs


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = np.log(2. * np.pi)
  return ag.reduce_sum(
      -.5 * (ag.square(sample - mean) * ag.exp(-logvar) + logvar + log2pi),
      axis=raxis,
  )


def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')


def minibatch_generator(images, batch_size):
  images = images[np.random.permutation(images.shape[0])]
  for i in np.arange(images.shape[0] // batch_size):
    yield images[i * batch_size:(i + 1) * batch_size]


if __name__ == "__main__":

  batch_size = 32
  latent_dim = 2
  epochs = 10

  # build the graph
  encoder = Encoder(latent_dim)
  decoder = Decoder()

  images = ag.placeholder(shape=[batch_size, 28, 28, 1])

  params = encoder(images)
  mean = params[:, :latent_dim]
  logvar = params[:, latent_dim:]

  eps = ag.random_normal([batch_size, latent_dim])
  z = eps * ag.exp(logvar * 0.5) + mean
  x_logit = decoder(z)

  cross_ent = ag.sigmoid_cross_entropy_with_logits(
      logits=x_logit, labels=images,
  )
  logpx_z = -ag.reduce_sum(cross_ent, axis=[1, 2, 3])

  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  loss = -ag.reduce_mean(logpx_z + logpz - logqz_x)

  # optimizer
  optimizer = ag.optimizers.AdamOptimizer(
      alpha=1e-4,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-07,
  )

  grads_and_vars = optimizer.compute_gradients(
      loss, encoder.variables + decoder.variables,
  )

  # data
  path = "/home/chaoji/data/mnist"
  train_images = read_mnist_images(
      os.path.join(path, "train-images-idx3-ubyte.gz"),
  )
  train_images = train_images.reshape(
      train_images.shape[0], 28, 28,
      1,
  ).astype('float32')

  test_images = read_mnist_images(
      os.path.join(path, "t10k-images-idx3-ubyte.gz"),
  )
  test_images = test_images.reshape(
      test_images.shape[0], 28, 28,
      1,
  ).astype("float32")
  train_size = train_images.shape[0]
  test_size = test_images.shape[0]

  train_images = preprocess_images(train_images)
  test_images = preprocess_images(test_images)

  # training loop
  for epoch in np.arange(1, epochs + 1):
    train_data_generator = minibatch_generator(train_images, batch_size)
    test_data_generator = minibatch_generator(test_images, batch_size)

    for i in np.arange(train_size // batch_size):
      if i % 100 == 0:
        print(i)
      sys.stdout.flush()
      images.set_value(next(train_data_generator))
      optimizer.apply_gradients(grads_and_vars, reset_runtime=True)

    losses = []
    for i in np.arange(test_size // batch_size):
      images.set_value(next(test_data_generator))
      losses.append(loss.eval())

    print("loss", epoch, -np.mean(losses))

    decoder.save_variable_weights(f"vae_weights/weights_{epoch}")
