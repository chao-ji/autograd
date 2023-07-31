"""Show VAE generated images."""
import autograd as ag
import matplotlib.pyplot as plt
import numpy as np

from vae import Decoder

if __name__ == "__main__":
  graph = ag.get_default_graph()

  latent_dim = 2
  noises = ag.placeholder(shape=(1, latent_dim))
  decoder = Decoder()
  logits = decoder(noises)
  decoder.load_variable_weights("vae_weights/weights_311.npy")

  n = 20
  digit_size = 28
  norm_x = np.random.normal(size=(100,))
  norm_y = np.random.normal(size=(100,))
  grid_x = np.quantile(norm_x, np.linspace(0.05, 0.95, n))
  grid_y = np.quantile(norm_y, np.linspace(0.05, 0.95, n))
  image_width = digit_size * n
  image_height = image_width
  image = np.zeros((image_height, image_width))
  sigmoid = lambda x: 1 / (1 + np.exp(-x))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z_value = np.array([[xi, yi]])
      noises.set_value(z_value)
      probs_value = sigmoid(logits.eval())
      graph.runtime.reset()
      digit = np.reshape(probs_value[0], (digit_size, digit_size))
      image[
          i * digit_size:(i + 1) * digit_size,
          j * digit_size:(j + 1) * digit_size,
      ] = digit

  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.savefig("vae_image.png")
