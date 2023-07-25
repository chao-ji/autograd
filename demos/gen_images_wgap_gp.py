"""Show wgan-generated images."""
import matplotlib.pyplot as plt
import numpy as np
from wgan_gp import Generator

import autograd as ag

if __name__ == "__main__":
  graph = ag.get_default_graph()

  noise_dim = 128
  batch_size = 100
  noises = ag.random_normal([batch_size, noise_dim])

  generator = Generator()
  fake_images = generator(noises)

  generator.load_variable_weights("gp_weights/weights_12000.npy")

  preds = graph.runtime.get_tensor_value(fake_images)
  graph.runtime.reset()

  fig = plt.figure(figsize=(10, 10))
  for i in range(preds.shape[0]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

  plt.savefig("wgan_gp_image.png")
