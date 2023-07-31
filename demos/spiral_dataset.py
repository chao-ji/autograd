import os

import matplotlib.pyplot as plt
import numpy as np

import autograd as ag

random_state = np.random.RandomState(0)

if __name__ == "__main__":

  num_features = 2  # 2 features
  num_classes = 3  # number of classes
  batch = 150  # number of points per class

  # build the graph (3-layer MLP)
  num_hidden1 = 50
  num_hidden2 = 50
  reg = 1e-3

  inputs = ag.placeholder([150, num_features])
  labels = ag.placeholder([150, num_classes])

  kernel_initializer = ag.initializers.TruncatedNormalInitializer(
      mean=0.0, stddev=0.01,
  )

  dense1 = ag.layers.Dense(
      num_hidden1,
      use_bias=True,
      activation="relu",
      kernel_initializer=kernel_initializer,
  )
  dense2 = ag.layers.Dense(
      num_hidden2,
      use_bias=True,
      activation="relu",
      kernel_initializer=kernel_initializer,
  )
  dense3 = ag.layers.Dense(
      num_classes,
      use_bias=True,
      activation=None,
      kernel_initializer=kernel_initializer,
  )

  logits = dense3(dense2(dense1(inputs)))

  loss = ag.reduce_mean(
      ag.softmax_cross_entropy_with_logits(labels=labels, logits=logits),
  )

  l2norm = (
      ag.reduce_sum(ag.square(dense1.variables[0].weight)) +
      ag.reduce_sum(ag.square(dense1.variables[1].weight)) +
      ag.reduce_sum(ag.square(dense2.variables[0].weight)) +
      ag.reduce_sum(ag.square(dense2.variables[1].weight)) +
      ag.reduce_sum(ag.square(dense3.variables[0].weight)) +
      ag.reduce_sum(ag.square(dense3.variables[1].weight))
  )
  loss = loss + reg * l2norm

  # optimizer
  gd = ag.optimizers.GradientDescentOptimizer(alpha=0.1)
  variables = dense1.variables + dense2.variables + dense3.variables
  grads_and_vars = gd.compute_gradients(loss, variables)

  # data
  # Creaing an artifical classification dataset that are not linearly separable.
  x = np.zeros([batch * num_classes, num_features])
  y = np.zeros([batch * num_classes, num_classes])

  for j in range(num_classes):
    ix = range(batch * j, batch * (j + 1))
    r = np.linspace(0.0, 1, batch)
    t = np.linspace(j * 4, (j + 1) * 4, batch) + random_state.randn(batch) * 0.2
    x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix, j] = 1.0

  indices = np.arange(batch * num_classes)
  random_state.shuffle(indices)

  x_train = x[indices[:300]]
  y_train = y[indices[:300]]
  x_test = x[indices[300:]]
  y_test = y[indices[300:]]

  # The dataset is like a rotating galaxy with three spiral arms.
  plt.scatter(x[:150, 0], x[:150, 1], color='r')

  plt.scatter(x[150:300, 0], x[150:300, 1], color='g')
  plt.scatter(x[300:450, 0], x[300:450, 1], color='b')

  fig = plt.gcf()
  fig.set_size_inches(10, 10)
  plt.show()

  # training loops
  for i in np.arange(10000):

    inputs.set_value(x_train)
    labels.set_value(y_train)

    if i % 1000 == 0:
      print('step: %d, loss: %f' % (i, loss.eval()))

    gd.apply_gradients(grads_and_vars, reset_runtime=True)

  inputs.set_value(x_test)
  print(
      'test accuracy:',
      np.mean(np.argmax(logits.eval(), axis=1)==np.argmax(y_test, axis=1)),
  )
