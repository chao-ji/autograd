from operation import Operation

import numpy as np


class Const(Operation):
  def __init__(self, value, graph=None, name=None):
    super(Const, self).__init__(graph=graph, name=name)
    self._value = value

  def _run(self):
    """Returns numpy array."""
    return self._value   


class Placeholder(Operation):
  pass


class Variable(Operation):

  def __init__(self, initializaer, graph=None, name=None):
    pass



class Zeros(Operation):
  def _run(self, tensor_shape):
    outputs = np.zeros(tensor_shape, dtype="float32")
    return outputs


class Ones(Operation):
  def _run(self, tensor_shape):
    outputs = np.ones(tensor_shape, dtype="float32")
    return outputs


class ZerosLike(Operation):
  def _run(self, tensor_value):
    outputs = np.zeros_like(tensor_value, dtype="float32")
    return outputs


class OnesLike(Operation):
  def _run(self, tensor_value):
    outputs = np.ones_like(tensor_value, dtype="float32")
    return outputs


class Shape(Operation):
  def _run(self, *tensor_values):
    outputs = [np.asarray(tensor_value.shape) for tensor_value in tensor_values]
    return outputs


class Size(Operation):
  def _run(self, tensor_value):
    outputs = np.size(tensor_value)
    return outputs


class Rank(Operation):
  def _run(self, tensor_value):
    outputs = len(tensor_value.shape)
    return outputs
