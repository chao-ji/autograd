from operation import Operation
from tensor_shape import TensorShape
import numpy as np

from mixins import _ShapeAsIs, _ScalarShape


class Const(Operation):
  def __init__(self, value, graph=None, name=None):
    self._value = value
    super(Const, self).__init__(graph=graph, name=name)

  def _run(self):
    """Returns numpy array."""
    return self._value   

  def _compute_shapes(self):
    return [TensorShape(list(self._value.shape))]


class Placeholder(Operation):
  pass


class Variable(Operation):

  def __init__(self, initializaer, graph=None, name=None):
    pass


class ZerosLike(Operation, _ShapeAsIs):
  def _run(self, tensor_value):
    outputs = np.zeros_like(tensor_value, dtype="float32")
    return outputs


class OnesLike(Operation, _ShapeAsIs):
  def _run(self, tensor_value):
    outputs = np.ones_like(tensor_value, dtype="float32")
    return outputs


class Shape(Operation):
  def _run(self, *tensor_values):
    outputs = [np.asarray(tensor_value.shape) for tensor_value in tensor_values]
    return outputs

  def _compute_shapes(self):
    return [TensorShape([None]) if tensor.shape.level == 0 else 
        TensorShape([tensor.shape.ndims]) for tensor in self._input_list
    ]


class Size(Operation, _ScalarShape):
  def _run(self, tensor_value):
    outputs = np.size(tensor_value)
    return outputs


class Rank(Operation, _ScalarShape):
  def _run(self, tensor_value):
    outputs = len(tensor_value.shape)
    return outputs

