from collections import namedtuple

import numpy as np

from .mixins import _ScalarShape
from .operation import Operation
from .tensor_shape import TensorShape


class VariableSpec(object):

  def __init__(self, id, shape):
    self._id = id
    self._shape = shape

  @property
  def id(self):
    return self._id

  @property
  def shape(self):
    return self._shape


class Placeholder(Operation):

  def __init__(self, shape, graph=None, name=None):
    self._shape = list(shape)
    super(Placeholder, self).__init__(graph=graph, name=name)

  def _run(self):
    if self.id not in self._graph._runtime._placeholder_values:
      raise ValueError(f"Placeholder {self.id}'s value is not initialized.")
    outputs = self._graph._runtime._placeholder_values[self.id]
    return outputs

  def _compute_shapes(self):
    return [TensorShape(self._shape)]


class CreateVariable(Operation, _ScalarShape):

  def __init__(self, shape, init_fn, graph=None, name=None):
    self._shape = tuple(shape)
    self._init_fn = init_fn
    super(CreateVariable, self).__init__(graph=graph, name=name)

  def _run(self):
    if self.id not in self._graph._runtime._variable_values:
      init_value = self._init_fn(shape=self._shape)
      assert init_value.shape == self._shape
      self._graph._runtime._variable_values[self.id] = init_value
      #self._graph._runtime.set_variable_value(self.id, init_value)

    return np.asarray(VariableSpec(id=self.id, shape=self._shape))


class AssignVariable(Operation):

  def _run(self, variable_spec, new_value):

    variable_spec = variable_spec.item()
    self._graph._runtime._variable_values[variable_spec.id] = new_value

  @property
  def num_outputs(self):
    return 0

  def _compute_shapes(self):
    return None


class AddToVariable(Operation):

  def _run(self, variable_spec, delta):
    variable_spec = variable_spec.item()

    self._graph._runtime._variable_values[variable_spec.id] += delta

  @property
  def num_outputs(self):
    return 0

  def _compute_shapes(self):
    return None


class ReadVariable(Operation):

  def _run(self, variable_spec):
    variable_spec = variable_spec.item()
    outputs = self._graph._runtime.get_variable_value(variable_spec.id)
    return outputs

  def _compute_shapes(self):
    return [TensorShape(list(self._input_list[0].op._shape))]

  def mutable(self):
    return True
