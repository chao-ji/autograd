"""Defines Ops for creating, updating, and reading variables."""
from collections import namedtuple

import numpy as np

from .mixins import _ScalarShape
from .operation import Operation
from .tensor_shape import TensorShape


class Placeholder(Operation):
  """Op that generates a `Placeholder` Tensor.

  Input (0): None.

  Output (1): a `Placeholder` Tensor.

  Side Effect: None.
  """

  def __init__(self, shape, graph=None, name=None):
    """Constrcutor.

    Args:
      shape (List or Tuple): shape of the `Placeholder` Tensor.
      graph (Graph): (Optional) the parent `Graph`.
      name (str): (Optional) name of the Op.
    """
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
  """Op that initializes a variable.

  Input (0): None.

  Output (1): ID of this `CreateVariable` Op.

  Side Effect: setting the initialized value in the `Runtime` in which the
    `Graph` runs.
  """

  def __init__(self, shape, init_fn, graph=None, name=None):
    """Constructor.

    Args:
      shape (List or Tuple): shape of the variable.
      init_fn (callable): a function that return the value of the variable.
      graph (Graph): (Optional) the parent `Graph`.
      name (str): (Optional) name of the Op.
    """
    self._shape = tuple(shape)
    self._init_fn = init_fn
    super(CreateVariable, self).__init__(graph=graph, name=name)

  def _run(self):
    if self.id not in self._graph._runtime._variable_values:
      init_value = self._init_fn(shape=self._shape)
      assert init_value.shape == self._shape
      self._graph._runtime._variable_values[self.id] = init_value

    return np.asarray(self.id)


class AssignVariable(Operation):
  """Op that has the side effect of updating the value of a variable with
  `new_value`.

  Input (2): a Tensor from `CreateVariable` Op and another Tensor whose value is
    used to assign to the variable.

  Output (0): None.

  Side Effect: setting the new value in the `Runtime` in which the `Graph` runs.
  """

  def _run(self, creator_id, new_value):
    self._graph_runtime._variable_values[int(creator_id)] = new_value

  @property
  def num_outputs(self):
    return 0

  def _compute_shapes(self):
    return None


class AddToVariable(Operation):
  """Op that has the side effect of updating the value of a variable by adding
  `delta` to the orignal value.

  Input (2): a Tensor from `CreateVariable` Op and another Tensor whose value is
    used to add to the variable value.

  Output (0): None.

  Side Effect: setting the new value in the `Runtime` in which the `Graph` runs.
  """

  def _run(self, creator_id, delta):
    self._graph._runtime._variable_values[int(creator_id)] += delta

  @property
  def num_outputs(self):
    return 0

  def _compute_shapes(self):
    return None


class ReadVariable(Operation):
  """Op that returns the value of an initialized variable.

  Input (1): a Tensor from `CreateVariable` Op.

  Output (1): the value of the variable.

  Side Effect: None.
  """

  def _run(self, creator_id):
    outputs = self._graph._runtime.get_variable_value(int(creator_id))
    return outputs

  def _compute_shapes(self):
    return [TensorShape(list(self._input_list[0].op._shape))]

  def mutable(self):
    return True
