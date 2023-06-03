"""Operation 
"""
import numpy as np


class Operation(object):
  """
  """
  def __init__(self, graph, input_list=[], name=None):
    """Constructor.

    Args:
      graph (Graph): the Graph object in which this Operation is defined.
      input_list (List[Tuple]): a list of (`Operation, ``)
      name (str): name of this Operation.
    """
    self._graph = graph
    self._input_list = input_list # list of (Op, value_index)
    self._graph.add_op(op=self, name=name)

  def run(self):
    """Compute the value of the tensors coming out of this Op."""
    # avoid re-running this Op
    if self.name in self._graph._runtime._values:
      return

    # prepare the value of the input tensors of this Op
    input_tensor_values = []
    for op, tensor_index in self._input_list:
      op.run() # make sure all parent Ops have been executed
      value = self._graph._runtime._values[op.name][tensor_index]
      input_tensor_values.append(value)

    # run this Op using the actual tensor values from parent Ops
    outputs = self._run(*input_tensor_values)

    # save the output tensor values from this Op to the runtime
    if isinstance(outputs, (list, tuple)):
      self._graph._runtime._values[self.name].extend(list(outputs))
    else:
      self._graph._runtime._values[self.name].append(outputs)

  def backprop(self, backward_val):
    pass

  @property
  def name(self):
    return self._graph._ops[self].name

  @property
  def type(self):
    return self._graph._ops[self].type

  @property
  def id(self):
    return self._graph._ops[self].id 

  def get_shape_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._shape_ops:
      shape_op = Shape(
          input_list=[(self, tensor_index)],
          graph=self._graph,
          name=self.name+"_Shape"
      )    
      self._graph._shape_ops[(self, tensor_index)] = shape_op
    return self._graph._shape_ops[(self, tensor_index)]

  def get_size_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._size_ops:
      size_op = Size(
          input_list = [(self, tensor_index)],
          graph=self._graph,
          name=self.name+"_Size"
      )
      self._graph._size_ops[(self, tensor_index)] = size_op
    return self._graph._size_ops[(self, tensor_index)]

  def get_rank_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._rank_ops:
      rank_op = Rank(
          input_list = [(self, tensor_index)],
          graph=self._graph,
          name=self.name+"_Rank"
      )
      self._graph._rank_ops[(self, tensor_index)] = rank_op
    return self._graph._rank_ops[(self, tensor_index)]

  def get_zeros_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._zeros_ops:
      #zeros_op = Zeros(
      #    input_list=[(self.get_shape_op(), tensor_index)],
      #    graph=self._graph,
      #    name=self.name+"_Zeros"
      #) 
      zeros_op = ZerosLike(
          input_list=[(self, tensor_index)],
          graph=self._graph,
          name=self.name+"_Zeros"
      )
      self._graph._zeros_ops[(self, tensor_index)] = zeros_op
    return self._graph._zeros_ops[(self, tensor_index)]

  def get_ones_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._ones_ops:
      #ones_op = Ones(
      #    input_list=[(self.get_shape_op(), tensor_index)],
      #    graph=self._graph,
      #    name=self.name+"_Ones"
      #)
      ones_op = OnesLike(
          input_list=[(self, tensor_index)],
          graph=self._graph,
          name=self.name+"_Ones"
      )
      self._graph._ones_ops[(self, tensor_index)] = ones_op
    return self._graph._ones_ops[(self, tensor_index)]


class Zeros(Operation):
  def _run(self, tensor_shape):
    outputs = np.zeros(tensor_shape, dtype="float32")
    return outputs

    
class Ones(Operation):
  def _run(self, tensor_shape):
    outputs = np.ones(tensor_shape)
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
    outputs = np.asarray(len(tensor_value.shape))
    return outputs

