"""Operation 
"""
import numpy as np
from collections import defaultdict

from default_stack import _DEFAULT_GRAPH_STACK


from containers import get_default_graph


class Operation(object):
  """
  """
  def __init__(self, graph=None, input_list=[], name=None):
    """Constructor.

    Args:
      graph (Graph): (Optional) the Graph object in which this Operation is
        defined. If None, a default graph will be created. Defaults to None.
      input_list (List[Tuple]): (Optional) a list input tensors (Operation,
        tensor_index). Defaults to empty list.
      name (str): (Optional) name of this Operation. If None, a name will be
        automatically generated. Defaults to None.
    """
    self._graph = get_default_graph() if graph is None else graph
    self._input_list = input_list # list of (Op, value_index)
    self._graph.add_op(op=self, name=name)
    self._consumers = defaultdict(list) 

    if hasattr(self, "_grad_func"):
      for op, tensor_index in input_list:
        op._consumers[tensor_index].append(self)

  def __repr__(self):
    repstr = (
        f"<Operation '{self._graph._ops[self].name}', "
        f"type={self._graph._ops[self].type}, "
        f"id={self._graph._ops[self].id}>"
    )
    return repstr

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

  def backprop(self, grad_tensors):
    #return self._grad_func(grad_tensors)


    from math_ops import AddN 

    grad_accumulate = dict() 
    grad_accumulate[self] = defaultdict(
        list, {index:
            [grad_tensor] for index, grad_tensor in enumerate(grad_tensors)
        }
    )

    queue = [(self, grad_tensors)]    # list of (Op, list of tensors)
    while len(queue):
      op, grad_tensors = queue.pop(0)

      if not hasattr(op, "_grad_func"):
        continue

      #print("op", op)
      #print("input_list", op._input_list)
      #print("grad_tensors", grad_tensors)
      #grad_tensor_list = op._grad_func(grad_tensors)
      #print("grad_tensor_list", grad_tensor_list)
      #print("\n" * 5)

      for (op, tensor_index), grad_tensor in zip(
          op._input_list, 
          #grad_tensor_list
          op._grad_func(grad_tensors)
        ):
        if op not in grad_accumulate:
          grad_accumulate[op] = defaultdict(list)
        grad_accumulate[op][tensor_index].append(grad_tensor)

        #if all(len(op._consumers[k]) == len(grad_accumulate[op][k]) for k in op._consumers.keys()):
        if True:
          grad_tensors = []
          for k in sorted(grad_accumulate[op].keys()):
            if len(grad_accumulate[op][k]) > 1:
              grad_tensors.append(
                  (AddN(input_list=grad_accumulate[op][k], graph=self._graph), 0)
            )
            else:
              grad_tensors.append(grad_accumulate[op][k][0])
          queue.append((op, grad_tensors))

    return grad_accumulate


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
    """Create the `Shape` op for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      shape_op (Operation): a `Shape` op.
    """
    if (self, tensor_index) not in self._graph._shape_ops:
      shape_op = Shape(
          input_list=[(self, tensor_index)],
          name=self.name+"_Shape"
      )    
      self._graph._shape_ops[(self, tensor_index)] = shape_op
    return self._graph._shape_ops[(self, tensor_index)]

  def get_size_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._size_ops:
      size_op = Size(
          input_list = [(self, tensor_index)],
          name=self.name+"_Size"
      )
      self._graph._size_ops[(self, tensor_index)] = size_op
    return self._graph._size_ops[(self, tensor_index)]

  def get_rank_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._rank_ops:
      rank_op = Rank(
          input_list = [(self, tensor_index)],
          name=self.name+"_Rank"
      )
      self._graph._rank_ops[(self, tensor_index)] = rank_op
    return self._graph._rank_ops[(self, tensor_index)]

  def get_zeros_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._zeros_ops:
      zeros_op = ZerosLike(
          input_list=[(self, tensor_index)],
          name=self.name+"_Zeros"
      )
      self._graph._zeros_ops[(self, tensor_index)] = zeros_op
    return self._graph._zeros_ops[(self, tensor_index)]

  def get_ones_op(self, tensor_index=0):
    if (self, tensor_index) not in self._graph._ones_ops:
      ones_op = OnesLike(
          input_list=[(self, tensor_index)],
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

