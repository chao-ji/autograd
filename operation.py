"""Defines base class `Operation` for all Ops and some generic Op types."""
import numpy as np
from collections import defaultdict

from containers import get_default_graph
from tensor import Tensor


class Operation(object):
  """
  """
  def __init__(self, graph=None, input_list=[], name=None):
    """Constructor.

    Args:
      graph (Graph): (Optional) the Graph object in which this Operation is
        affiliated. If None, a default graph will be created. Defaults to None.
      input_list (List[Tensor]): (Optional) list of input tensors. Defaults to
        empty list.
      name (str): (Optional) name of this Operation. If None, a name will be
        automatically generated. Defaults to None.
    """
    self._graph = get_default_graph() if graph is None else graph
    self._input_list = input_list
    self._graph.add_op(op=self, name=name)
    self._bp_indices = self._get_bp_indices()
    self._outputs = self._create_output_tensors()

  def _get_bp_indices(self):
    if not hasattr(self, "_grad_func"):
      bp_indices = [] 
    else:
      bp_indices = list(range(len(self._input_list)))
    return bp_indices

  def __repr__(self):
    name, type_, id_ = self._name, self._type, self._id
    repstr = f"<Operation '{name}', type={type_}, id={id_}>"
    return repstr

  def run(self):
    """Compute the value of the tensors coming out of this op."""
    # avoid re-running this op
    if self.id in self._graph._runtime._values:
      return

    # prepare the value of the input tensors of this op
    input_tensor_values = []
    for tensor in self._input_list:
      tensor.op.run() # make sure all depending ops have been executed
      value = self._graph._runtime.get_tensor_value(tensor)
      input_tensor_values.append(value)

    # run this op using the actual tensor values from depending ops
    outputs = self._run(*input_tensor_values)

    # save the output tensor values from this op to the runtime
    if isinstance(outputs, (list, tuple)):
      self._graph._runtime._values[self.id].extend(list(outputs))
    else:
      self._graph._runtime._values[self.id].append(outputs)

    #print(self)
    #print("a", [o.shape if hasattr(o, "shape") else () for o in self._graph._runtime._values[self.id]]) 
    #print("b", [o.shape.raw_shape for o in self._outputs])
    #print()

  def _compute_expected_backprops(self):
    queue = [self]
    expected_backprops = dict()
    while len(queue):
      op = queue.pop(0)
      for bp_index in op._bp_indices:
        tensor = op._input_list[bp_index]
        if tensor.op.id not in expected_backprops:
          expected_backprops[tensor.op.id] = defaultdict(set)
        expected_backprops[tensor.op.id][tensor.tensor_index].add(op.id)
        queue.append(tensor.op)

    return expected_backprops

  def backprop(self, grad_tensors):
    """
    Args:
      grad_tensors (List[Tensor]): list of the same length as the output tensors of this op. 
    """
    from math_ops import AddN 

    expected_backprops = self._compute_expected_backprops()

    grad_accumulate = dict() # {op.id: {tensor_index: [list of received grad tensors]}}

    queue = [(self, grad_tensors)]    # list of (op, list of tensors)
    while len(queue):
      op, grad_tensors = queue.pop(0)

      if not hasattr(op, "_grad_func"):
        continue

      la = [op._input_list[bp_index] for bp_index in op._bp_indices]
      lb = op._grad_func(grad_tensors)
      assert len(la) == len(lb)

      for tensor, grad_tensor in zip(la, lb):


        op, tensor_index = tensor.op, tensor.tensor_index

        if op.id not in grad_accumulate:
          grad_accumulate[op.id] = defaultdict(list)
        grad_accumulate[op.id][tensor_index].append(grad_tensor)

        if all(
            len(expected_backprops[op.id][k]) == len(grad_accumulate[op.id][k])
            for k in expected_backprops[op.id].keys()
          ):
          grad_tensors = []
          for k in sorted(grad_accumulate[op.id].keys()):
            if len(grad_accumulate[op.id][k]) > 1:
              grad_tensor = AddN(input_list=grad_accumulate[op.id][k], graph=self._graph).output(0)
            else:
              grad_tensor = grad_accumulate[op.id][k][0]

            grad_tensors.append(grad_tensor)
            grad_accumulate[op.id][k] = [grad_tensor]
          queue.append((op, grad_tensors))

    return grad_accumulate, expected_backprops

  @property
  def name(self):
    return self._name

  @property
  def type(self):
    return self._type

  @property
  def id(self):
    return self._id

  @property
  def num_outputs(self):
    return 1

  def _create_output_tensors(self):
    if not hasattr(self, "_outputs"):
      shapes = self._compute_shapes()
      self._outputs = [Tensor(self, i, shape) for i, shape in zip(range(self.num_outputs), shapes)]
    return self._outputs

  def output(self, index=0):
    output_tensor = self._create_output_tensors()[index]
    return output_tensor

  def _get_dependent_tensor(self, op, name, dic, tensor_index):
    tensor = self.output(tensor_index)
    if tensor not in dic:
      dependent_tensor = op(input_list=[tensor], name=name).output(0)
      dic[tensor] = dependent_tensor
    return dic[tensor]

  def get_shape_tensor(self, tensor_index=0):
    """Create the shape tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      shape_tensor (Tensor): a shape tensor.
    """
    from generic_ops import Shape
    return self._get_dependent_tensor(Shape, self.name+"_Shape", self._graph._shape_tensors, tensor_index)

  def get_size_tensor(self, tensor_index=0):
    from generic_ops import Size
    return self._get_dependent_tensor(Size, self.name+"_Size", self._graph._size_tensors, tensor_index)

  def get_rank_tensor(self, tensor_index=0):
    from generic_ops import Rank
    return self._get_dependent_tensor(Rank, self.name+"_Rank", self._graph._rank_tensors, tensor_index)

  def get_zeros_tensor(self, tensor_index=0):
    from generic_ops import ZerosLike
    return self._get_dependent_tensor(ZerosLike, self.name+"_ZerosLike", self._graph._zeroslike_tensors, tensor_index)

  def get_ones_tensor(self, tensor_index=0):
    from generic_ops import OnesLike
    return self._get_dependent_tensor(OnesLike, self.name+"_OnesLike", self._graph._oneslike_tensors, tensor_index)

