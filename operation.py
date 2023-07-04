"""Defines base class `Operation` for all Ops and some generic Op types."""
import numpy as np
from collections import defaultdict

from containers import get_default_graph


class Operation(object):
  """
  """
  def __init__(self, graph=None, input_list=[], dependent_ops=[], name=None):
    """Constructor.

    Args:
      graph (Graph): (Optional) the Graph object in which this Operation is
        affiliated. If None, a default graph will be created. Defaults to None.
      input_list (List[Tensor]): (Optional) list of input tensors. Defaults to
        empty list.
      dependent_ops (List[Operation]): (Optional) list of Ops that should be
        run prior to this Op. Defaults to empty list.
      name (str): (Optional) name of this Operation. If None, a name will be
        automatically generated. Defaults to None.
    """
    self._graph = get_default_graph() if graph is None else graph
    self._input_list = input_list
    self._graph.add_op(op=self, name=name)
    self._bp_indices = self._get_bp_indices()
    self._outputs = self._create_output_tensors()
    self._dependent_ops = dependent_ops

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
    if not self.mutable and self.id in self._graph._runtime._values:
      return

    # prepare the value of the input tensors of this op
    input_tensor_values = []
    for tensor in self._input_list:
      tensor.op.run() # make sure all depending ops have been executed
      value = self._graph._runtime.get_tensor_value(tensor)
      input_tensor_values.append(value)

    # run dependent ops if any
    for op in self._dependent_ops:
      op.run()

    # run this op using the actual tensor values from depending ops
    outputs = self._run(*input_tensor_values)

    if self.mutable:
      self._graph._runtime._values[self.id] = []

    # save the output tensor values from this op to the runtime
    if isinstance(outputs, (list, tuple)):
      self._graph._runtime._values[self.id].extend(list(outputs))
    elif outputs is not None:
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

  def backprop(self, x_tensors, dy_tensors=None):
    """
    Args:
      x_tensors (List[Tensor]): list of tensors whose gradients are to be returned. 
      dy_tensors (List[Tensor]): list of gradient tensors w.r.t. the outputs of
        this Op. If None, defaults to tensor filled with ones.
   
    Returns:
      dx_tensors (List[Tensor]):  
    """
    from math_ops import AddN 
    from generic_ops import OnesLike

    if dy_tensors is None:
      dy_tensors = [OnesLike(input_list=[y_tensor]).output(0) for y_tensor in self._outputs]

    cum_grad = dict()
    expected_backprops = self._compute_expected_backprops()
    queue = [(self, dy_tensors)]

    while len(queue):
      op, dy_tensors = queue.pop(0)

      # Ops without grad functions are treated as constants and hence ignored
      if not hasattr(op, "_grad_func"):
        continue

      for tensor, grad_tensor in zip(
          # list of input tensors to `op`
          [op._input_list[bp_index] for bp_index in op._bp_indices],
          # list of computed gradient tensors w.r.t. input tensors to `op`
          op._grad_func(dy_tensors)
        ):

        if tensor.op.id not in cum_grad:
          cum_grad[tensor.op.id] = defaultdict(list)
        cum_grad[tensor.op.id][tensor.tensor_index].append(grad_tensor)

        if all(
            (
              len(expected_backprops[tensor.op.id][tensor_index]) ==
              len(cum_grad[tensor.op.id][tensor_index])
            )
            for tensor_index in expected_backprops[tensor.op.id].keys()
          ):

          dy_tensors = []
          for tensor_index in sorted(cum_grad[tensor.op.id].keys()):
            if len(cum_grad[tensor.op.id][tensor_index]) > 1:
              grad_tensor = AddN(
                  input_list=cum_grad[tensor.op.id][tensor_index],
                  graph=self._graph
              ).output(0)
            else:
              grad_tensor = cum_grad[tensor.op.id][tensor_index][0]

            dy_tensors.append(grad_tensor)
            cum_grad[tensor.op.id][tensor_index] = [grad_tensor]
          queue.append((tensor.op, dy_tensors))

    dx_tensors = []
    for x_tensor in x_tensors:
      dx_tensors.append(cum_grad[x_tensor.op.id][x_tensor.tensor_index][0])

    return dx_tensors

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

  @property
  def mutable(self):
    return False

  def _create_output_tensors(self):
    from tensor import Tensor

    if not hasattr(self, "_outputs"):
      shapes = self._compute_shapes()
      if shapes is None:
        self._outputs = []
      else:
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

