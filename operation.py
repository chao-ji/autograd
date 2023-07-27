"""Defines base class `Operation` for all Ops."""
from collections import defaultdict

import numpy as np

from .containers import get_default_graph


class Operation(object):
  """Base class for all Ops."""

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
    """Returns the indices of input tensors that expect a backpropped gradient
    tensor. Can be overrided by subclasses.

    If an Op does not have a `_grad_func`, then its output tensors are treated
    as "constants", hence no gradient will be backpropped to its input tensors.

    Returns:
      bp_indices (List[int]): list of indices of input tensors that expect
        backpropped gradient.
    """
    if not hasattr(self, "_grad_func"):
      # no backpropped gradient.
      bp_indices = []
    else:
      # assume all input tensors expect backpropped gradient.
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
      tensor.op.run()  # make sure all depending ops have been executed
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

  @property
  def graph(self):
    return self._graph

  def _create_output_tensors(self):
    """Set attribute `self._outputs` as the created tensors.

    Returns:
      outputs (List[Tensor]): the created Tensor instances.
    """
    from .tensor import Tensor

    if not hasattr(self, "_outputs"):
      shapes = self._compute_shapes()
      if shapes is None:
        # For Ops with no output tensors, set `self._outputs` to empty list.
        self._outputs = []
      else:
        # Assign tensor index and shape to each Tensor.
        self._outputs = [
            Tensor(self, i, shape)
            for i, shape in zip(range(self.num_outputs), shapes)
        ]
    return self._outputs

  def output(self, index=0):
    """Get one output tensor.

    Args:
      index (int): output index of the tensor. Defaults to 0.

    Returns:
      output_tensor (Tensor): the tensor with the provided output index.
    """
    output_tensor = self._create_output_tensors()[index]
    return output_tensor

  def _get_dependent_tensor(self, op, name, dic, tensor_index):
    """Retrieve or create a "dependet" tensor.

    Dependent tensors are those that depend on this Op (e.g. Shape, Rank). This
    is to avoid creating multiple `Shape` (or other) tensors of the same Op.
    """
    tensor = self.output(tensor_index)
    if tensor not in dic:
      dependent_tensor = op(input_list=[tensor], name=name).output(0)
      dic[tensor] = dependent_tensor
    return dic[tensor]

  def get_shape_tensor(self, tensor_index=0):
    """Create the Shape tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      shape_tensor (Tensor): a Shape tensor.
    """
    from .generic_ops import Shape
    return self._get_dependent_tensor(
        Shape,
        self.name + "_Shape",
        self._graph._shape_tensors,
        tensor_index,
    )

  def get_size_tensor(self, tensor_index=0):
    """Create the Size tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      size_tensor (Tensor): a Size tensor.
    """
    from .generic_ops import Size
    return self._get_dependent_tensor(
        Size,
        self.name + "_Size",
        self._graph._size_tensors,
        tensor_index,
    )

  def get_rank_tensor(self, tensor_index=0):
    """Create the Rank tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      rank_tensor (Tensor): a Rank tensor.
    """
    from .generic_ops import Rank
    return self._get_dependent_tensor(
        Rank,
        self.name + "_Rank",
        self._graph._rank_tensors,
        tensor_index,
    )

  def get_zeros_tensor(self, tensor_index=0):
    """Create the ZerosLike tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      zeros_like_tensor (Tensor): a ZerosLike tensor.
    """
    from .generic_ops import ZerosLike
    return self._get_dependent_tensor(
        ZerosLike,
        self.name + "_ZerosLike",
        self._graph._zeroslike_tensors,
        tensor_index,
    )

  def get_ones_tensor(self, tensor_index=0):
    """Create the OnesLike tensor for one of the output tensor of this op.

    Args:
      tensor_index (int): (Optional) index of the output tensor whose `Shape` op
        is to be created. Defaults to 0.

    Returns:
      zeros_like_tensor (Tensor): a OnesLike tensor.
    """
    from .generic_ops import OnesLike
    return self._get_dependent_tensor(
        OnesLike,
        self.name + "_OnesLike",
        self._graph._oneslike_tensors,
        tensor_index,
    )


def _compute_expected_backprops(op):
  """Traverse the graph from the `self` Op in the order of BFS, and count the
  number of gradient tensors that will be backpropped to each Op.

  Returns:
    expected_backprops (Dict[int, defaultdict(set)]): dict mapping ID of `op`
      to a defaultdict, which in turn maps the tensor index to the set of
      downstream Ops that will backprop gradients to `op`.
  """
  queue = [op]
  expected_backprops = dict()

  while len(queue):
    op = queue.pop(0)
    for bp_index in op._bp_indices:
      # `tensor` is an input of `op` that expects a backpropped gradient
      tensor = op._input_list[bp_index]
      if tensor.op.id not in expected_backprops:
        expected_backprops[tensor.op.id] = defaultdict(set)
      # record the set of IDs of Ops that will backprop gradients to `tensor`
      # (can be uniquely identified by `op.id` and `tensor_index`)
      expected_backprops[tensor.op.id][tensor.tensor_index].add(op.id)
      queue.append(tensor.op)

  return expected_backprops


def backprop(y_tensors, x_tensors, dy_tensors=None):
  """Backprop gradients `dy_tensors` to each tensor in `x_tensors`.

  Args:
    y_tensors (List[Tensor]): list of tensors from which gradients will be
      backpropped.
    x_tensors (List[Tensor]): list of tensors whose gradients are to be
      returned.
    dy_tensors (List[Tensor]): list of gradient tensors w.r.t. tensors in
      `y_tensors`. If None, defaults to tensors filled with ones.

  Returns:
    dx_tensors (List[Tensor]): List of gradient tensors backpropped to each
      tensor in `x_tensors`.
  """
  from .generic_ops import OnesLike
  from .math_ops import AddN

  # make sure `dy_tensors` matches `y_tensors`
  if dy_tensors is not None:
    assert len(y_tensors) == len(dy_tensors)
    for y_tensor, dy_tensor in zip(y_tensors, dy_tensors):
      y_tensor.shape._compatible_with(dy_tensor.shape)
  else:
    dy_tensors = [None] * len(y_tensors)

  # sort `dy_tensors`
  ops = list(set([y_tensor.op for y_tensor in y_tensors]))
  dy_tensor_map = defaultdict(dict)  #dict()
  for y_tensor, dy_tensor in zip(y_tensors, dy_tensors):
    dy_tensor_map[y_tensor.op.id][y_tensor.tensor_index] = dy_tensor

  # prepare grad tensors
  # if dy_tensor is None, defaults to tensors filled with ones
  grad_tensors = []
  for op in ops:
    for i, out_tensor in enumerate(op._outputs):
      if i not in dy_tensor_map[op.id] or dy_tensor_map[op.id][i] is None:
        grad_tensors.append(
            OnesLike(
                input_list=[out_tensor],
                graph=y_tensors[0]._graph,
            ).output(0),
        )
      else:
        grad_tensors.append(dy_tensor_map[op.id][i])

  if len(ops) > 1:
    input_list = []
    for op in ops:
      input_list.extend(op._outputs)
    op = _VirtualBackprop(input_list=input_list, graph=y_tensors[0].op._graph)
  else:
    op = ops[0]

  expected_backprops = _compute_expected_backprops(op)
  queue = [(op, grad_tensors)]
  cum_grad = dict()

  while len(queue):
    op, dy_tensors = queue.pop(0)

    # Ops without grad functions are treated as constants and hence ignored
    if not hasattr(op, "_grad_func"):
      continue

    # make sure `dy_tensors` matches the output tensors of `op`
    assert len(op._outputs) == len(dy_tensors)
    for y_tensor, dy_tensor in zip(op._outputs, dy_tensors):
      assert y_tensor.shape._compatible_with(dy_tensor.shape)

    # list of input tensors to `op` that expect backpropped gradient
    tensors = [op._input_list[bp_index] for bp_index in op._bp_indices]
    # list of computed gradient tensors w.r.t. input tensors to `op`
    grad_tensors = op._grad_func(dy_tensors)

    assert len(tensors) == len(grad_tensors)
    for tensor, grad_tensor in zip(tensors, grad_tensors):
      assert tensor.shape._compatible_with(grad_tensor.shape)

      if tensor.op.id not in cum_grad:
        cum_grad[tensor.op.id] = defaultdict(list)
      cum_grad[tensor.op.id][tensor.tensor_index].append(grad_tensor)

      # when each tensor of an Op has received the expected number of
      # backpropped gradients, compute the full gradient by adding them up,
      # and enque the tuple (op, dy_tensors).
      if all((
          len(expected_backprops[tensor.op.id][tensor_index]) ==
          len(cum_grad[tensor.op.id][tensor_index])
      ) for tensor_index in expected_backprops[tensor.op.id].keys()):

        dy_tensors = []
        for tensor_index in sorted(cum_grad[tensor.op.id].keys()):
          if len(cum_grad[tensor.op.id][tensor_index]) > 1:
            grad_tensor = AddN(
                input_list=cum_grad[tensor.op.id][tensor_index],
                graph=tensor.op._graph,
            ).output(0)
          else:
            grad_tensor = cum_grad[tensor.op.id][tensor_index][0]

          dy_tensors.append(grad_tensor)
          cum_grad[tensor.op.id][tensor_index] = [grad_tensor]
        queue.append((tensor.op, dy_tensors))

  dx_tensors = []
  for x_tensor in x_tensors:
    dx_tensors.append(cum_grad[x_tensor.op.id][x_tensor.tensor_index][0])

  y_tensors[0].op.graph.cum_grad = cum_grad
  return dx_tensors


class _VirtualBackprop(Operation):

  def _grad_func(self, in_grad_tensors):
    out_grad_tensors = list(in_grad_tensors)
    return out_grad_tensors

  def _compute_shapes(self):
    from .tensor_shape import TensorShape

    return [
        TensorShape(input_tensor.shape.raw_shape)
        for input_tensor in self._input_list
    ]

  @property
  def num_outputs(self):
    return len(self._input_list)
