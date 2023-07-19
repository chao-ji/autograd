"""Data flow related Operations."""
import numpy as np

from .generic_ops import Const
from .mixins import _ShapeAsIs
from .operation import Operation
from .tensor_shape import TensorShape


class DynamicStitch(Operation):

  def __init__(self, input_list, graph=None, accumulate=False, name=None):
    super(DynamicStitch, self).__init__(
        graph=graph, name=name, input_list=input_list
    )
    self._accumulate = accumulate

  def _run(self, *inputs_list):
    size = len(inputs_list) // 2
    indices, data = inputs_list[:size], inputs_list[size:]

    data = np.concatenate([
        data[i].reshape((-1,) + data[i].shape[indices[i].ndim:])
        for i in range(len(data))
    ])
    indices = np.concatenate([indices[i].ravel() for i in range(len(indices))])

    outputs = np.zeros((indices.max() + 1,) + data.shape[1:], dtype="float32")
    for i, ind in enumerate(indices):
      if self._accumulate:
        outputs[ind] += data[i]
      else:
        outputs[ind] = data[i]
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      size = len(self._input_list) // 2
      out_grad_tensors = []
      for i, (indices, params) in enumerate(
          zip(self._input_list[:size], in_grad_tensors * size)
      ):
        bp_data = Gather(
            input_list=[
                params, indices,
                Const(value=np.asarray(0, dtype="int32")).output(0)
            ]
        )
        out_grad_tensors.append(bp_data.output(0))

    return out_grad_tensors

  def _get_bp_indices(self):
    size = len(self._input_list) // 2
    bp_indices = set(range(size, size * 2))
    return bp_indices

  def _compute_shapes(self):
    # validation
    assert len(self._input_list) % 2 == 0

    size = len(self._input_list) // 2
    constant = None
    constant_ndims = None
    for indices, data in zip(self._input_list[:size], self._input_list[size:]):
      if indices.shape.level > 0 and data.shape.level > 0:
        assert data.shape.ndims >= indices.shape.ndims
        if constant_ndims is None:
          constant_ndims = data.shape.ndims - indices.shape.ndims
        else:
          assert data.shape.ndims - indices.shape.ndims == constant_ndims

        if indices.shape.level == 2 and data.shape.level == 2:
          if constant is None:
            constant = TensorShape(data.shape.raw_shape[-constant_ndims:])
          else:
            assert constant._compatible_with(data.shape[-constant_ndims:])
            constant._merge(data.shape[-constant_ndims:])

    # compute shapes
    if constant is not None:
      return [TensorShape((None,) + constant.raw_shape)]
    elif constant_ndims is not None:
      return [TensorShape([None] * (constant_ndims + 1))]
    else:
      return [TensorShape(None)]


class Gather(Operation):

  def _run(self, params, indices, axis):
    outputs = np.take(params, indices, axis=axis.item())
    return outputs

  def _grad_func(self, in_grad_tensors):
    from .array_ops import Concat, ExpandDims, Fill, Range, Slice, Transpose
    from .math_ops import Add, FloorMod, Sub

    with self._graph.as_default_graph():

      zero_array_tensor = Const(value=np.asarray([0], dtype="int32")).output(0)
      zero_scalar_tensor = Const(value=np.asarray(0, dtype="int32")).output(0)
      one_array_tensor = Const(value=np.asarray([1], dtype="int32")).output(0)
      one_scalar_tensor = Const(value=np.asarray(1, dtype="int32")).output(0)

      op, tensor_index = (
          self._input_list[0].op, self._input_list[0].tensor_index
      )
      mod_tensor = FloorMod(
          input_list=[
              self._input_list[2],
              op.get_rank_tensor(tensor_index=tensor_index)
          ]
      ).output(0)
      op, tensor_index = in_grad_tensors[0].op, in_grad_tensors[0].tensor_index
      range0 = Range(
          input_list=[
              mod_tensor,
              op.get_rank_tensor(tensor_index=tensor_index), one_scalar_tensor
          ]
      )
      range1 = Range(
          input_list=[zero_scalar_tensor, mod_tensor, one_scalar_tensor]
      )
      perm = Concat(
          input_list=[
              zero_scalar_tensor,
              range0.output(0),
              range1.output(0),
          ]
      )
      transpose = Transpose(input_list=[in_grad_tensors[0], perm.output(0)])
      ds = DynamicStitch(
          accumulate=True,
          input_list=[self._input_list[1],
                      transpose.output(0)]
      )

      rank_tensor = ds.get_rank_tensor(tensor_index=0)
      sub_tensor = Sub(input_list=[rank_tensor, mod_tensor]).output(0)
      range2 = Range(input_list=[sub_tensor, rank_tensor, one_scalar_tensor])
      range3 = Range(
          input_list=[zero_scalar_tensor, sub_tensor, one_scalar_tensor]
      )

      perm1 = Concat(
          input_list=[
              zero_scalar_tensor,
              range2.output(0),
              range3.output(0),
          ]
      )

      transpose1 = Transpose(input_list=[ds.output(0), perm1.output(0)])

      shape_tensor = transpose1.get_shape_tensor(tensor_index=0)
      ed_tensor = ExpandDims(input_list=[mod_tensor, zero_scalar_tensor]
                            ).output(0)

      slice0 = Slice(input_list=[shape_tensor, ed_tensor, one_array_tensor])

      op, tensor_index = self._input_list[0].op, self._input_list[0
                                                                 ].tensor_index
      slice1 = Slice(
          input_list=[
              op.get_shape_tensor(tensor_index=tensor_index), ed_tensor,
              one_array_tensor
          ]
      )

      sub = Sub(input_list=[slice1.output(0), slice0.output(0)])

      slice2 = Slice(input_list=[shape_tensor, zero_array_tensor, ed_tensor])

      slice3 = Slice(
          input_list=[
              shape_tensor,
              Add(input_list=[ed_tensor, one_array_tensor]).output(0),
              Const(value=np.asarray([-1], dtype="int32")).output(0)
          ]
      )

      concat = Concat(
          input_list=[
              zero_scalar_tensor,
              slice2.output(0),
              sub.output(0),
              slice3.output(0)
          ]
      )

      fill = Fill(input_list=[concat.output(0), zero_scalar_tensor])

      concat1 = Concat(
          input_list=[mod_tensor,
                      transpose1.output(0),
                      fill.output(0)]
      )

      out_grad_tensors = [concat1.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    if self._input_list[2].shape.level > 0:
      assert self._input_list[2].shape.ndims == 0

    # compute shapes
    if self._input_list[0].shape.level > 0 and self._input_list[
        1].shape.level > 0:
      if (
          self._input_list[0].shape.level == 2 and
          self._input_list[1].shape.level == 2 and
          hasattr(self._input_list[2].op, "_value")
      ):
        axis = self._input_list[2].op._value.item()
        params_shape = list(self._input_list[0].shape.raw_shape)
        indices_shape = list(self._input_list[1].shape.raw_shape)

        raw_shape = params_shape[:axis] + indices_shape + params_shape[axis +
                                                                       1:]
        return [TensorShape(raw_shape)]
      else:
        return [
            TensorShape([None] * (
                self._input_list[0].shape.ndims +
                self._input_list[1].shape.ndims - 1
            ))
        ]
    else:
      return [TensorShape(None)]


class BroadcastTo(Operation):
  """"""

  def _run(self, inputs, target_shape):
    shape = np.pad(
        inputs.shape, [len(target_shape) - len(inputs.shape), 0],
        constant_values=1
    )
    multiples = np.where(shape != target_shape, target_shape, 1)
    outputs = np.tile(inputs, multiples)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from .array_ops import Reshape
    from .math_ops import BroadcastGradientArgs, Sum

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0
                                                                 ].tensor_index
      shape_tensor = op.get_shape_tensor(tensor_index=tensor_index)

      bga = BroadcastGradientArgs(
          input_list=[shape_tensor, self._input_list[1]],
      )
      sum0 = Sum(input_list=[in_grad_tensors[0], bga.output(0)])

      bp_inputs = Reshape(input_list=[sum0.output(0), shape_tensor])
      out_grad_tensors = [bp_inputs.output(0)]

    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    if hasattr(self._input_list[1].op, "_value"):
      target_shape = self._input_list[1].op._value
      assert target_shape.ndim == 1 and (target_shape > 0).all()
      if self._input_list[0].shape.level > 0:
        assert all([
            x is None or x == 1 or x == y for x, y in
            zip(self._input_list[0].shape[::-1], target_shape[::-1])
        ])

    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims == 1
      if self._input_list[1].shape.level == 2:
        orig_ndims = self._input_list[0].shape.ndims
        assert orig_ndims is None or orig_ndims <= self._input_list[1].shape[0]

    # compute shapes
    if hasattr(self._input_list[1].op, "_value"):
      return [TensorShape(self._input_list[1].op._value.tolist())]
    elif self._input_list[1].shape.level == 2:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


class Select(Operation):

  def _run(self, condition, x, y):
    outputs = np.where(condition, x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from .array_ops import Reshape
    from .math_ops import BroadcastGradientArgs, Sum

    with self._graph.as_default_graph():

      op_x = self._input_list[1].op
      tensor_index_x = self._input_list[1].tensor_index
      op_y = self._input_list[2].op
      tensor_index_y = self._input_list[2].tensor_index

      shape_tensor = op_x.get_shape_tensor(tensor_index=tensor_index_x)
      shape1_tensor = op_y.get_shape_tensor(tensor_index=tensor_index_y)
      shape2_tensor = self.get_shape_tensor(tensor_index=0)
      select = Select(
          input_list=[
              self._input_list[0], in_grad_tensors[0],
              Const(value=np.asarray(0, dtype="float32")).output(0)
          ]
      )
      select1 = Select(
          input_list=[
              self._input_list[0],
              Const(value=np.asarray(0, dtype="float32")).output(0),
              in_grad_tensors[0]
          ]
      )
      bga = BroadcastGradientArgs(input_list=[shape_tensor, shape2_tensor])
      bga1 = BroadcastGradientArgs(input_list=[shape1_tensor, shape2_tensor])
      sum0 = Sum(input_list=[select.output(0), bga.output(0)])
      sum1 = Sum(input_list=[select1.output(0), bga1.output(0)])
      bp_x = Reshape(input_list=[sum0.output(0), shape_tensor])
      bp_y = Reshape(input_list=[sum1.output(0), shape1_tensor])
      out_grad_tensors = [bp_x.output(0), bp_y.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [1, 2]

  def _compute_shapes(self):
    # validation
    c_shape = self._input_list[0].shape
    x_shape = self._input_list[1].shape
    y_shape = self._input_list[2].shape

    assert (
        c_shape._broadcastable_with(x_shape) and
        c_shape._broadcastable_with(y_shape) and
        x_shape._broadcastable_with(y_shape)
    )

    # compute shapes
    if c_shape.level > 0 and x_shape.level > 0 and y_shape.level > 0:
      max_ndims = max(c_shape.ndims, x_shape.ndims, y_shape.ndims)
      c_shape = [None] * (max_ndims - c_shape.ndims) + list(c_shape.raw_shape)
      x_shape = [None] * (max_ndims - x_shape.ndims) + list(x_shape.raw_shape)
      y_shape = [None] * (max_ndims - y_shape.ndims) + list(y_shape.raw_shape)

      raw_shape = []
      for c, x, y in zip(c_shape, x_shape, y_shape):
        size = set([c, x, y])
        if len(size) == 1 and None in size:
          raw_shape.append(None)
        else:
          size = size - set([None])
          if len(size) == 1 and 1 in size:
            raw_shape.append(1)
          else:
            size = size - set([1])
            raw_shape.append(list(size)[0])
      return [TensorShape(raw_shape)]
    else:
      return [TensorShape(None)]


class StopGradient(Operation, _ShapeAsIs):

  def _run(self, inputs):
    outputs = inputs
    return outputs


class Identity(Operation, _ShapeAsIs):

  def _run(self, inputs):
    outputs = inputs
    return outputs

  def _grad_func(self, in_grad_tensors):
    out_grad_tensors = in_grad_tensors
    return out_grad_tensors
