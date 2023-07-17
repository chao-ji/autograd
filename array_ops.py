"""Operations on multi-dimensional arrays."""
import numpy as np

from .operation import Operation

from .generic_ops import Const
from .math_ops import Sum, Mean
from .tensor_shape import TensorShape
from .mixins import _ShapeAsIs, _TensorShapeAsInput


class Reshape(Operation):

  def _run(self, inputs, shape):
    outputs = np.reshape(inputs, shape.astype("int32"))
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Reshape(
          input_list=[
              in_grad_tensors[0],
              op.get_shape_tensor(tensor_index=tensor_index)
          ]
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    if hasattr(self._input_list[1].op, "_value"):
      target_shape = self._input_list[1].op._value
      assert (target_shape < 0).sum() <= 1
      if (target_shape >= 1).all():
        assert self._input_list[0].shape._partial_size() <= np.prod(target_shape).item()
    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims <= 1

    # compute shapes
    if hasattr(self._input_list[1].op, "_value"):
      target_shape = self._input_list[1].op._value
      if target_shape.ndim == 0 and target_shape == -1:
        return [TensorShape([None])]
      elif (target_shape >= 0).all():
        return [TensorShape(target_shape)]
      elif self._input_list[0].shape.level == 2 and -1 in self._input_list[1].op._value:
        # infer size when target shape contains -1
        index = self._input_list[1].op._value.tolist().index(-1)
        new_size = np.prod(self._input_list[0].shape.raw_shape) / np.prod(self._input_list[1].op._value.tolist())
        raw_shape = list(self._input_list[1].op._value.tolist())
        raw_shape[index] = -int(new_size)
        return [TensorShape(raw_shape)]
      else:
        return [TensorShape([None] * self._input_list[1].shape[0])]
    elif self._input_list[1].shape.level == 2:
      return [TensorShape([None] * self._input_list[1].shape[0])]
    else:
      return [TensorShape(None)]


class Transpose(Operation):

  def _run(self, inputs, perm):
    outputs = np.transpose(inputs, perm)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      invert_perm = InvertPermutation(input_list=[self._input_list[1]])
      bp_inputs = Transpose(input_list=[in_grad_tensors[0], invert_perm.output(0)])
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    if hasattr(self._input_list[1].op, "_value"):
      assert self._input_list[1].op._value.ndim == 1
      perm = self._input_list[1].op._value.tolist()
      assert len(set(perm)) == max(perm) + 1 and min(perm) == 0
      if self._input_list[0].shape.level > 0:
        assert len(perm) == self._input_list[0].shape.ndims

    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims == 1
    if self._input_list[0].shape.level > 0 and self._input_list[1].shape.level == 2:
      assert self._input_list[0].shape.ndims == self._input_list[1].shape[0]

    # compute shapes
    if hasattr(self._input_list[1].op, "_value"):
      return [TensorShape([self._input_list[0].shape[p] for p in perm])]
    elif self._input_list[0].shape.level != 0:
      ndims = self._input_list[0].shape.ndims
      return [TensorShape([None] * ndims)]
    elif self._input_list[1].shape.level == 2:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


class InvertPermutation(Operation, _ShapeAsIs):

  def _run(self, perm):
    outputs = np.argsort(perm)
    return outputs


class Range(Operation):

  def _run(self, start, limit, delta):
    outputs = np.arange(start, limit, delta).astype("int32")
    return outputs

  def _compute_shapes(self):
    # validation
    for tensor in self._input_list:
      if tensor.shape.level > 0:
        assert tensor.shape.ndims == 0

    # compute shapes
    return [TensorShape([None])]


class Pack(Operation):

  def __init__(self, axis, input_list, graph=None, name=None):
    self._axis = axis
    super(Pack, self).__init__(graph=graph, input_list=input_list, name=name)

  def _run(self, *input_tensor_values):
    outputs = np.stack(input_tensor_values, axis=self._axis)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = Unpack(num=len(self._input_list), axis=self._axis, input_list=[in_grad_tensors[0]])
      out_grad_tensors = [bp_inputs.output(i) for i in range(len(self._input_list))]
    return out_grad_tensors

  def _compute_shapes(self):
    # validation
    for tensor in self._input_list[1:]:
      assert self._input_list[0].shape._compatible_with(tensor.shape)

    ndims = None
    for tensor in self._input_list:
      if tensor.shape.level > 0:
        if ndims is None:
          ndims = tensor.shape.ndims
        else:
          assert ndims == tensor.shape.ndims
    if ndims is not None:
      assert -(ndims + 1) <= self._axis < ndims + 1

    # compute shapes
    if ndims is None:
      return [TensorShape(None)]
    else:
      shape = TensorShape([None] * ndims)
      for tensor in self._input_list:
        shape._merge(tensor.shape)
      raw_shape = list(shape.raw_shape)

      axis = self._axis if ndims == 0 else self._axis % (ndims + 1)
      raw_shape = raw_shape[:axis] + [len(self._input_list)] + raw_shape[axis:]
      return [TensorShape(raw_shape)]


class Unpack(Operation):

  def __init__(self, num, axis, input_list, graph=None, name=None):
    self._num = num
    self._axis = axis
    super(Unpack, self).__init__(graph=graph, input_list=input_list, name=name)

  def _run(self, inputs):
    axis_size = inputs.shape[self._axis]
    outputs = np.split(inputs, axis_size, axis=self._axis)
    outputs = [np.squeeze(output, axis=self._axis) for output in outputs]
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = Pack(
          axis=self._axis,
          input_list=in_grad_tensors
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  @property
  def num_outputs(self):
    return self._num

  def _compute_shapes(self):
    # validation
    if self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      axis = self._axis if ndims == 0 else self._axis % ndims
      if self._input_list[0].shape[axis] is not None:
        assert self._input_list[0].shape[axis] == self._num
      assert -ndims <= axis < ndims

    # compute shapes
    if self._input_list[0].shape.level == 0:
      return [TensorShape(None) for _ in range(self._num)]
    else:
      raw_shape = list(self._input_list[0].shape.raw_shape)
      raw_shape = raw_shape[:axis] + raw_shape[axis + 1:]
      return [TensorShape(raw_shape) for _ in range(self._num)]


class Tile(Operation):

  def _run(self, inputs, multiples):
    outputs = np.tile(inputs, multiples)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      pack = Pack(
          axis=0,
          input_list=[
              self._input_list[1],
              op.get_shape_tensor(tensor_index=tensor_index)
          ]
      )
      transpose = Transpose(
          input_list=[
            pack.output(0),
            Const(value=np.asarray((1, 0), dtype="int32")).output(0)
          ]
      )
      reshape = Reshape(
          input_list=[
            transpose.output(0),
            Const(value=np.asarray(-1, dtype="int32")).output(0)
          ]
      )
      reshape1 = Reshape(
          input_list=[in_grad_tensors[0], reshape.output(0)]
      )
      reduction_indices = Range(
          input_list=[
              Const(value=np.asarray(0, dtype="int32")).output(0),
              reshape.get_size_tensor(tensor_index=0),
              Const(value=np.asarray(2), dtype="int32").output(0)
          ]
      )
      bp_inputs = Sum(
          input_list=[reshape1.output(0), reduction_indices.output(0)]
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims == 1
      if self._input_list[1].shape.level == 2 and self._input_list[0].shape.level > 0:
        assert self._input_list[1].shape[0] == self._input_list[0].shape.ndims
    if hasattr(self._input_list[1].op, "_value"):
      assert self._input_list[1].op._value.ndim == 1
      assert (self._input_list[1].op._value >= 1).all()
      if self._input_list[0].shape.level > 0:
        assert self._input_list[1].op._value.shape[0] == self._input_list[0].shape.ndims

    # compute shapes
    if hasattr(self._input_list[1].op, "_value") and self._input_list[0].shape.level > 0:
      raw_shape = []
      for size, multiple in zip(self._input_list[0].shape, self._input_list[1].op._value):
        raw_shape.append(None if size is None else size * multiple)
      return [TensorShape(raw_shape)]
    elif self._input_list[0].shape.level == 0:
      if self._input_list[1].shape.level > 0 and self._input_list[1].shape[0] is not None:
        ndims = self._input_list[1].shape[0]
        return [TensorShape([None] * ndims)]
      else:
        return [TensorShape(None)]
    else:
      ndims = self._input_list[0].shape.ndims
      return [TensorShape([None] * ndims)]


class StridedSlice(Operation):

  def _run(self, inputs, begin, end, strides):
    shape = inputs.shape
    begin = begin % shape
    end = np.where(end < 0, end % shape, end)
    slices = np.stack([begin, end, strides]).T
    slice_indices_list = list(
        map(lambda s: np.arange(s[1][0], s[1][1], s[1][2]) *
            np.prod(shape[s[0]+1:]).astype("int"), enumerate(slices)
        )
    )
    slice_indices = np.meshgrid(*slice_indices_list, indexing="ij")
    offsets = np.stack(slice_indices).reshape(len(shape), -1).T
    indices = offsets.sum(axis=1)

    outputs = inputs.ravel()[indices].reshape(
        tuple(map(len, slice_indices_list))
    )
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      ssg = StridedSliceGrad(
          input_list=[
              op.get_shape_tensor(tensor_index=tensor_index)
          ] + self._input_list[1:4] + in_grad_tensors
      )
      out_grad_tensors = [ssg.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    # validation
    ndims = None
    if self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims

    for tensor in self._input_list[1:]:
      if tensor.shape.level > 0:
        assert tensor.shape.ndims == 1
        if tensor.shape.level == 2:
          if ndims is not None:
            assert ndims == tensor.shape[0]
          else:
            ndims = tensor.shape[0]

    # compute shapes
    if (self._input_list[0].shape.level > 0 and
        hasattr(self._input_list[1].op, "_value") and
        hasattr(self._input_list[2].op, "_value") and
        hasattr(self._input_list[3].op, "_value")):
      raw_shape = []
      for i, b, e, s in zip(
          self._input_list[0].shape.raw_shape,
          self._input_list[1].op._value.tolist(),
          self._input_list[2].op._value.tolist(),
          self._input_list[3].op._value.tolist(),
        ):
        if i is None:
          raw_shape.append(None)
        else:
          b = b % i
          e = np.where(e < 0, e % i, e)

          r = np.arange(b, e, s)

          #raw_shape.append(min(r.max() + 1, i) - max(r.min(), 0))
          raw_shape.append(len(set(r.tolist()).intersection(set(np.arange(0, i).tolist()))))

      return [TensorShape(raw_shape)]
    elif self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      return [TensorShape([None] * ndims)]
    elif self._input_list[1].shape.level == 2:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[2].shape.level == 2:
      ndims = self._input_list[2].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[3].shape.level == 2:
      ndims = self._input_list[3].shape[0]
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


class StridedSliceGrad(Operation):

  def _run(self, shape, begin, end, strides, grads):
    begin = begin % shape
    end = np.where(end < 0, end % shape, end)
    slices = np.stack([begin, end, strides]).T
    slice_indices_list = list(
        map(lambda s: np.arange(s[1][0], s[1][1], s[1][2]) *
            np.prod(shape[s[0]+1:]).astype("int"), enumerate(slices)
        )
    )

    slice_indices = np.meshgrid(*slice_indices_list, indexing="ij")
    offsets = np.stack(slice_indices).reshape(len(shape), -1).T
    indices = offsets.sum(axis=1)

    inputs_grads = np.zeros(shape).ravel()
    inputs_grads[indices] = grads.ravel()
    inputs_grads = inputs_grads.reshape(shape)

    return inputs_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_grads = StridedSlice(
          input_list=in_grad_tensors + self._input_list[1:4]
      )
      out_grad_tensors = [bp_grads.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [4]

  def _compute_shapes(self):
    # validation
    ndims = None
    if self._input_list[4].shape.level > 0:
      ndims = self._input_list[4].shape.ndims

    for tensor in self._input_list[:4]:
      if tensor.shape.level > 0:
        assert tensor.shape.ndims == 1
        if tensor.shape.level == 2:
          if ndims is not None:
            assert ndims == tensor.shape[0]
          else:
            ndims = tensor.shape[0]

    # compute shapes
    if hasattr(self._input_list[0].op, "_value"):
      return [TensorShape(self._input_list[0].op._value.astype("int32").tolist())]
    elif self._input_list[0].shape.level == 2:
      ndims = self._input_list[0].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[1].shape.level == 2:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[2].shape.level == 2:
      ndims = self._input_list[2].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[3].shape.level == 2:
      ndims = self._input_list[3].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_shape[4].shape.level > 0:
      ndims = self._input_shape[4].shape.ndims
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


class Slice(Operation):

  def _run(self, inputs, begin, size):
    shape = inputs.shape
    #begin = begin % shape
    # 0 <= begin < shape
    size = np.where(size == -1, np.asarray(shape) - begin, size)

    end = begin + size
    #end = np.where(end < 0, end % shape, end)
    slices = np.stack([begin, end]).T
    slice_indices_list = list(
        map(lambda s: np.arange(s[1][0], s[1][1]) *
            np.prod(shape[s[0]+1:]).astype("int"), enumerate(slices)
        )
    )
    slice_indices = np.meshgrid(*slice_indices_list, indexing="ij")
    offsets = np.stack(slice_indices).reshape(len(shape), -1).T
    indices = offsets.sum(axis=1)

    outputs = inputs.ravel()[indices].reshape(
        tuple(map(len, slice_indices_list))
    )
    return outputs

  def _grad_func(self, in_grad_tensors):
    from .math_ops import Sub

    with self._graph.as_default_graph():

      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      sub = Sub(input_list=[
              op.get_shape_tensor(tensor_index=tensor_index),
              self.get_shape_tensor(tensor_index=0),
          ],
      )
      sub1 = Sub(
          input_list=[sub.output(0), self._input_list[1]],
      )
      pack = Pack(
          axis=0,
          input_list=[
              op.get_rank_tensor(tensor_index=tensor_index),
              Const(value=np.asarray(1, dtype="int32")).output(0)
          ],
      )
      reshape = Reshape(input_list=[self._input_list[1], pack.output(0)])
      reshape1 = Reshape(input_list=[sub1.output(0), pack.output(0)])
      concat = Concat(
          input_list=[
              Const(value=np.asarray(1, dtype="int32")).output(0),
              reshape.output(0), reshape1.output(0)
          ],
      )

      bp_inputs = Pad(input_list=in_grad_tensors+[concat.output(0)])
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    #return [TensorShape(None)]
    # validation
    ndims = None
    if self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims

    for tensor in self._input_list[1:]:
      if tensor.shape.level > 0:
        assert tensor.shape.ndims == 1
        if tensor.shape.level == 2:
          if ndims is not None:
            assert ndims == tensor.shape[0]
          else:
            ndims = tensor.shape[0]

    if hasattr(self._input_list[1].op, "_value"):
      assert (self._input_list[1].op._value >= 0).all()
    if hasattr(self._input_list[2].op, "_value"):
      assert (self._input_list[2].op._value >= -1).all()

    # compute shapes
    if (self._input_list[0].shape.level > 0 and
        hasattr(self._input_list[1].op, "_value") and
        hasattr(self._input_list[2].op, "_value")):
      raw_shape = []
      for i, b, s in zip(
          self._input_list[0].shape,
          self._input_list[1].op._value.tolist(),
          self._input_list[2].op._value.tolist()
        ):
        if i is None:
          raw_shape.append(None)
        else:
          raw_shape.append(i - b if s == -1 else s)
      return [TensorShape(raw_shape)]
    elif self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      return [TensorShape([None] * ndims)]
    elif self._input_list[1].shape.level == 2:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    elif self._input_list[2].shape.level == 2:
      ndims = self._input_list[2].shape[0]
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


    if self._input_list[0].shape.ndims is None:
      return TensorShape(None)
    elif (hasattr(self._input_list[1].op, "_value") and
          hasattr(self._input_list[2].op, "_value")
      ):
      raw_shape = []
      for i, b, s in zip(
          self._input_list[0].shape.raw_shape,
          self._input_list[1].op._value.astype("int32").tolist(),
          self._input_list[2].op._value.astype("int32").tolist(),
        ):
        if i is None:
          raw_shape.append(None if s == -1 else s)
        else:
          raw_shape.append(i - b if s == -1 else s)
      return TensorShape(raw_shape)
    else:
      return TensorShape([None] * self._input_list[0].shape.ndims)


class Concat(Operation):

  def _run(self, axis, *input_tensor_values):
    outputs = np.concatenate(input_tensor_values, axis=axis)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from .math_ops import FloorMod

    with self._graph.as_default_graph():
      shapes = [
          arg.op.get_shape_tensor(tensor_index=arg.tensor_index)
              for arg in self._input_list[1:]
      ]
      op, tensor_index = self._input_list[1].op, self._input_list[1].tensor_index
      mod = FloorMod(
          input_list=[self._input_list[0]] + [
              op.get_rank_tensor(tensor_index=tensor_index)
          ],
      )
      offset = ConcatOffset(
          input_list=[mod.output(0)]+shapes,
      )
      out_grad_tensors = []
      for i in range(len(self._input_list) - 1):
        out_grad_tensors.append(
            Slice(input_list=[in_grad_tensors[0], offset.output(i), shapes[i]],
            ).output(0)
        )
    return out_grad_tensors

  def _get_bp_indices(self):
    backprop_indices = list(range(1, len(self._input_list)))
    return backprop_indices

  def _compute_shapes(self):
    # validation
    diff_axes = set()
    for tensor in self._input_list[2:]:
      if self._input_list[1].shape.level > 0 and tensor.shape.level > 0:
        assert self._input_list[1].shape.ndims == tensor.shape.ndims
        diff_axes.update(self._input_list[1].shape._diff_at(tensor.shape))
    assert len(diff_axes) <= 1

    ndims = set([tensor.shape.ndims for tensor in self._input_list[1:] if tensor.shape.level > 0])
    assert len(ndims) <= 1
    ndims = None if len(ndims) == 0 else list(ndims)[0]

    if hasattr(self._input_list[0].op, "_value"):
      axis = self._input_list[0].op._value.item()
      if ndims is not None:
        axis = axis if ndims == 0 else axis % ndims
      if len(diff_axes) == 1:
        assert list(diff_axes)[0] == axis

    # compute shapes
    if ndims is None:
      return [TensorShape(None)]
    else:
      if hasattr(self._input_list[0].op, "_value"):
        shape = TensorShape([None] * ndims)
        for tensor in self._input_list[1:]:
          shape._merge(tensor.shape, skip=[axis])

        size = []
        for tensor in self._input_list[1:]:
          if tensor.shape.level == 0 or tensor.shape[axis] is None:
            size = None
            break
          else:
            size.append(tensor.shape[axis])
        raw_shape = list(shape._raw_shape)
        raw_shape[axis] = None if size is None else sum(size)

        return [TensorShape(raw_shape)]
      else:
        return [TensorShape([None] * ndims)]


class ConcatOffset(Operation):

  def _run(self, concat_dim, *shape):
    shapes = np.cumsum([0] + [s[concat_dim] for s in shape[:-1]])
    shapes = np.pad(np.expand_dims(shapes, axis=1), [[0, 0], [concat_dim, len(shape[0]) - concat_dim - 1]])
    shapes = [s for s in shapes]
    return shapes

  @property
  def num_outputs(self):
    return len(self._input_list) - 1

  def _compute_shapes(self):
    # validation
    diff_axes = set()
    for tensor in self._input_list[2:]:
      if self._input_list[1].shape.level > 0 and tensor.shape.level > 0:
        assert self._input_list[1].shape.ndims == tensor.shape.ndims
        diff_axes.update(self._input_list[1].shape._diff_at(tensor.shape))
    assert len(diff_axes) <= 1

    ndims = set([tensor.shape.ndims for tensor in self._input_list[1:] if tensor.shape.level > 0])
    assert len(ndims) <= 1
    ndims = None if len(ndims) == 0 else list(ndims)[0]

    if hasattr(self._input_list[0].op, "_value"):
      axis = self._input_list[0].op._value.item()
      if ndims is not None:
        axis = axis if ndims == 0 else axis % ndims
      if len(diff_axes) == 1:
        assert list(diff_axes)[0] == axis

    # compute shapes
    return [tensor.shape for tensor in self._input_list[1:]]


class Pad(Operation):

  def __init__(self, input_list, graph=None, name=None, constant_values=0):
    super(Pad, self).__init__(graph=graph, input_list=input_list, name=name)
    self._constant_values = constant_values

  def _run(self, inputs, paddings):
    outputs = np.pad(inputs, paddings, constant_values=self._constant_values)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      pack = Pack(
          axis=0,
          input_list=[
              op.get_rank_tensor(tensor_index=tensor_index),
              Const(value=np.asarray(1, dtype="int32")).output(0)
          ],
      )

      slice0 = Slice(
          input_list=[
              self._input_list[1],
              Const(value=np.asarray((0, 0), dtype="int32")).output(0),
              pack.output(0)
          ],
      )
      reshape = Reshape(
          input_list=[
              slice0.output(0),
              Const(value=np.asarray(-1, dtype="int32")).output(0)
          ],
      )

      bp_inputs = Slice(
          input_list=[
              in_grad_tensors[0],
              reshape.output(0),
              op.get_shape_tensor(tensor_index=tensor_index),
          ],
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):

    # validation
    if hasattr(self._input_list[1].op, "_value"):
      paddings = self._input_list[1].op._value
      assert paddings.ndim == 2 and paddings.shape[1] == 2 and (paddings >= 0).all()
      if self._input_list[0].shape.level > 0:
        assert paddings.shape[0] == self._input_list[0].shape.ndims

    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims == 2
    if (self._input_list[0].shape.level > 0 and
        self._input_list[1].shape.level > 0 and
        self._input_list[1].shape[0] is not None
      ):
      assert self._input_list[0].shape.ndims == self._input_list[1].shape[0]

    # compute shapes
    if self._input_list[0].shape.level > 0 and hasattr(self._input_list[1].op, "_value"):
      raw_shape = []
      for i, padding in zip(self._input_list[0].shape, paddings):
        if i is None:
          raw_shape.append(None)
        else:
          raw_shape.append(i + padding[0] + padding[1])
      return [TensorShape(raw_shape)]
    elif self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      return [TensorShape([None] * ndims)]
    elif self._input_list[1].shape.level > 0 and self._input_list[1].shape[0] is not None:
      ndims = self._input_list[1].shape[0]
      return [TensorShape([None] * ndims)]
    else:
      return [TensorShape(None)]


class ExpandDims(Operation):

  def _run(self, inputs, axis):
    outputs = np.expand_dims(inputs, axis.item())
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Reshape(
          input_list=[
              in_grad_tensors[0], op.get_shape_tensor(tensor_index=tensor_index),
          ]
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]

  def _compute_shapes(self):
    #return [TensorShape(None)]

    # validation
    if self._input_list[1].shape.level > 0:
      assert self._input_list[1].shape.ndims <= 1
      if self._input_list[1].shape.ndims == 1 and self._input_list[1].shape[0] is not None:
        assert self._input_list[1].shape[0] == 1

    axis = None
    if hasattr(self._input_list[1].op, "_value"):
      axis = self._input_list[1].op._value
      assert axis.size == 1
      axis = axis.item()

    ndims = None
    raw_shape = None
    if self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      raw_shape = list(self._input_list[0].shape.raw_shape)
      if axis is not None:
        assert -ndims - 1 <= axis <= ndims

    # compute shapes
    if axis is not None and raw_shape is not None:
      axis = axis % (ndims + 1)
      return [TensorShape(raw_shape[:axis] + [1] + raw_shape[axis:])]
    elif raw_shape is not None:
      return [TensorShape([None] * (ndims + 1))]
    else:
      return [TensorShape(None)]


class Squeeze(Operation):

  def __init__(self, input_list, graph=None, axis=[], name=None):
    if not len(axis):
      self._axis = None
    else:
      self._axis = tuple(axis)
    super(Squeeze, self).__init__(graph=graph, input_list=input_list, name=name)

  def _run(self, inputs):
    outputs = np.squeeze(inputs, axis=self._axis)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Reshape(
          input_list=[
              in_grad_tensors[0], op.get_shape_tensor(tensor_index=tensor_index),
          ],
      )
      out_grad_tensors = [bp_inputs.output(0)]
    return out_grad_tensors

  def _compute_shapes(self):

    # validation
    ndims = None
    if self._input_list[0].shape.level > 0:
      ndims = self._input_list[0].shape.ndims
      if self._axis is not None:
        axis = np.asarray(self._axis)
        assert np.logical_and(-ndims <= axis, axis < ndims).all()
        axis = axis % ndims
        assert all([i not in axis or s == 1 or s is None for i, s in enumerate(self._input_list[0].shape)])

    # compute shapes
    if ndims is not None:
      if self._input_list[0].shape.level == 2:
        if self._axis is None:
          raw_shape = [s for s in self._input_list[0].shape if s != 1]
        else:
          axis = np.asarray(self._axis) % ndims
          raw_shape = [s for i, s in enumerate(self._input_list[0].shape) if i not in axis]
        return [TensorShape(raw_shape)]
      else:
        if self._axis is None:
          raw_shape = None
        else:
          axis = np.asarray(self._axis) % ndims
          raw_shape = [s for i, s in enumerate(self._input_list[0].shape) if i not in axis]
        return [TensorShape(raw_shape)]
    else:
      return [TensorShape(None)]


class Fill(Operation, _TensorShapeAsInput):

  def _run(self, dims, value):
    outputs = np.ones(dims, dtype=value.dtype) * value
    return outputs

  def _compute_shapes(self):
    # validation
    if hasattr(self._input_list[0].op, "_value"):
      target_shape = self._input_list[0].op._value
      assert (target_shape > 0).all()
    if self._input_list[0].shape.level > 0:
      assert self._input_list[0].shape.ndims == 1

    # compute shapes
    if hasattr(self._input_list[0].op, "_value"):
      target_shape = self._input_list[0].op._value.tolist()
      return [TensorShape(target_shape)]
    elif self._input_list[0].shape.level == 2:
      return [TensorShape([None] * self._input_list[0].shape[0])]
    else:
      return [TensorShape(None)]


class ListDiff(Operation):

  def _run(self, x, y):
    outputs = np.setdiff1d(x, y)
    return outputs

  def _compute_shapes(self):
    return [TensorShape([None])]
