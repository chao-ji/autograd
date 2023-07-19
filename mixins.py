import numpy as np

from .tensor_shape import TensorShape


class _BinaryOp(object):

  def _compute_shapes(self):
    # validation
    assert self._input_list[0].shape._broadcastable_with(
        self._input_list[1].shape
    )

    # compute shapes
    x_shape = self._input_list[0].shape
    y_shape = self._input_list[1].shape

    if x_shape.ndims is None or y_shape.ndims is None:
      return [TensorShape(None)]

    if x_shape.ndims == 0:
      return [TensorShape(y_shape._raw_shape)]
    elif y_shape.ndims == 0:
      return [TensorShape(x_shape._raw_shape)]

    shape = []
    min_ndims = min(x_shape.ndims, y_shape.ndims)
    for i in range(min_ndims):
      dim_x = x_shape._raw_shape[-1 - i]
      dim_y = y_shape._raw_shape[-1 - i]
      if dim_x is not None and dim_y is not None:
        if dim_x == dim_y:
          shape.append(dim_x)
        elif dim_x == 1:
          shape.append(dim_y)
        elif dim_y == 1:
          shape.append(dim_x)
        else:
          raise ValueError(
              'operands x(%s) and y(%s) have incompatible shapes for '
              'broadcasting.' % (x_shape, y_shape)
          )
      elif dim_x is not None:
        shape.append(None if dim_x == 1 else dim_x)
      elif dim_y is not None:
        shape.append(None if dim_y == 1 else dim_y)
      else:
        shape.append(None)

    if min_ndims == x_shape.ndims:
      return [TensorShape(list(y_shape._raw_shape[:-min_ndims]) + shape[::-1])]
    else:
      return [TensorShape(list(x_shape._raw_shape[:-min_ndims]) + shape[::-1])]


class _ReductionOp(object):

  def _compute_shapes(self):
    if not (
        self._input_list[0].shape.level > 0 and
        hasattr(self._input_list[1].op, "_value")
    ):
      return [TensorShape(None)]
    else:
      raw_shape = []
      ndims = self._input_list[0].shape.ndims
      reduction_indices = self._input_list[1].op._value.ravel().tolist()
      reduction_indices = [i % ndims for i in reduction_indices]
      for i, s in enumerate(self._input_list[0].shape._raw_shape):
        if i not in reduction_indices:
          raw_shape.append(s)
        elif self._keepdims:
          raw_shape.append(1)
    return [TensorShape(raw_shape)]


class _ShapeAsIs(object):

  def _compute_shapes(self):
    return [
        TensorShape(
            None if tensor.shape.raw_shape is
            None else list(tensor.shape.raw_shape)
        ) for tensor in self._input_list
    ]


class _ScalarShape(object):

  def _compute_shapes(self):
    return [TensorShape([])]


class _PickFirstAmongCompatibleShapes(object):

  def _compute_shapes(self):
    # validation
    for tensor in self._input_list[1:]:
      assert self._input_list[0].shape._compatible_with(tensor.shape)

    # compute shapes
    shape = TensorShape(self._input_list[0].shape.raw_shape)
    for tensor in self._input_list[1:]:
      shape._merge(tensor.shape)
    return [shape]


def _compute_static_spatial_dim_size(
    input_size, kernel_size, stride_size, padding
):
  if input_size is None:
    out_size = None
  else:
    if padding == "SAME":
      out_size = input_size * stride_size
    else:  # padding == 'VALID'
      if kernel_size is None:
        out_size = None
      else:
        out_size = input_size * stride_size + max(kernel_size - stride_size, 0)
  return out_size


class _TensorShapeAsInput(object):

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
