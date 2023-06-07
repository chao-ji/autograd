"""Math related Operations."""
import numpy as np

from operation import Operation
from origin_ops import Const


class BroadcastGradientArgs(Operation):
  """BroadcastGradientArgs"""

  def _run(self, x_shape, y_shape):
    if len(x_shape) > len(y_shape):
      y_shape = np.pad(
          y_shape, [[len(x_shape) - len(y_shape), 0]], constant_values=1
      )
    elif len(y_shape) > len(x_shape):
      x_shape = np.pad(
          x_shape, [[len(y_shape) - len(x_shape), 0]], constant_values=1
      )
    reduction_indices_x = np.where(x_shape == 1)[0]
    reduction_indices_y = np.where(y_shape == 1)[0]
    return reduction_indices_x, reduction_indices_y


class Add(Operation):
  """AddV2."""

  def _run(self, x, y):
    outputs = np.add(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)]
      )

      sum0 = Sum(
          input_list=[in_grad_tensors[0], (bga, 0)]
      )
      sum1 = Sum(
          input_list=[in_grad_tensors[0], (bga, 1)]
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)]
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)]
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class Neg(Operation):
  """Neg"""
  def _run(self, x):
    outputs = np.negative(x)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = Neg(
          input_list=[in_grad_tensors[0]]
      )
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class Sub(Operation):
  """Sub"""
  def _run(self, x, y):
    outputs = np.subtract(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)]
      )

      sum0 = Sum(
          input_list=[in_grad_tensors[0], (bga, 0)]
      )

      neg = Neg(
          input_list=[in_grad_tensors[0]]
      )
      sum1 = Sum(
          input_list=[(neg, 0), (bga, 1)]
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)]
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)]
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class Mul(Operation):
  """Mul"""
  def _run(self, x, y):
    outputs = np.multiply(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)]
      )

      mul = Mul(
          input_list=[in_grad_tensors[0], self._input_list[1]]
      )
      mul1 = Mul(
          input_list=[in_grad_tensors[0], self._input_list[0]]
      )

      sum0 = Sum(
          input_list=[(mul, 0), (bga, 0)]
      )
      sum1 = Sum(
          input_list=[(mul1, 0), (bga, 1)]
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)]
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)]
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class RealDiv(Operation):
  """RealDiv"""
  def _run(self, x, y):
    outputs = np.divide(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      neg = Neg(
          input_list=[self._input_list[0]]
      )
      div = RealDiv(
          input_list=[in_grad_tensors[0], self._input_list[1]]
      )
      div1 = RealDiv(
          input_list=[(neg, 0), self._input_list[1]]
      )
      div2 = RealDiv(
          input_list=[(div1, 0), self._input_list[1]]
      )
      mul = Mul(
          input_list=[(div2, 0), in_grad_tensors[0]]
      )

      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)]
      )
      sum0 = Sum(
          input_list=[(div, 0), (bga, 0)]
      )
      sum1 = Sum(
          input_list=[(mul, 0), (bga, 1)]
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)]
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)]
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class FloorDiv(Operation):
  """FloorDiv"""
  def _run(self, x, y):
    outputs = x // y
    return outputs


class FloorMod(Operation):
  """FloorMod"""
  def _run(self, x, y):
    outputs = x % y
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      div = FloorDiv(
          input_list=self._input_list,
      )
      neg = Neg(
          input_list=[(div, 0)],
      )

      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)]
      )
      mul = Mul(
          input_list=[in_grad_tensors[0], (neg, 0)],
      )

      sum0 = Sum(
          input_list=[in_grad_tensors[0], (bga, 0)]
      )
      sum1 = Sum(
          input_list=[(mul, 0)] + [(bga, 1)]
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)]
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)]
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class Maximum(Operation):
  """Maximum"""
  def _run(self, x, y):
    outputs = np.maximum(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape
    from data_flow_ops import Select

    with self._graph.as_default_graph():
      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      ge = GreaterEqual(
          input_list=[self._input_list[0],
          self._input_list[1]],
      )
      op, tensor_index = in_grad_tensors[0]
      zeros = op.get_zeros_op(tensor_index=tensor_index)

      select = Select(
          input_list=[(ge, 0), in_grad_tensors[0], (zeros, 0)],
      )
      select1 = Select(
          input_list=[(ge, 0), (zeros, 0), in_grad_tensors[0]],
      )
      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)],
      )

      sum0 = Sum(
          input_list=[(select, 0), (bga, 0)],
      )
      sum1 = Sum(
          input_list=[(select1, 0), (bga, 1)],
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)],
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)],
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class Minimum(Operation):
  """Minimum"""
  def _run(self, x, y):
    outputs = np.maximum(x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    pass


class DivNoNan(Operation):
  """DivNoNan"""
  def _run(self, x, y):
    outputs = np.where(np.not_equal(y, 0), np.divide(x, y), 0)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Reshape

    with self._graph.as_default_graph():
      div = DivNoNan(
          input_list=[in_grad_tensors[0], self._input_list[1]],
      )
      neg = Neg(
          input_list=[self._input_list[0]],
      )
      div1 = DivNoNan(
          input_list=[(neg, 0), self._input_list[1]],
      )
      div2 = DivNoNan(
          input_list=[(div1, 0), self._input_list[1]],
      )

      mul = Mul(
          input_list=[in_grad_tensors[0], (div2, 0)],
      )

      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), (shape1, 0)],
      )

      sum0 = Sum(
          input_list=[(div, 0), (bga, 0)],
      )
      sum1 = Sum(
          input_list=[(mul, 0), (bga, 1)],
      )

      bp_x = Reshape(
          input_list=[(sum0, 0), (shape, 0)],
      )
      bp_y = Reshape(
          input_list=[(sum1, 0), (shape1, 0)],
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class AddN(Operation):
  """AddN"""
  def _run(self, *input_tensor_values):
    self._num = len(input_tensor_values)

    outputs = input_tensor_values[0]
    for tensor_value in input_tensor_values[1:]:
      outputs = np.add(outputs, tensor_value)
    return outputs

  def _grad_fun(self, in_grad_tensors):
    with self._graph.as_default_graph():
      out_grad_tensors = in_grad_tensors * self._num
    return out_grad_tensors


class Mean(Operation):

  def __init__(self, input_list, graph=None, keepdims=False, name=None):
    super(Mean, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.mean(
        inputs,
        axis=tuple(reduction_indices.ravel().tolist()),
        keepdims=self._keepdims
    )
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Fill, Range, Reshape
    from data_flow_ops import DynamicStitch, BroadcastTo

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0]
      shape = op.get_shape_op(tensor_index=tensor_index)
      size = shape.get_size_op(tensor_index=0)
      add = Add(input_list=[(size, 0), self._input_list[1]])
      mod = FloorMod(input_list=[(add, 0), (size, 0)])
      shape1 = mod.get_shape_op(tensor_index=0)
      fill = Fill(
          input_list=[
            (shape1, 0),
            (Const(value=np.asarray(1)), 0)
          ],
      )
      range0 = Range(
          input_list=[
              (Const(value=np.asarray(0)), 0),
              (size, 0),
              (Const(value=np.asarray(1)), 0)
          ],
      )
      ds = DynamicStitch(
          input_list=[
              (range0, 0), (mod, 0), (shape, 0), (fill, 0)
          ],
      ) 
      shape3 = self.get_shape_op(tensor_index=0)
      prod1 = Prod(
          input_list=[
              (shape3, 0),
              (Const(value=np.asarray([0])), 0)
          ],
      )
      prod = Prod(
          input_list=[
              (shape, 0),
              (Const(value=np.asarray([0])), 0)
          ],
      )
      maximum = Maximum(
          input_list=[
              (Const(value=np.asarray(1.)), 0),
              (prod1, 0)
          ],
      )
      div = FloorDiv(input_list=[(prod, 0), (maximum, 0)])
      reshape = Reshape(
          input_list=[in_grad_tensors[0], (ds, 0)],
      )
      bt = BroadcastTo(input_list=[(reshape, 0), (shape, 0)])
      bp_inputs = RealDiv(input_list=[(bt, 0), (div, 0)])
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class Sum(Operation):
  def __init__(self, input_list, graph=None, keepdims=False, name=None):
    super(Sum, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.sum(
        inputs,
        axis=tuple(reduction_indices.ravel().tolist()),
        keepdims=self._keepdims
    )
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Fill, Range, Reshape
    from data_flow_ops import DynamicStitch, BroadcastTo

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0]
      shape = op.get_shape_op(tensor_index=tensor_index)
      size = shape.get_size_op(tensor_index=0)

      add = Add(input_list=[(size, 0), self._input_list[1]])
      mod = FloorMod(input_list=[(add, 0), (size, 0)])
      shape1 = mod.get_shape_op(tensor_index=0)

      fill = Fill(
          input_list=[
              (shape1, 0),
              (Const(value=np.asarray(1)), 0)
          ],
      ) 
      range0 = Range(input_list=[
          (Const(value=np.asarray(0)), 0),
          (size, 0),
          (Const(value=np.asarray(1)), 0)]
      )    
      ds = DynamicStitch(
          input_list=[(range0, 0), (mod, 0), (shape, 0), (fill, 0)]
      )
      reshape = Reshape(input_list=[in_grad_tensors[0], (ds, 0)])
      bp_inputs = BroadcastTo(input_list=[(reshape, 0), (shape, 0)])
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class Prod(Operation):
  def __init__(self, input_list, graph=None, keepdims=False, name=None):
    super(Prod, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.prod(
        inputs,
        axis=tuple(reduction_indices.ravel().tolist()),
        keepdims=self._keepdims
    )
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import (
        Reshape,
        Range,
        ListDiff,
        Concat,
        Transpose,
        Pack,
        InvertPermutation,
        Fill
    )
    from data_flow_ops import Gather, DynamicStitch, BroadcastTo

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0]
      rank = op.get_rank_op(tensor_index=tensor_index)
      reshape = Reshape(input_list=[
          self._input_list[1],
          (Const(value=np.asarray([-1])), 0)])
      add1 = Add(input_list=[(reshape, 0), (rank, 0)])
      shape = op.get_shape_op(tensor_index=tensor_index)
      size = shape.get_size_op(tensor_index=0)
      range1 = Range(input_list=[
          (Const(value=np.asarray(0)), 0),
          (rank, 0), 
          (Const(value=np.asarray(1)), 0)])
      mod1 = FloorMod(input_list=[(add1, 0), (rank, 0)])
      listdiff = ListDiff(input_list=[(range1, 0), (mod1, 0)])    
      gather = Gather(input_list=[(shape, 0), (mod1, 0)])
      prod = Prod(input_list=[
          (gather, 0),
          (Const(value=np.asarray([0])), 0)
        ])
      gather1 = Gather(input_list=[(shape, 0), (listdiff, 0)]) 
      concat = Concat(input_list=[
          (Const(value=np.asarray(0)), 0),
          (mod1, 0),
          (listdiff, 0),
          ])
      transpose = Transpose(input_list=[self._input_list[0], (concat, 0)])
      shape2 = transpose.get_shape_op(tensor_index=0)
      prod1 = Prod(input_list=[
          (gather1, 0),
          (Const(value=np.asarray([0])), 0)
        ]) 
      pack = Pack(axis=0, input_list=[(prod, 0), (prod1, 0)])
      reshape2 = Reshape(input_list=[(transpose, 0), (pack, 0)])
      cumprod = Cumprod(exclusive=True, reverse=False,
          input_list=[(reshape2, 0), (Const(value=np.asarray(0)), 0)])
      cumprod1 = Cumprod(exclusive=True, reverse=True,
          input_list=[(reshape2, 0), (Const(value=np.asarray(0)), 0)])
      mul = Mul(input_list=[(cumprod, 0), (cumprod1, 0)]) 
      reshape3 = Reshape(input_list=[(mul, 0), (shape2, 0)])
      invert_perm = InvertPermutation(input_list=[(concat, 0)])
      transpose1 = Transpose(
          input_list=[(reshape3, 0), (invert_perm, 0)]
      )
      add = Add(input_list=[(size, 0), self._input_list[1]])
      mod = FloorMod(input_list=[(add, 0), (size, 0)])
      range0 = Range(input_list=[
        (Const(value=np.asarray(0)), 0),
        (size, 0),
        (Const(value=np.asarray(1)), 0)
        ],
      )
      shape1 = mod.get_shape_op(tensor_index=0)
      fill = Fill(input_list=[(shape1, 0), (Const(value=np.asarray(1)), 0)])
      ds = DynamicStitch(
          input_list=[(range0, 0), (mod, 0), (shape, 0), (fill, 0)]
      )
      reshape1 = Reshape(input_list=[in_grad_tensors[0], (ds, 0)])
      bt = BroadcastTo(input_list=[(reshape1, 0), (shape, 0)])
      mul1 = Mul(input_list=[(bt, 0), (transpose1, 0)])
      bp_inputs = Reshape(input_list=[(mul1, 0), (shape, 0)])
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class MatMul(Operation):
  """Single-batch only: 2D x 2D matrix """
  def __init__(self,
               input_list,
               graph=None,
               transpose_x=False,
               transpose_y=False,
               name=None
    ):
    super(MatMul, self).__init__(graph=graph, name=name, input_list=input_list)
    self._transpose_x = transpose_x
    self._transpose_y = transpose_y

  def _run(self, x, y):
    xx = x.T if self._transpose_x else x
    yy = y.T if self._transpose_y else y
    outputs = np.dot(xx, yy)
    if self._transpose_x:
      outputs = outputs.T
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_x = MatMul(
          graph=self._graph,
          input_list=in_grad_tensors+self._input_list[1:],
          transpose_y=True
      )
      bp_y = MatMul(
          graph=self._graph,
          input_list=in_grad_tensors+self._input_list[:1],
          transpose_x=True
      )

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class BatchMatMul(Operation):
  def __init__(
        self,
        input_list,
        graph=None,
        transpose_x=False,
        transpose_y=False,
        name=None
    ):
    super(BatchMatMul, self).__init__(
        graph=graph, name=name, input_list=input_list
    )
    self._transpose_x = transpose_x
    self._transpose_y = transpose_y

  def _run(self, x, y):
    x_multiples, y_multiples = [], []
    for i, j in zip(x.shape[::-1][2:], y.shape[::-1][2:]):
      if i == 1:
        x_multiples.append(j)
        y_multiples.append(1)
      elif j == 1:
        y_multiples.append(i)
        x_multiples.append(1)
      else:
        x_multiples.append(1)
        y_multiples.append(1)

    if x.ndim > y.ndim:
      x_multiples = x_multiples + [1] * (x.ndim - y.ndim)
      y_multiples = y_multiples + list(x.shape[:x.ndim-y.ndim])
    if y.ndim > x.ndim:
      y_multiples = y_multiples + [1] * (y.ndim - x.ndim)
      x_multiples = x_multiples + list(y.shape[:y.ndim-x.ndim])


    x_multiples = x_multiples[::-1] + [1, 1]
    y_multiples = y_multiples[::-1] + [1, 1]
    x = np.tile(x, x_multiples)
    y = np.tile(y, y_multiples)
    x_flat = np.reshape(x, (-1,) + x.shape[-2:])
    y_flat = np.reshape(y, (-1,) + y.shape[-2:])
    def _func(index):
      index = index.item()
      return np.dot(
        x_flat[index].T if self._transpose_x else x_flat[index],
        y_flat[index].T if self._transpose_y else y_flat[index]
      )
    outputs = np.apply_along_axis(
        _func, 1, np.arange(x_flat.shape[0])[:, np.newaxis]
    )
    outputs = np.reshape(outputs, x.shape[:-2] + outputs.shape[-2:])
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import StridedSlice, Reshape

    with self._graph.as_default_graph():
      if not self._transpose_x and not self._transpose_y:
        bmm = BatchMatMul(
            input_list=[in_grad_tensors[0], self._input_list[1]],
            transpose_x=False,
            transpose_y=True,
        )
        bmm1 = BatchMatMul(
            input_list=[self._input_list[0], in_grad_tensors[0]],
            transpose_x=True,
            transpose_y=False,
        )
      elif not self._transpose_x and self._transpose_y:
        bmm = BatchMatMul(
            input_list=[in_grad_tensors[0], self._input_list[1]],
            transpose_x=False,
            transpose_y=False,
        )
        bmm1 = BatchMatMul(
            input_list=[in_grad_tensors[0], self._input_list[0]],
            transpose_x=True,
            transpose_y=False,
        )
      elif self._transpose_x and not self._transpose_y:
        bmm = BatchMatMul(
            input_list=[self._input_list[1], in_grad_tensors[0]],
            transpose_x=False,
            transpose_y=True,
        )
        bmm1 = BatchMatMul(
            input_list=[self._input_list[0], in_grad_tensors[0]],
            transpose_x=False,
            transpose_y=False,
        )
      else:
        bmm = BatchMatMul(
            input_list=[self._input_list[1], in_grad_tensors[0]],
            transpose_x=True,
            transpose_y=True,
        )
        bmm1 = BatchMatMul(
            input_list=[in_grad_tensors[0], self._input_list[0]],
            transpose_x=True,
            transpose_y=True,
        )


      op_x, tensor_index_x = self._input_list[0]
      op_y, tensor_index_y = self._input_list[1]
      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)

      ss = StridedSlice(
          input_list=[
              (shape, 0),
              (Const(value=np.asarray([0])), 0),
              (Const(value=np.asarray([-2])), 0),
              (Const(value=np.asarray([1])), 0),
          ]
      )
      ss1 = StridedSlice(
          input_list=[
              (shape1, 0),
              (Const(value=np.asarray([0])), 0),
              (Const(value=np.asarray([-2])), 0),
              (Const(value=np.asarray([1])), 0),
          ]
      )

      bga = BroadcastGradientArgs(
          input_list=[(ss, 0), (ss1, 0)]
      ) 

      sum0 = Sum(input_list=[(bmm, 0), (bga, 0)])
      sum1 = Sum(input_list=[(bmm1, 0), (bga, 1)])

      bp_x = Reshape(input_list=[(sum0, 0), (shape, 0)])
      bp_y = Reshape(input_list=[(sum1, 0), (shape1, 0)])

      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors


class SquaredDifference(Operation):
  def _run(self, inputs1, inputs2):
    outputs = np.square(inputs1 - inputs2)
    return outputs 


class Square(Operation):
  def _run(self, inputs):
    outputs = np.square(inputs)
    return outputs

  def _grad_func(self, in_grad_tensors):

    with self._graph.as_default_graph():
      mul = Mul(
          input_list=[
              (Const(value=np.asarray(2.)), 0),
              self._input_list[0]
          ]
      )
      mul1 = Mul(input_list=[(mul, 0), in_grad_tensors[0]])
      out_grad_tensors = [(mul1, 0)]   
    return out_grad_tensors


class GreaterEqual(Operation):
  def _run(self, x, y):
    outputs = np.greater_equal(x, y)
    return outputs


class Greater(Operation):
  def _run(self, x, y):
    outputs = np.greater(x, y)
    return outputs


class LessEqual(Operation):
  def _run(self, x, y):
    outputs = np.less_equal(x, y)
    return outputs


class Less(Operation):
  def _run(self, x, y):
    outputs = np.less(x, y)
    return outputs


class Equal(Operation):
  def _run(self, x, y):
    outputs = np.equal(x, y)
    return outputs


class NotEqual(Operation):
  def _run(self, x, y):
    outputs = np.not_equal(x, y)
    return outputs


class Cumsum(Operation):
  def __init__(
      self,
      reverse,
      exclusive,
      input_list,
      graph=None,
      name=None
    ):
    super(Cumsum, self).__init__(graph=graph, input_list=input_list, name=name)
    self._exclusive = exclusive
    self._reverse = reverse

  def _run(self, inputs, axis):
    if self._reverse:
      inputs = np.take(
          inputs, np.arange(inputs.shape[axis] - 1, -1, -1), axis=axis
      )
    if self._exclusive:
      inputs = np.take(inputs, np.arange(0, inputs.shape[axis] - 1), axis=axis)
      paddings = np.zeros((inputs.ndim, 2)).astype("int")
      paddings[axis, 0] = 1
      inputs = np.pad(inputs, paddings)
    outputs = np.cumsum(inputs, axis)
    if self._reverse:
      outputs = np.take(
          outputs, np.arange(outputs.shape[axis] - 1, -1, -1), axis=axis
      )
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = Cumsum(
        reverse=not self._reverse,
        exclusive=self._exclusive,
        input_list=[in_grad_tensors[0], self._input_list[1]],
      )
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class Cumprod(Operation):
  def __init__(
      self,
      reverse,
      exclusive,
      input_list,
      graph=None,
      name=None
    ):
    super(Cumprod, self).__init__(graph=graph, input_list=input_list, name=name)
    self._exclusive = exclusive
    self._reverse = reverse

  def _run(self, inputs, axis):
    if self._reverse:
      inputs = np.take(
          inputs, np.arange(inputs.shape[axis] - 1, -1, -1), axis=axis
      )
    if self._exclusive:
      inputs = np.take(inputs, np.arange(0, inputs.shape[axis] - 1), axis=axis)
      paddings = np.zeros((inputs.ndim, 2)).astype("int")
      paddings[axis, 0] = 1
      inputs = np.pad(inputs, paddings, constant_values=1)
    outputs = np.cumprod(inputs, axis)
    if self._reverse:
      outputs = np.take(
          outputs, np.arange(outputs.shape[axis] - 1, -1, -1), axis=axis
      )
    return outputs

  def _grad_func(self, in_grad_tensors):

    with self._graph.as_default_graph():
      prod = Cumprod(
          reverse=self._reverse,
          exclusive=self._exclusive,
          input_list=self._input_list,
      )
      mul = Mul(
          input_list=[in_grad_tensors[0], (prod, 0)]
      )
      cum0 = Cumsum(
          reverse=not self._reverse,
          exclusive=self._exclusive,
          input_list=[(mul, 0), self._input_list[1]],
      )
      bp_inputs = DivNoNan(
          input_list=[(cum0, 0), self._input_list[0]]
      )
      out_grad_tensors = [(bp_inputs, 0)]
    return out_grad_tensors


class Exp(Operation):
  def _run(self, inputs):
    outputs = np.exp(inputs)
    return outputs

  def _grad_func(self, in_grad_tensors):

    with self._graph.as_default_graph():
      bp_inputs = Mul(input_list=[in_grad_tensors[0], (self, 0)])
      out_grad_tensors = [(bp_inputs, 0)]

    return out_grad_tensors


class Log1p(Operation):
  def _run(self, inputs):
    outputs = np.log(np.add(1, inputs))
    return outputs

  def _grad_func(self, in_grad_tensors):
    pass


class Reciprocal(Operation):
  pass
