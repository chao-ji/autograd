
import numpy as np

from operation import Operation
from origin_ops import Const


class Mean(Operation):
  def __init__(self, graph, input_list, keepdims=False, name=None):
    super(Mean, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.mean(inputs, axis=tuple(reduction_indices.ravel().tolist()), keepdims=self._keepdims)
    return outputs

  def backprop(self, bwval_list):
    from arithmetic_ops import Add, FloorMod, Maximum, FloorDiv, RealDiv 
    from array_ops import Fill, Range, Reshape
    from data_flow_ops import DynamicStitch, BroadcastTo

    shape = self._input_list[0][0].get_shape_op() 
    size = shape.get_size_op()
    add = Add(input_list=[(size, 0), self._input_list[1]], graph=self._graph)
    mod = FloorMod(input_list=[(add, 0), (size, 0)], graph=self._graph)
    shape1 = mod.get_shape_op()
    fill = Fill(input_list=[
        (shape1, 0),
        (Const(value=np.asarray(1), graph=self._graph), 0)
      ],
      graph=self._graph
    )

    range0 = Range(input_list=[
        (Const(value=np.asarray(0), graph=self._graph), 0),
        (size, 0),
        (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph)
    ds = DynamicStitch(input_list=[
        (range0, 0), (mod, 0), (shape, 0), (fill, 0)], graph=self._graph) 

    shape3 = self.get_shape_op()   
    prod = Prod(input_list=[(shape3, 0), (Const(value=np.asarray([0]), graph=self._graph), 0)], graph=self._graph)
    prod1 = Prod(input_list=[(shape, 0), (Const(value=np.asarray([0]), graph=self._graph), 0)], graph=self._graph)

    maximum = Maximum(input_list=[(Const(value=np.asarray(1.), graph=self._graph), 0), (prod1, 0)], graph=self._graph)
    
    div = FloorDiv(input_list=[(maximum, 0), (prod, 0)], graph=self._graph)
    
    reshape = Reshape(input_list=[bwval_list[0], (ds, 0)], graph=self._graph)
    
    bt = BroadcastTo(input_list=[(reshape, 0), (shape, 0)], graph=self._graph)

    bp_inputs = RealDiv(input_list=[(bt, 0), (div, 0)], graph=self._graph)
    return bp_inputs

class Sum(Operation):

  def __init__(self, graph, input_list, keepdims=False, name=None):
    super(Sum, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.sum(inputs, axis=tuple(reduction_indices.ravel().tolist()), keepdims=self._keepdims)
    return outputs

  def backprop(self, bwval_list):
    from arithmetic_ops import Add, FloorMod
    from array_ops import Fill, Range, Reshape
    from data_flow_ops import DynamicStitch, BroadcastTo

    shape = self._input_list[0][0].get_shape_op()

    size = shape.get_size_op()

    add = Add(input_list=[(size, 0), self._input_list[1]], graph=self._graph)
    mod = FloorMod(input_list=[(add, 0), (size, 0)], graph=self._graph)

    shape1 = mod.get_shape_op()

    fill = Fill(input_list=[(shape1, 0), (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph) 
    range0 = Range(input_list=[
        (Const(value=np.asarray(0), graph=self._graph), 0),
        (size, 0),
        (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph)    

    ds = DynamicStitch(input_list=[(range0, 0), (mod, 0), (shape, 0), (fill, 0)], graph=self._graph)
    reshape = Reshape(input_list=[bwval_list[0], (ds, 0)], graph=self._graph)
    bp_inputs = BroadcastTo(input_list=[(reshape, 0), (shape, 0)], graph=self._graph)
    return bp_inputs

    return range0, mod, shape, fill 


class Prod(Operation):
  def __init__(self, graph, input_list, keepdims=False, name=None):
    super(Prod, self).__init__(graph=graph, input_list=input_list, name=name)
    self._keepdims = keepdims

  def _run(self, inputs, reduction_indices):
    outputs = np.prod(inputs, axis=tuple(reduction_indices.ravel().tolist()), keepdims=self._keepdims)
    return outputs

  def backprop(self, bwval_list):
    from array_ops import (Reshape, Range, ListDiff, Concat, Transpose, Pack, InvertPermutation, Fill)
    from data_flow_ops import Gather, DynamicStitch, BroadcastTo
    from arithmetic_ops import Add, Mul, FloorMod

    rank = self._input_list[0][0].get_rank_op()
    reshape = Reshape(input_list=[
        self._input_list[1],
        (Const(value=np.asarray([-1]), graph=self._graph), 0)], graph=self._graph)
    add1 = Add(input_list=[(reshape, 0), (rank, 0)], graph=self._graph)
    shape = self._input_list[0][0].get_shape_op()
    size = shape.get_size_op()

    range1 = Range(input_list=[
        (Const(value=np.asarray(0), graph=self._graph), 0),
        (rank, 0), 
        (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph)

    mod1 = FloorMod(input_list=[(add1, 0), (rank, 0)], graph=self._graph)

    listdiff = ListDiff(input_list=[(range1, 0), (mod1, 0)], graph=self._graph)    
    gather = Gather(input_list=[(shape, 0), (mod1, 0)], graph=self._graph)
   
    prod = Prod(input_list=[
        (gather, 0),
        (Const(value=np.asarray([0]), graph=self._graph), 0)
      ], graph=self._graph)

    gather1 = Gather(input_list=[(shape, 0), (listdiff, 0)], graph=self._graph) 

    concat = Concat(input_list=[
        (Const(value=np.asarray(0), graph=self._graph), 0),
        (mod1, 0),
        (listdiff, 0),
        ], graph=self._graph)

    transpose = Transpose(input_list=[self._input_list[0], (concat, 0)], graph=self._graph)
    shape2 = transpose.get_shape_op()

    prod1 = Prod(input_list=[
        (gather1, 0),
        (Const(value=np.asarray([0]), graph=self._graph), 0)
      ], graph=self._graph) 

    pack = Pack(axis=0, input_list=[(prod, 0), (prod1, 0)], graph=self._graph)

    reshape2 = Reshape(input_list=[(transpose, 0), (pack, 0)], graph=self._graph)

    cumprod = Cumprod(exclusive=True, reverse=False,
        input_list=[(reshape2, 0), (Const(value=np.asarray(0), graph=self._graph), 0)], graph=self._graph)
    cumprod1 = Cumprod(exclusive=True, reverse=True,
        input_list=[(reshape2, 0), (Const(value=np.asarray(0), graph=self._graph), 0)], graph=self._graph)
    mul = Mul(input_list=[(cumprod, 0), (cumprod1, 0)], graph=self._graph) 

    reshape3 = Reshape(input_list=[(mul, 0), (shape2, 0)], graph=self._graph)
    invert_perm = InvertPermutation(input_list=[(concat, 0)], graph=self._graph)
    transpose1 = Transpose(
        input_list=[(reshape3, 0), (invert_perm, 0)], graph=self._graph
    )


    add = Add(input_list=[(size, 0), self._input_list[1]], graph=self._graph)

    mod = FloorMod(input_list=[(add, 0), (size, 0)], graph=self._graph)

    range0 = Range(input_list=[
      (Const(value=np.asarray(0), graph=self._graph), 0),
      (size, 0),
      (Const(value=np.asarray(1), graph=self._graph), 0)
      ],
      graph=self._graph
    )

    shape1 = mod.get_shape_op() 
    
    fill = Fill(input_list=[(shape1, 0), (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph)

    ds = DynamicStitch(input_list=[(range0, 0), (mod, 0), (shape, 0), (fill, 0)], graph=self._graph)

    reshape1 = Reshape(input_list=[bwval_list[0], (ds, 0)], graph=self._graph)

    bt = BroadcastTo(input_list=[(reshape1, 0), (shape, 0)], graph=self._graph)

    mul1 = Mul(input_list=[(bt, 0), (transpose1, 0)], graph=self._graph)

    bp_inputs = Reshape(input_list=[(mul1, 0), (shape, 0)], graph=self._graph)
    return bp_inputs


class MatMul(Operation):
  """Single-batch only: 2D x 2D matrix """
  def __init__(self, input_list, graph, transpose_x=False, transpose_y=False, name=None):
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

  def backprop(self, bwval_list):
    bp_x = MatMul(graph=self._graph, input_list=bwval_list+self._input_list[1:], transpose_y=True)
    bp_y = MatMul(graph=self._graph, input_list=bwval_list+self._input_list[:1], transpose_x=True)

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])

    return bp_x, bp_y



class BatchMatMul(Operation):
  def __init__(self, input_list, graph, transpose_x=False, transpose_y=False, name=None):
    super(BatchMatMul, self).__init__(graph=graph, name=name, input_list=input_list)
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

  def backprop(self, bwval_list):
    from array_ops import StridedSlice, Reshape
    from arithmetic_ops import BroadcastGradientArgs

    if not self._transpose_x and not self._transpose_y:
      bmm = BatchMatMul(
          input_list=[bwval_list[0], self._input_list[1]],
          graph=self._graph,
          transpose_x=False,
          transpose_y=True,
      )
      bmm1 = BatchMatMul(
          input_list=[self._input_list[0], bwval_list[0]],
          graph=self._graph,
          transpose_x=True,
          transpose_y=False,
      )
    elif not self._transpose_x and self._transpose_y:
      bmm = BatchMatMul(
          input_list=[bwval_list[0], self._input_list[1]],
          graph=self._graph,
          transpose_x=False,
          transpose_y=False,
      )
      bmm1 = BatchMatMul(
          input_list=[bwval_list[0], self._input_list[0]],
          graph=self._graph,
          transpose_x=True,
          transpose_y=False,
      )
    elif self._transpose_x and not self._transpose_y:
      bmm = BatchMatMul(
          input_list=[self._input_list[1], bwval_list[0]],
          graph=self._graph,
          transpose_x=False,
          transpose_y=True,
      )
      bmm1 = BatchMatMul(
          input_list=[self._input_list[0], bwval_list[0]],
          graph=self._graph,
          transpose_x=False,
          transpose_y=False,
      )
    else:
      bmm = BatchMatMul(
          input_list=[self._input_list[1], bwval_list[0]],
          graph=self._graph,
          transpose_x=True,
          transpose_y=True,
      )
      bmm1 = BatchMatMul(
          input_list=[bwval_list[0], self._input_list[0]],
          graph=self._graph,
          transpose_x=True,
          transpose_y=True,
      )

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    ss = StridedSlice(
        input_list=[
            (x_shape, 0),
            (Const(value=np.asarray([0]), graph=self._graph), 0),
            (Const(value=np.asarray([-2]), graph=self._graph), 0),
            (Const(value=np.asarray([1]), graph=self._graph), 0),
          ], graph=self._graph
    )
    ss1 = StridedSlice(
        input_list=[
            (y_shape, 0),
            (Const(value=np.asarray([0]), graph=self._graph), 0),
            (Const(value=np.asarray([-2]), graph=self._graph), 0),
            (Const(value=np.asarray([1]), graph=self._graph), 0),
          ], graph=self._graph
    )

    bga = BroadcastGradientArgs(
        input_list=[(ss, 0), (ss1, 0)], graph=self._graph
    ) 

    sum0 = Sum(input_list=[(bmm, 0), (bga, 0)], graph=self._graph)
    sum1 = Sum(input_list=[(bmm1, 0), (bga, 1)], graph=self._graph)

    #return sum0, sum1

    bp_x = Reshape(input_list=[(sum0, 0), (x_shape, 0)], graph=self._graph)
    bp_y = Reshape(input_list=[(sum1, 0), (y_shape, 0)], graph=self._graph)
    return bp_x, bp_y


class SquaredDifference(Operation):
  def _run(self, inputs1, inputs2):
    outputs = np.square(inputs1 - inputs2)
    return outputs 




class GreaterEqual(Operation):
  def _run(self, x, y):
    outputs = np.greater_equal(x, y)
    return outputs


class Cumsum(Operation):
  def __init__(self, reverse, exclusive, graph, input_list, name=None):
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

  def backprop(self, bwval_list):
    bp_inputs = Cumsum(
      reverse=not self._reverse,
      exclusive=self._exclusive,
      input_list=[bwval_list[0], self._input_list[1]],
      graph=self._graph
    )
    return bp_inputs


class Cumprod(Operation):
  def __init__(self, reverse, exclusive, graph, input_list, name=None):
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

  def backprop(self, bwval_list):
    from arithmetic_ops import Mul, DivNoNan

    prod = Cumprod(
        reverse=self._reverse,
        exclusive=self._exclusive,
        input_list=self._input_list,
        graph=self._graph
    )
    mul = Mul(
        input_list=[bwval_list[0], (prod, 0)], graph=self._graph
    )
    cum0 = Cumsum(
        reverse=not self._reverse,
        exclusive=self._exclusive,
        input_list=[(mul, 0), self._input_list[1]],
        graph=self._graph
    )
    bp_inputs = DivNoNan(
        input_list=[(cum0, 0), self._input_list[0]], graph=self._graph)
    return bp_inputs
    


class Exp(Operation):
  def _run(self, inputs):
    outputs = np.exp(inputs)
    return outputs

  def backprop(self, bwval_list):
    from arithmetic_ops import Mul 

    bp_inputs = Mul(input_list=[bwval_list[0], (self, 0)], graph=self._graph)
    return bp_inputs

