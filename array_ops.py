"""
"""
import numpy as np

from operation import Operation

from origin_ops import Const
from math_ops import Sum, Mean


class Reshape(Operation):

  def _run(self, inputs, shape):
    outputs = np.reshape(inputs, shape.astype("int32"))
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    input_list = [(bwval_op, tensor_index), (self._input_list[0][0].get_shape_op(), 0)]
    bp_inputs = Reshape(graph=self._graph, input_list=input_list)
    self._input_list[0][0].backprop(bwval_list=[(bp_inputs, 0)])
    return bp_inputs 


class Transpose(Operation):

  def _run(self, inputs, perm):
    outputs = np.transpose(inputs, perm)
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    input_list = [self._input_list[1]]
    invert_perm = InvertPermutation(graph=self._graph, input_list=input_list)

    input_list = [(bwval_op, tensor_index), (invert_perm, 0)]
    bp_inputs = Transpose(graph=self._graph, input_list=input_list)

    self._input_list[0][0].backprop(bwval_list=[(bp_inputs, 0)])
    return bp_inputs


class InvertPermutation(Operation):

  def _run(self, perm):
    return np.argsort(perm)

  def backprop(self, bwval_list):
    return


class Range(Operation):
  def _run(self, start, limit, delta):
    #print(start, limit, delta)
    return np.arange(start, limit, delta)

  def backprop(self, bwval_list):
    return


class Pack(Operation):
  def __init__(self, axis, graph, input_list, name=None):
    super(Pack, self).__init__(graph=graph, input_list=input_list, name=name)
    self._axis = axis
    self._num = len(input_list)

  def _run(self, *input_tensor_values):
    outputs = np.stack(input_tensor_values, axis=self._axis)
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    input_list = [(bwval_op, tensor_index)]
    bp_inputs = Unpack(axis=self._axis, graph=self._graph, input_list=input_list)
    for i in range(self._num): 
      self._input_list[i][0].backprop(bwval_list=[(bp_inputs, i)])
    return bp_inputs
    

class Unpack(Operation):
  def __init__(self, axis, graph, input_list, num=None, name=None):
    super(Unpack, self).__init__(graph=graph, input_list=input_list, name=name)
    self._axis = axis
  
  def _run(self, inputs):
    axis_size = inputs.shape[self._axis]
    outputs = np.split(inputs, axis_size, axis=self._axis) 
    outputs = [np.squeeze(output, axis=self._axis) for output in outputs]
    return outputs

  def backprop(self, bwval_list):
    input_list = bwval_list
    bp_inputs = Pack(axis=self._axis, graph=self._graph, input_list=input_list)

    self._input_list[0][0].backprop(bwval_list=[(bp_inputs, 0)])
    return bp_inputs



class Tile(Operation):

  def _run(self, inputs, multiples):
    outputs = np.tile(inputs, multiples)
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    input_list = [self._input_list[1], (self._input_list[0][0].get_shape_op(), 0)]
    pack = Pack(axis=0, graph=self._graph, input_list=input_list)
    input_list = [(pack, 0), (Const(value=np.asarray((1, 0)), graph=self._graph), 0)]
    transpose = Transpose(input_list=input_list, graph=self._graph)
    input_list = [(transpose, 0), (Const(value=np.asarray(-1), graph=self._graph), 0 )]
    reshape = Reshape(input_list=input_list, graph=self._graph)

    input_list = [(bwval_op, tensor_index), (reshape, 0)]
    reshape1 = Reshape(input_list=input_list, graph=self._graph)

    input_list = [(Const(value=np.asarray(0), graph=self._graph), 0), (reshape.get_size_op(), 0), (Const(value=np.asarray(2), graph=self._graph), 0)]
    reduction_indices = Range(input_list=input_list, graph=self._graph)

    input_list = [(reshape1, 0), (reduction_indices, 0)]
    bp_inputs = Sum(input_list=input_list, graph=self._graph)

    return bp_inputs


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

  def backprop(self, bwval_list):
    input_list = [(self._input_list[0][0].get_shape_op(), 0)
        ] + self._input_list[1:4] + bwval_list
    inputs_grads = StridedSliceGrad(graph=self._graph, input_list=input_list)
    return inputs_grads


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

  def backprop(self, bwval_list):
    bp_grads = StridedSlice(
        graph=self._graph,
        input_list=bwval_list + self._input_list[1:4]
    )     
    return bp_grads


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

  def backprop(self, bwval_list): 
    from arithmetic_ops import Sub
    sub = Sub(input_list=
        [(self._input_list[0][0].get_shape_op(), 0)] + 
        [(self.get_shape_op(), 0)],
          graph=self._graph
    ) 

    sub1 = Sub(input_list=[(sub, 0), self._input_list[1]], graph=self._graph)
    pack = Pack(axis=0, input_list=[(self._input_list[0][0].get_rank_op(), 0), (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph) 

    reshape = Reshape(input_list=[self._input_list[1], (pack, 0)], graph=self._graph)
    reshape1 = Reshape(input_list=[(sub1, 0), (pack, 0)], graph=self._graph)

    concat = Concat(input_list=[(Const(value=np.asarray(1), graph=self._graph), 0), (reshape, 0), (reshape1, 0)], graph=self._graph)

    bp_inputs = Pad(input_list=bwval_list+[(concat, 0)], graph=self._graph) 
    return bp_inputs

class Concat(Operation):
  
  def _run(self, axis, *input_tensor_values):
    outputs = np.concatenate(input_tensor_values, axis=axis) 
    return outputs

  def backprop(self, bwval_list):
    from arithmetic_ops import FloorMod

    shapes = [(arg[0].get_shape_op(), 0) for arg in self._input_list[1:]]

    mod = FloorMod(input_list=[self._input_list[0]] + [(self._input_list[1][0].get_rank_op(), 0)], graph=self._graph)

    offset = ConcatOffset(input_list=[(mod, 0)]+shapes, graph=self._graph) 

    input_list = [(offset, i) for i in range(len(self._input_list) - 1)]

    grads_list = []
    for i in range(len(self._input_list) - 1):

      grads = Slice(input_list=[bwval_list[0]] + [(offset, i)] + [shapes[i]], graph=self._graph)
      grads_list.append(grads)
    return grads_list


class ConcatOffset(Operation):
  def _run(self, concat_dim, *shape):
    shapes = np.cumsum([0] + [s[concat_dim] for s in shape[:-1]])
    shapes = np.pad(np.expand_dims(shapes, axis=1), [[0, 0], [concat_dim, len(shape[0]) - concat_dim - 1]])
    shapes = [s for s in shapes]
    return shapes

  def backprop(self, bwval_list):
    return


class Pad(Operation):
  def __init__(self, graph, input_list, name=None, constant_values=0):
    super(Pad, self).__init__(graph=graph, input_list=input_list, name=name)
    self._constant_values = constant_values

  def _run(self, inputs, paddings):
    outputs = np.pad(inputs, paddings, constant_values=self._constant_values)
    return outputs

  def backprop(self, bwval_list):
    pack = Pack(axis=0, input_list=[(self._input_list[0][0].get_rank_op(), 0), (Const(value=np.asarray(1), graph=self._graph), 0)], graph=self._graph)

    slice0 = Slice(input_list=[self._input_list[1], (Const(value=np.asarray((0, 0)), graph=self._graph), 0), (pack, 0)], graph=self._graph)
    reshape = Reshape(input_list=[(slice0, 0), (Const(value=np.asarray(-1), graph=self._graph), 0)], graph=self._graph) 

    bp_inputs = Slice(input_list=[bwval_list[0], (reshape, 0), (self._input_list[0][0].get_shape_op(), 0)], graph=self._graph)
    return bp_inputs


class ExpandDims(Operation):
  def _run(self, inputs, axis):
    outputs = np.expand_dims(inputs, axis.item())
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    bp_inputs = Reshape(
        graph=self._graph,
        input_list=[bwval_list[0]] + [(self._input_list[0][0].get_shape_op(), 0)] 
    )
    return bp_inputs


class Squeeze(Operation):
  def __init__(self, graph, input_list, axis=None, name=None)
    super(Squeeze, self).__init__(graph=graph, input_list=input_list, name=name)
    self._axis = axis

  def _run(self, inputs):
    outputs = np.squeeze(inputs, axis=self._axis) 

  def backprop(self, bwval_list):
    bp_inputs = Reshape(
        input_list=[
            bwval_list[0], (self._input_list[0][0].get_shape_op(), 0)
        ],
        graph=self._graph
    )
    return bp_inputs

class Fill(Operation):
  def _run(self, dims, value):
    outputs = np.ones(dims, dtype="float32") * value
    return outputs

  def backprop(self, bwval_list):
    return



class ListDiff(Operation):
  def _run(self, x, y):
    outputs = np.setdiff1d(x, y)
    return outputs

  def backprop(self, bwval_list):
    return
 
