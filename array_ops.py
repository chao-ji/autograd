"""Operations on multi-dimensional arrays."""
import numpy as np

from operation import Operation

from origin_ops import Const
from math_ops import Sum, Mean
from tensor import Tensor


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
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class Transpose(Operation):

  def _run(self, inputs, perm):
    outputs = np.transpose(inputs, perm)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      invert_perm = InvertPermutation(input_list=[self._input_list[1]])
      bp_inputs = Transpose(input_list=[in_grad_tensors[0], Tensor(invert_perm, 0)])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class InvertPermutation(Operation):

  def _run(self, perm):
    outputs = np.argsort(perm)
    return outputs


class Range(Operation):

  def _run(self, start, limit, delta):
    outputs = np.arange(start, limit, delta)
    return outputs


class Pack(Operation):

  def __init__(self, axis, input_list, graph=None, name=None):
    super(Pack, self).__init__(graph=graph, input_list=input_list, name=name)
    self._axis = axis

  def _run(self, *input_tensor_values):
    outputs = np.stack(input_tensor_values, axis=self._axis)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = Unpack(axis=self._axis, input_list=[in_grad_tensors[0]])
      out_grad_tensors = [Tensor(bp_inputs, i) for i in range(len(self._input_list))]
    return out_grad_tensors


class Unpack(Operation):

  def __init__(self, axis, input_list, graph=None, num=None, name=None):
    super(Unpack, self).__init__(graph=graph, input_list=input_list, name=name)
    self._axis = axis
  
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
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors



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
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      transpose = Transpose(
          input_list=[
            Tensor(pack, 0),
            Tensor(Const(value=np.asarray((1, 0))), 0)
          ]
      )
      reshape = Reshape(
          input_list=[
            Tensor(transpose, 0),
            Tensor(Const(value=np.asarray(-1)), 0)
          ]
      )
      reshape1 = Reshape(
          input_list=[in_grad_tensors[0], Tensor(reshape, 0)]
      )
      reduction_indices = Range(
          input_list=[
              Tensor(Const(value=np.asarray(0)), 0),
              Tensor(reshape.get_size_op(tensor_index=0), 0),
              Tensor(Const(value=np.asarray(2)), 0)
          ]
      )
      bp_inputs = Sum(
          input_list=[Tensor(reshape1, 0), Tensor(reduction_indices, 0)]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


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
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ] + self._input_list[1:4] + in_grad_tensors
      )
      out_grad_tensors = [Tensor(ssg, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


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
      out_grad_tensors = [Tensor(bp_grads, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [4]


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
    from math_ops import Sub

    with self._graph.as_default_graph():

      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      sub = Sub(input_list=[
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0), 
              Tensor(self.get_shape_op(tensor_index=0), 0)
          ],
      ) 
      sub1 = Sub(
          input_list=[Tensor(sub, 0), self._input_list[1]],
      )
      pack = Pack(
          axis=0,
          input_list=[
              Tensor(op.get_rank_op(tensor_index=tensor_index), 0),
              Tensor(Const(value=np.asarray(1)), 0)
          ],
      ) 
      reshape = Reshape(input_list=[self._input_list[1], Tensor(pack, 0)])
      reshape1 = Reshape(input_list=[Tensor(sub1, 0), Tensor(pack, 0)])
      concat = Concat(
          input_list=[
              Tensor(Const(value=np.asarray(1)), 0),
              Tensor(reshape, 0), Tensor(reshape1, 0)
          ],
      )

      bp_inputs = Pad(input_list=in_grad_tensors+[Tensor(concat, 0)])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class Concat(Operation):

  def _run(self, axis, *input_tensor_values):
    outputs = np.concatenate(input_tensor_values, axis=axis) 
    return outputs

  def _grad_func(self, in_grad_tensors):
    from math_ops import FloorMod
    
    with self._graph.as_default_graph():
      shapes = [
          Tensor(arg.op.get_shape_op(tensor_index=arg.tensor_index), 0)
              for arg in self._input_list[1:]
      ]
      op, tensor_index = self._input_list[1].op, self._input_list[1].tensor_index
      mod = FloorMod(
          input_list=[self._input_list[0]] + [
              Tensor(op.get_rank_op(tensor_index=tensor_index), 0)
          ],
      )
      offset = ConcatOffset(
          input_list=[Tensor(mod, 0)]+shapes,
      ) 
      out_grad_tensors = []
      for i in range(len(self._input_list) - 1):
        out_grad_tensors.append(Tensor(
            Slice(input_list=[in_grad_tensors[0], Tensor(offset, i), shapes[i]],
            ), 0
        ))
    return out_grad_tensors

  def _get_bp_indices(self):
    backprop_indices = list(range(1, len(self._input_list)))
    return backprop_indices 


class ConcatOffset(Operation):

  def _run(self, concat_dim, *shape):
    shapes = np.cumsum([0] + [s[concat_dim] for s in shape[:-1]])
    shapes = np.pad(np.expand_dims(shapes, axis=1), [[0, 0], [concat_dim, len(shape[0]) - concat_dim - 1]])
    shapes = [s for s in shapes]
    return shapes


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
              Tensor(op.get_rank_op(tensor_index=tensor_index), 0),
              Tensor(Const(value=np.asarray(1)), 0)
          ],
      )

      slice0 = Slice(
          input_list=[
              self._input_list[1],
              Tensor(Const(value=np.asarray((0, 0))), 0),
              Tensor(pack, 0)
          ],
      )
      reshape = Reshape(
          input_list=[
              Tensor(slice0, 0),
              Tensor(Const(value=np.asarray(-1)), 0)
          ],
      ) 

      bp_inputs = Slice(
          input_list=[
              in_grad_tensors[0],
              Tensor(reshape, 0),
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ],
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors 

  def _get_bp_indices(self):
    return [0]


class ExpandDims(Operation):

  def _run(self, inputs, axis):
    outputs = np.expand_dims(inputs, axis.item())
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Reshape(
          input_list=[
              in_grad_tensors[0], Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ] 
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class Squeeze(Operation):

  def __init__(self, input_list, graph=None, axis=None, name=None):
    super(Squeeze, self).__init__(graph=graph, input_list=input_list, name=name)
    if isinstance(axis, int):
      axis = (axis,)
    self._axis = tuple(axis)

  def _run(self, inputs):
    outputs = np.squeeze(inputs, axis=self._axis) 
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Reshape(
          input_list=[
              in_grad_tensors[0], Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ],
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class Fill(Operation):

  def _run(self, dims, value):
    outputs = np.ones(dims, dtype="float32") * value
    return outputs


class ListDiff(Operation):

  def _run(self, x, y):
    outputs = np.setdiff1d(x, y)
    return outputs

