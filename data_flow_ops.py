"""Data flow related Operations."""
import numpy as np

from operation import Operation
from origin_ops import Const


class DynamicStitch(Operation):
  def __init__(self, input_list, graph=None, accumulate=False, name=None):
    super(DynamicStitch, self).__init__(
        graph=graph, name=name, input_list=input_list
    )
    self._accumulate = accumulate

  def _run(self, *inputs_list):
    size = len(inputs_list) // 2
    indices, data = inputs_list[:size], inputs_list[size:]

    data = np.concatenate(
        [data[i].reshape((-1,) + data[i].shape[indices[i].ndim:]) 
            for i in range(len(data))
        ]
    )
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
        bp_data = Gather(input_list=[params, indices])
        out_grad_tensors.append((bp_data, 0))
    return out_grad_tensors


class Gather(Operation):
  """Always gather from axis=0"""
  def _run(self, params, indices):
    outputs = np.take(params, indices, axis=0)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Slice, Concat, Fill
    from arithmetic_ops import Sub

    with self._graph.as_default_graph():
      ds = DynamicStitch(
          accumulate=True,
          input_list=[self._input_list[1], in_grad_tensors[0]]
      )

      slice0 = Slice(input_list=[
          (ds.get_shape_op(tensor_index=0), 0),
          (Const(value=np.asarray([0])), 0),
          (Const(value=np.asarray([1])), 0)
        ])

      op, tensor_index = self._input_list[0]
      slice1 = Slice(input_list=[
          (op.get_shape_op(tensor_index=tensor_index), 0),
          (Const(value=np.asarray([0])), 0),
          (Const(value=np.asarray([1])), 0)
        ])

      sub = Sub(input_list=[(slice1, 0), (slice0, 0)]) 

      slice2 = Slice(input_list=[
          (ds.get_shape_op(tensor_index=0), 0),
          (Const(value=np.asarray([1])), 0),
          (Const(value=np.asarray([-1])), 0)
        ])

      shape = Concat(input_list=[(
          Const(value=np.asarray(0)), 0),
          (sub, 0),
          (slice2, 0)
        ])
  
      fill = Fill(input_list=[
          (shape, 0),
          (Const(value=np.asarray(0)), 0)])

      concat = Concat(input_list=[
          (Const(value=np.asarray(0)), 0),
          (ds, 0),
          (fill, 0)]
      )
      out_grad_tensors = [(concat, 0)]
    return out_grad_tensors



class BroadcastTo(Operation):
  """"""
  def _run(self, inputs, target_shape):
    shape = np.pad(
        inputs.shape,
        [len(target_shape) - len(inputs.shape), 0],
        constant_values=1
    )
    multiples = np.where(shape != target_shape, target_shape, 1)
    outputs = np.tile(inputs, multiples)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from arithmetic_ops import BroadcastGradientArgs, Sum
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0]
      shape = op.get_shape_op(tensor_index=tensor_index)

      bga = BroadcastGradientArgs(
          input_list=[(shape, 0), self._input_list[1]],
      )
      sum0 = Sum(input_list=[in_grad_tensors[0], (bga, 0)])

      bp_inputs = Reshape(input_list=[(sum0, 0), (shape, 0)])
      out_grad_tensors = [(bp_inputs, 0)]

    return out_grad_tensors



class Select(Operation):

  def _run(self, condition, x, y):
    outputs = np.where(condition, x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from arithmetic_ops import BroadcastGradientArgs
    from math_ops import Sum
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op_x, tensor_index_x = self._input_list[1]
      op_y, tensor_index_y = self._input_list[2]

      shape = op_x.get_shape_op(tensor_index=tensor_index_x)
      shape1 = op_y.get_shape_op(tensor_index=tensor_index_y)
      shape2 = self.get_shape_op(tensor_index=0)
      select = Select(
          input_list=[
              self._input_list[0],
              in_grad_tensors[0],
              (Const(value=np.asarray(0)), 0)
          ]
      )
      select1 = Select(
          input_list=[
              self._input_list[0],
              (Const(value=np.asarray(0)), 0),
              in_grad_tensors[0]
          ]
      )
      bga = BroadcastGradientArgs(input_list=[(shape, 0), (shape2, 0)])
      bga1 = BroadcastGradientArgs(input_list=[(shape1, 0), (shape2, 0)])
      sum0 = Sum(input_list=[(select, 0), (bga, 0)])
      sum1 = Sum(input_list=[(select1, 0), (bga1, 0)])
      bp_x = Reshape(input_list=[(sum0, 0), (shape, 0)])
      bp_y = Reshape(input_list=[(sum1, 0), (shape1, 0)]) 
      out_grad_tensors = [(bp_x, 0), (bp_y, 0)]
    return out_grad_tensors
 
