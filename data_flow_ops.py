"""Data flow related Operations."""
import numpy as np

from operation import Operation
from generic_ops import Const


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
        bp_data = Gather(input_list=[params, indices, Const(value=np.asarray(0)).output(0)])
        out_grad_tensors.append(bp_data.output(0))

    return out_grad_tensors

  def _get_bp_indices(self):
    size = len(self._input_list) // 2
    bp_indices = set(range(size, size * 2))
    return bp_indices


class Gather(Operation):
  def _run(self, params, indices, axis):
    outputs = np.take(params, indices, axis=axis.item())
    return outputs

  def _grad_func(self, in_grad_tensors):
    from array_ops import Slice, Concat, Fill, ExpandDims, Range, Transpose
    from math_ops import Sub, Add, FloorMod

    with self._graph.as_default_graph():

      zero_array_tensor = Const(value=np.asarray([0])).output(0)
      zero_scalar_tensor = Const(value=np.asarray(0)).output(0)
      one_array_tensor = Const(value=np.asarray([1])).output(0)
      one_scalar_tensor = Const(value=np.asarray(1)).output(0)

      op, tensor_index = (
          self._input_list[0].op,
          self._input_list[0].tensor_index
      )
      mod_tensor = FloorMod(
          input_list=[
              self._input_list[2],
              op.get_rank_tensor(tensor_index=tensor_index)
          ]).output(0)
      op, tensor_index = in_grad_tensors[0].op, in_grad_tensors[0].tensor_index
      range0 = Range(
          input_list=[
              mod_tensor,
              op.get_rank_tensor(tensor_index=tensor_index),
              one_scalar_tensor
          ]
      )
      range1 = Range(
          input_list=[
              zero_scalar_tensor,
              mod_tensor,
              one_scalar_tensor
          ]
      )
      perm = Concat(input_list=[
          zero_scalar_tensor,
          range0.output(0),
          range1.output(0),
        ])
      transpose = Transpose(input_list=[in_grad_tensors[0], perm.output(0)])
      ds = DynamicStitch(
          accumulate=True,
          input_list=[self._input_list[1], transpose.output(0)]
      )

      rank_tensor = ds.get_rank_tensor(tensor_index=0)
      sub_tensor = Sub(input_list=[rank_tensor, mod_tensor]).output(0)
      range2 = Range(input_list=[sub_tensor, rank_tensor, one_scalar_tensor])
      range3 = Range(
          input_list=[zero_scalar_tensor, sub_tensor, one_scalar_tensor]
      )

      perm1 = Concat(input_list=[
          zero_scalar_tensor,
          range2.output(0),
          range3.output(0),
        ])

      transpose1 = Transpose(input_list=[ds.output(0), perm1.output(0)])

      shape_tensor = transpose1.get_shape_tensor(tensor_index=0) 
      ed_tensor = ExpandDims(
              input_list=[
                  mod_tensor,
                  zero_scalar_tensor
              ]
          ).output(0)

      slice0 = Slice(input_list=[
          shape_tensor,
          ed_tensor,
          one_array_tensor
        ])
 
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      slice1 = Slice(input_list=[
          op.get_shape_tensor(tensor_index=tensor_index),
          ed_tensor,
          one_array_tensor
          ])
 
      sub = Sub(input_list=[slice1.output(0), slice0.output(0)]) 
 
      slice2 = Slice(input_list=[
          shape_tensor,
          zero_array_tensor,
          ed_tensor
        ])

      slice3 = Slice(input_list=[
          shape_tensor,
          Add(input_list=[ed_tensor, one_array_tensor]).output(0),
          Const(value=np.asarray([-1])).output(0)
        ])

      concat = Concat(input_list=[
          zero_scalar_tensor,
          slice2.output(0),
          sub.output(0),
          slice3.output(0)
        ])
  
      fill = Fill(input_list=[
          concat.output(0),
          zero_scalar_tensor
        ]
      )

      concat1 = Concat(input_list=[
          mod_tensor,
          transpose1.output(0),
          fill.output(0)]
      )


      out_grad_tensors = [concat1.output(0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


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
    from math_ops import BroadcastGradientArgs, Sum
    from array_ops import Reshape

    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      shape_tensor = op.get_shape_tensor(tensor_index=tensor_index)

      bga = BroadcastGradientArgs(
          input_list=[shape_tensor,
                      self._input_list[1]
          ],
      )
      sum0 = Sum(input_list=[in_grad_tensors[0], bga.output(0)])

      bp_inputs = Reshape(input_list=[sum0.output(0), shape_tensor])
      out_grad_tensors = [bp_inputs.output(0)]

    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class Select(Operation):

  def _run(self, condition, x, y):
    outputs = np.where(condition, x, y)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from math_ops import Sum, BroadcastGradientArgs
    from array_ops import Reshape

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
              self._input_list[0],
              in_grad_tensors[0],
              Const(value=np.asarray(0)).output(0)
          ]
      )
      select1 = Select(
          input_list=[
              self._input_list[0],
              Const(value=np.asarray(0)).output(0),
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


class StopGradient(Operation):
  def _run(self, inputs):
    outputs = inputs
    return outputs
