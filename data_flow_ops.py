import numpy as np

from operation import Operation
from origin_ops import Const


class DynamicStitch(Operation):
  def __init__(self, input_list, graph, accumulate=False, name=None):
    super(DynamicStitch, self).__init__(graph=graph, name=name, input_list=input_list)
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

  def backprop(self, bwval_list):
    size = len(self._input_list) // 2

    bp_data_list = []
    for i, (indices, params) in enumerate(zip(self._input_list[:size], bwval_list * size)):
      bp_data = Gather(graph=self._graph, input_list=[params, indices])

      self._input_list[size+i][0].backprop(bwval_list=[(bp_data, 0)])

      bp_data_list.append(bp_data)
    return bp_data_list


class Gather(Operation):
  """Always gather from axis=0"""
  def _run(self, params, indices):
    outputs = np.take(params, indices, axis=0)
    return outputs

  def backprop(self, bwval_list):
    from array_ops import Slice, Concat, Fill
    from arithmetic_ops import Sub
    bwval_op, tensor_index = bwval_list[0]

    bp_params = DynamicStitch(
        accumulate=True,
        graph=self._graph,
        input_list=[self._input_list[1]] + [(bwval_op, tensor_index)]
    )

    slice0 = Slice(input_list=[
        (bp_params.get_shape_op(), 0),
        (Const(value=np.asarray([0]), graph=self._graph), 0),
        (Const(value=np.asarray([1]), graph=self._graph), 0)
      ], graph=self._graph)

    slice1 = Slice(input_list=[
        (self._input_list[0][0].get_shape_op(), 0),
        (Const(value=np.asarray([0]), graph=self._graph), 0),
        (Const(value=np.asarray([1]), graph=self._graph), 0)
      ], graph=self._graph)

    sub = Sub(input_list=[(slice1, 0), (slice0, 0)], graph=self._graph) 

    slice2 = Slice(input_list=[
        (bp_params.get_shape_op(), 0),
        (Const(value=np.asarray([1]), graph=self._graph), 0),
        (Const(value=np.asarray([-1]), graph=self._graph), 0)
      ], graph=self._graph)

    shape = Concat(input_list=[(
        Const(value=np.asarray(0), graph=self._graph), 0),
        (sub, 0),
        (slice2, 0)
      ], graph=self._graph)
  
    fill = Fill(input_list=[
        (shape, 0),
        (Const(value=np.asarray(0), graph=self._graph), 0)], graph=self._graph)

    bp_params = Concat(input_list=[
        (Const(value=np.asarray(0), graph=self._graph), 0),
        (bp_params, 0),
        (fill, 0)], graph=self._graph)

    return bp_params #, fill



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

  def backprop(self, bwval_list):
    from arithmetic_ops import BroadcastGradientArgs, Sum
    from array_ops import Reshape

    shape = self._input_list[0][0].get_shape_op()

    bga = BroadcastGradientArgs(input_list=[(shape, 0), self._input_list[1]], graph=self._graph)

    sum0 = Sum(input_list=[bwval_list[0], (bga, 0)], graph=self._graph)

    bp_inputs = Reshape(input_list=[(sum0, 0), (shape, 0)], graph=self._graph)
    return bp_inputs



class Select(Operation):

  def _run(self, condition, x, y):
    outputs = np.where(condition, x, y)
    return outputs


  def backprop(self, bwval_list):
    from arithmetic_ops import BroadcastGradientArgs
    from math_ops import Sum
    from array_ops import Reshape

    x_shape = self._input_list[1][0].get_shape_op()
    y_shape = self._input_list[2][0].get_shape_op()
    shape = self.get_shape_op()

    select = Select(input_list=[self._input_list[0], bwval_list[0], (Const(value=np.asarray(0), graph=self._graph), 0)], graph=self._graph)
    select1 = Select(input_list=[self._input_list[0], (Const(value=np.asarray(0), graph=self._graph), 0), bwval_list[0]], graph=self._graph)


    bga = BroadcastGradientArgs(input_list=[(x_shape, 0), (shape, 0)], graph=self._graph)
    bga1 = BroadcastGradientArgs(input_list=[(y_shape, 0), (shape, 0)], graph=self._graph)
    
    sum0 = Sum(input_list=[(select, 0), (bga, 0)], graph=self._graph)
    sum1 = Sum(input_list=[(select1, 0), (bga1, 0)], graph=self._graph)

    bp_x = Reshape(input_list=[(sum0, 0), (x_shape, 0)], graph=self._graph)
    bp_y = Reshape(input_list=[(sum1, 0), (y_shape, 0)], graph=self._graph)  
    return bp_x, bp_y

