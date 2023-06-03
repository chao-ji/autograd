"""
"""
import numpy as np

from operation import Operation
from array_ops import Reshape 
from math_ops import Mean, Sum



class BroadcastGradientArgs(Operation):

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

  def backprop(self, bwval_list):
    return


class Add(Operation):

  def _run(self, x, y):
    outputs = x + y 
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    input_list = [(x_shape, 0), (y_shape, 0)]

    bga = BroadcastGradientArgs(graph=self._graph, input_list=input_list)

    input_list = [(bwval_op, tensor_index), (bga, 0)]
    sum_x = Sum(graph=self._graph, input_list=input_list) 
    input_list = [(bwval_op, tensor_index), (bga, 1)]
    sum_y = Sum(graph=self._graph, input_list=input_list)

    bp_x = Reshape(graph=self._graph, input_list=[(sum_x, 0), (x_shape, 0)])
    bp_y = Reshape(graph=self._graph, input_list=[(sum_y, 0), (y_shape, 0)]) 

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])

    return bp_x, bp_y


class Neg(Operation):
  def _run(self, x):
    return -x

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]
    bp_inputs = Neg(graph=self._graph, input_list=[(bwval_op, tensor_index)])
    self._input_list[0][0].backprop(bwval_list=[(bp_inputs, 0)]) 
    return bp_inputs


class Sub(Operation):
  def _run(self, x, y):
    outputs = x - y
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    input_list = [(x_shape, 0), (y_shape, 0)]

    bga = BroadcastGradientArgs(graph=self._graph, input_list=input_list)

    input_list = [(bwval_op, tensor_index), (bga, 0)]
    sum_x = Sum(graph=self._graph, input_list=input_list)

    neg_grad = Neg(graph=self._graph, input_list=[(bwval_op, tensor_index)])
    input_list = [(neg_grad, 0), (bga, 1)]
    sum_y = Sum(graph=self._graph, input_list=input_list)

    bp_x = Reshape(graph=self._graph, input_list=[(sum_x, 0), (x_shape, 0)])
    bp_y = Reshape(graph=self._graph, input_list=[(sum_y, 0), (y_shape, 0)])

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])

    return bp_x, bp_y


class Mul(Operation):
  def _run(self, x, y):
    outputs = x * y
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    input_list = [(x_shape, 0), (y_shape, 0)]

    bga = BroadcastGradientArgs(graph=self._graph, input_list=input_list)     

    x_grad = Mul(graph=self._graph, input_list=[(bwval_op, tensor_index), self._input_list[1]])
    y_grad = Mul(graph=self._graph, input_list=[(bwval_op, tensor_index), self._input_list[0]])

    sum_x = Sum(graph=self._graph, input_list=[(x_grad, 0), (bga, 0)])

    sum_y = Sum(graph=self._graph, input_list=[(y_grad, 0), (bga, 1)])

    bp_x = Reshape(graph=self._graph, input_list=[(sum_x, 0), (x_shape, 0)])
    bp_y = Reshape(graph=self._graph, input_list=[(sum_y, 0), (y_shape, 0)])

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])

    return bp_x, bp_y    


class RealDiv(Operation):
  def _run(self, x, y):
    outputs = x / y
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]

    neg = Neg(graph=self._graph, input_list=[self._input_list[0]])
    div = RealDiv(graph=self._graph, input_list=[(bwval_op, tensor_index), self._input_list[1]])
    div1 = RealDiv(graph=self._graph, input_list=[(neg, 0), self._input_list[1]])
    div2 = RealDiv(graph=self._graph, input_list=[(div1, 0), self._input_list[1]])
    mul = Mul(graph=self._graph, input_list=[(div2, 0), (bwval_op, tensor_index)])

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    input_list = [(x_shape, 0), (y_shape, 0)]
    bga = BroadcastGradientArgs(graph=self._graph, input_list=input_list)
    x_grad = Sum(graph=self._graph, input_list=[(div, 0), (bga, 0)])
    y_grad = Sum(graph=self._graph, input_list=[(mul, 0), (bga, 1)])

    bp_x = Reshape(graph=self._graph, input_list=[(x_grad, 0), (x_shape, 0)])
    bp_y = Reshape(graph=self._graph, input_list=[(y_grad, 0), (y_shape, 0)])

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])
    return bp_x, bp_y


class FloorDiv(Operation):
  def _run(self, x, y):
    outputs = x // y
    return outputs

  def backprop(self, bwval_list):
    return


class FloorMod(Operation):
  def _run(self, x, y):
    outputs = x % y
    return outputs

  def backprop(self, bwval_list):
    bwval_op, tensor_index = bwval_list[0]

    div = FloorDiv(input_list=self._input_list, graph=self._graph)
    neg = Neg(input_list=[(div, 0)], graph=self._graph)

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    input_list = [(x_shape, 0), (y_shape, 0)]
    bga = BroadcastGradientArgs(graph=self._graph, input_list=input_list)
    mul = Mul(input_list=bwval_list + [(neg, 0)], graph=self._graph)

    x_grad = Sum(graph=self._graph, input_list=bwval_list+[(bga, 0)])
    y_grad = Sum(graph=self._graph, input_list=[(mul, 0)]+[(bga, 1)])

    bp_x = Reshape(graph=self._graph, input_list=[(x_grad, 0), (x_shape, 0)])
    bp_y = Reshape(graph=self._graph, input_list=[(y_grad, 0), (y_shape, 0)])

    self._input_list[0][0].backprop(bwval_list=[(bp_x, 0)])
    self._input_list[1][0].backprop(bwval_list=[(bp_y, 0)])
    return bp_x, bp_y


class Maximum(Operation):
  def _run(self, x, y):
    outputs = np.maximum(x, y)
    return outputs

  def backprop(self, bwval_list):
    from math_ops import GreaterEqual, Sum
    from array_ops import Reshape
    from data_flow_ops import Select

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    ge = GreaterEqual(input_list=[self._input_list[0], self._input_list[1]], graph=self._graph) 

    zeros = bwval_list[0][0].get_zeros_op()

    select = Select(input_list=[(ge, 0), bwval_list[0], (zeros, 0)], graph=self._graph)
    select1 = Select(input_list=[(ge, 0), (zeros, 0), bwval_list[0]], graph=self._graph)
    
    bga = BroadcastGradientArgs(input_list=[(x_shape, 0), (y_shape, 0)], graph=self._graph)

    sum0 = Sum(input_list=[(select, 0), (bga, 0)], graph=self._graph)
    sum1 = Sum(input_list=[(select1, 0), (bga, 1)], graph=self._graph)

    bp_x = Reshape(input_list=[(sum0, 0), (x_shape, 0)], graph=self._graph)
    bp_y = Reshape(input_list=[(sum1, 0), (y_shape, 0)], graph=self._graph)

    return bp_x, bp_y

 
class DivNoNan(Operation):
  def _run(self, x, y):
    outputs = np.where(y != 0, x / y, 0)
    return outputs

  def backprop(self, bwval_list):
    div = DivNoNan(
        input_list=[bwval_list[0], self._input_list[1]], graph=self._graph
    )
    neg = Neg(input_list=[self._input_list[0]], graph=self._graph)
    div1 = DivNoNan(input_list=[(neg, 0), self._input_list[1]], graph=self._graph)
    div2 = DivNoNan(input_list=[(div1, 0), self._input_list[1]], graph=self._graph)

    mul = Mul(input_list=[bwval_list[0], (div2, 0)], graph=self._graph)

    x_shape = self._input_list[0][0].get_shape_op()
    y_shape = self._input_list[1][0].get_shape_op()

    bga = BroadcastGradientArgs(input_list=[(x_shape, 0), (y_shape, 0)], graph=self._graph)
 
    sum0 = Sum(input_list=[(div, 0), (bga, 0)], graph=self._graph)
    sum1 = Sum(input_list=[(mul, 0), (bga, 1)], graph=self._graph)

    bp_x = Reshape(input_list=[(sum0, 0), (x_shape, 0)], graph=self._graph)
    bp_y = Reshape(input_list=[(sum1, 0), (y_shape, 0)], graph=self._graph)

    return bp_x, bp_y

