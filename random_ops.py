import numpy as np

from .mixins import _ShapeAsIs, _TensorShapeAsInput
from .operation import Operation
from .tensor_shape import TensorShape


class RandomUniform(Operation, _TensorShapeAsInput):

  def __init__(self, input_list, name=None, graph=None):
    super(RandomUniform, self).__init__(
        graph=graph,
        input_list=input_list,
        name=name,
    )

  def _run(self, shape):
    outputs = np.random.uniform(size=shape).astype("float32")
    return outputs


class RandomStandardNormal(Operation, _TensorShapeAsInput):

  def __init__(self, input_list, name=None, graph=None):
    super(RandomStandardNormal, self).__init__(
        graph=graph,
        input_list=input_list,
        name=name,
    )

  def _run(self, shape):
    outputs = np.random.randn(*shape).astype("float32")
    return outputs
