from collections import namedtuple

import numpy as np

from .generic_ops import Const
from .initializers import (
    GlorotNormalInitializer, GlorotUniformInitializer, HeNormalInitializer,
    HeUniformInitializer, OnesInitializer, RandomUniformInitializer,
    TruncatedNormalInitializer, ZerosInitializer,
)
from .math_ops import Add
from .resource_ops import AddToVariable, CreateVariable, ReadVariable
from .wrappers import relu, tanh, sigmoid, leaky_relu

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "leaky_relu": leaky_relu,
}

INITIALIZERS = {
    "glorot_uniform": GlorotUniformInitializer,
    "zeros": ZerosInitializer,
    "truncated_normal": TruncatedNormalInitializer,
    "random_uniform": RandomUniformInitializer,
    "ones": OnesInitializer,
    "glorot_normal": GlorotNormalInitializer,
    "he_uniform": HeUniformInitializer,
    "he_normal": HeNormalInitializer,
}

Variable = namedtuple("Variable", ["weight", "handle", "trainable"])


class Layer(object):

  def __init__(
      self, activation=None, kernel_initializer=None, bias_initializer=None,
  ):
    self._variables = []
    if callable(activation):
      self._activation = activation
    elif isinstance(activation, str):
      self._activation = ACTIVATIONS[activation]
    else:
      self._activation = None

    if callable(kernel_initializer):
      self._kernel_initializer = kernel_initializer
    elif isinstance(kernel_initializer, str):
      self._kernel_initializer = INITIALIZERS[kernel_initializer]()
    else:
      self._kernel_initializer = None

    if callable(bias_initializer):
      self._bias_initializer = bias_initializer
    elif isinstance(bias_initializer, str):
      self._bias_initializer = INITIALIZERS[bias_initializer]()
    else:
      self._bias_initializer = None

  @property
  def variables(self):
    if not len(self._variables):
      for k, v in self.__dict__.items():
        if isinstance(v, Layer):
          self._variables.extend(v._variables)
    return self._variables

  def get_variable_weight(self, index):
    assert index in self._variables
    runtime = self._variables[index].weight.op._graph._runtime
    return runtime.get_variable_value(
        runtime.get_tensor_value(self._variables[index].handle).item().id,
    )

  def _build(self, shape_list, init_fn_list, flag_list, trainable_list):
    for shape, init_fn, flag, trainable in zip(
        shape_list, init_fn_list, flag_list, trainable_list,
    ):
      if not flag:
        continue

      create_var = CreateVariable(shape, init_fn)
      create_var.run()

      read_var = ReadVariable(input_list=[create_var.output(0)]).output(0)

      runtime = create_var._graph._runtime

      self._variables.append(
          Variable(
              weight=read_var, handle=create_var.output(0), trainable=trainable,
          ),
      )

  def save_variable_weights(self, filename):
    assert len(self.variables) > 0
    runtime = self.variables[0].weight.op._graph._runtime
    vids = [
        runtime.get_tensor_value(v.handle).item().id for v in self.variables
    ]
    variable_values = np.asarray(
        [
            runtime.get_variable_value(vid) for vid in vids
        ],
                                     dtype="object",
    )
    return variable_values

  def load_variable_weights(self, filename):
    weights = np.load(filename, allow_pickle=True)
    assert len(weights) == len(self.variables)

    runtime = self.variables[0].weight.op._graph._runtime
    for i, v in enumerate(self.variables):
      vid = runtime.get_tensor_value(v.handle).item().id
      runtime.set_variable_value(vid, weights[i])


class Dense(Layer):

  def __init__(
      self,
      units,
      activation=None,
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
  ):
    super(
        Dense,
        self,
    ).__init__(activation, kernel_initializer, bias_initializer)
    self._units = units
    self._use_bias = use_bias

  def build(self, input_shape):
    if not len(self._variables):
      in_channels = input_shape[1]

      shape_list = [[in_channels, self._units], [self._units]]
      init_fn_list = [
          lambda shape: self._kernel_initializer(shape),
          lambda shape: self._bias_initializer(shape),
      ]
      flag_list = [True, self._use_bias]
      trainable_list = [True] * 2
      self._build(shape_list, init_fn_list, flag_list, trainable_list)

  def __call__(self, inputs):
    from .math_ops import MatMul

    self.build(inputs.shape.raw_shape)
    filters = self._variables[0].weight
    outputs = MatMul(input_list=[inputs, filters]).output(0)

    if self._use_bias:
      bias = self._variables[1].weight
      outputs = Add(input_list=[outputs, bias]).output(0)
    if self._activation:
      outputs = self._activation(outputs)
    return outputs


class Conv2D(Layer):

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding="SAME",
      activation=None,
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
  ):
    super(
        Conv2D,
        self,
    ).__init__(activation, kernel_initializer, bias_initializer)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias

  def build(self, input_shape):
    if not len(self._variables):
      filters_shape = list(self._kernel_size) + [input_shape[3], self._filters]

      shape_list = [filters_shape, [self._filters]]
      init_fn_list = [
          lambda shape: self._kernel_initializer(shape),
          lambda shape: self._bias_initializer(shape),
      ]
      flag_list = [True, self._use_bias]
      trainable_list = [True] * 2
      self._build(shape_list, init_fn_list, flag_list, trainable_list)

  def __call__(self, inputs):
    from .nn_ops import Conv2D as Conv2dOp
    self.build(inputs.shape.raw_shape)

    filters = self._variables[0].weight
    outputs = Conv2dOp(
        input_list=[inputs, filters],
        strides=self._strides,
        padding=self._padding,
    ).output(0)
    if self._use_bias:
      bias = self._variables[1].weight
      outputs = Add(input_list=[outputs, bias]).output(0)
    if self._activation:
      outputs = self._activation(outputs)
    return outputs


class Conv2DTranspose(Layer):

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding="SAME",
      activation=None,
      use_bias=True,
      outputs_shape=None,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
  ):
    super(
        Conv2DTranspose,
        self,
    ).__init__(activation, kernel_initializer, bias_initializer)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias
    self._outputs_shape = outputs_shape

  def build(self, input_shape):

    if not len(self._variables):
      filters_shape = list(self._kernel_size) + [self._filters, input_shape[3]]

      shape_list = [filters_shape, [self._filters]]
      init_fn_list = [
          lambda shape: self._kernel_initializer(shape),
          lambda shape: self._bias_initializer(shape),
      ]
      flag_list = [True, self._use_bias]
      trainable_list = [True] * 2
      self._build(shape_list, init_fn_list, flag_list, trainable_list)

  def _infer_spatial_size(self, input_size, kernel_size, stride_size):
    if self._padding == "SAME":
      out_size = input_size * stride_size
    else:
      out_size = input_size * stride_size + max(kernel_size - stride_size, 0)
    return out_size

  def __call__(self, inputs):
    from .nn_ops import Conv2DBackpropInput
    self.build(inputs.shape.raw_shape)

    filters = self._variables[0].weight

    out_height = self._infer_spatial_size(
        inputs.shape.raw_shape[1], self._kernel_size[0], self._strides[0],
    )
    out_width = self._infer_spatial_size(
        inputs.shape.raw_shape[2], self._kernel_size[1], self._strides[1],
    )

    if self._outputs_shape is None:
      outputs_shape = inputs.shape.raw_shape[
          0
      ], out_height, out_width, self._filters
    else:
      outputs_shape = self._outputs_shape
    outputs_shape = Const(
        value=np.asarray(outputs_shape, dtype="int32"),
    ).output(0)

    outputs = Conv2DBackpropInput(
        input_list=[filters, inputs, outputs_shape],
        strides=self._strides,
        padding=self._padding,
    ).output(0)
    if self._use_bias:
      bias = self._variables[1].weight
      outputs = Add(input_list=[outputs, bias]).output(0)
    if self._activation:
      outputs = self._activation(outputs)
    return outputs


class BatchNormalization(Layer):

  def __init__(
      self,
      axis=-1,
      momentum=0.99,
      epsilon=0.0001,
      beta_initializer="zeros",
      gamma_initializer="ones",
      moving_mean_initializer="zeros",
      moving_variance_initializer="ones",
  ):
    super(BatchNormalization, self).__init__()
    self._axis = axis
    self._momentum = momentum
    self._epsilon = epsilon
    self._beta_initializer = INITIALIZERS[beta_initializer]()
    self._gamma_initializer = INITIALIZERS[gamma_initializer]()
    self._moving_mean_initializer = INITIALIZERS[moving_mean_initializer]()
    self._moving_variance_initializer = INITIALIZERS[
        moving_variance_initializer
    ]()

  def build(self, input_shape):
    if not len(self._variables):
      ndims = len(input_shape)
      self._axis %= ndims

      shape_list = [[input_shape[self._axis]]] * 4
      init_fn_list = [
          lambda shape: self._beta_initializer(shape),
          lambda shape: self._gamma_initializer(shape),
          lambda shape: self._moving_mean_initializer(shape),
          lambda shape: self._moving_variance_initializer(shape),
      ]
      flag_list = [True] * 4
      trainable_list = [True] * 2 + [False] * 2

      self._build(shape_list, init_fn_list, flag_list, trainable_list)

  def __call__(self, inputs, training=False):
    from .math_ops import Mul, Rsqrt, Sub

    self.build(inputs.shape.raw_shape)
    reduction_indices = [
        i for i in range(inputs.shape.ndims) if i != self._axis
    ]
    beta = self._variables[0].weight
    gamma = self._variables[1].weight

    if training:
      from .math_ops import Mean, Mul, SquaredDifference

      # compute batch mean and variance and use them for normalization
      axis = Const(value=np.asarray(reduction_indices, dtype="int32")).output(0)
      mean = Mean(input_list=[inputs, axis]).output(0)
      variance = Mean(
          input_list=[
              SquaredDifference(input_list=[inputs, mean]).output(0), axis,
          ],
      ).output(0)

      # update moving mean and variance
      moving_mean = self._variables[2].weight
      moving_variance = self._variables[3].weight

      const = Const(
          value=np.asarray(1 - self._momentum, dtype="float32"),
      ).output(0)
      delta_moving_mean = Mul(
          input_list=[
              const,
              Sub(input_list=[mean, moving_mean]).output(0),
          ],
      ).output(0)
      delta_moving_variance = Mul(
          input_list=[
              const,
              Sub(input_list=[variance, moving_variance]).output(0),
          ],
      ).output(0)

      update_moving_mean = AddToVariable(
          input_list=[self._variables[2].handle, delta_moving_mean],
      )
      update_moving_variance = AddToVariable(
          input_list=[self._variables[3].handle, delta_moving_variance],
      )

    else:
      # use moving mean and variance for normalization
      mean = self._variables[2].weight
      variance = self._variables[3].weight

    epsilon = Const(value=np.asarray(self._epsilon, dtype="float32")).output(0)
    add = Add(input_list=[variance, epsilon]).output(0)
    rsqrt = Rsqrt(input_list=[add]).output(0)
    mul = Mul(input_list=[rsqrt, gamma]).output(0)
    mul1 = Mul(input_list=[mul, inputs]).output(0)
    mul2 = Mul(input_list=[mul, mean]).output(0)
    sub = Sub(input_list=[beta, mul2]).output(0)
    outputs = Add(input_list=[mul1, sub]).output(0)

    if training:
      from .data_flow_ops import Identity
      outputs = Identity(
          input_list=[outputs],
          # make sure moving mean and variances are updated
          dependent_ops=[update_moving_mean, update_moving_variance],
      ).output(0)

    return outputs
