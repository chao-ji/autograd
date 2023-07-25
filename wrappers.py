"""Functions that wrap raw operations."""
import numpy as np

from .array_ops import (
    Concat, ExpandDims, Fill, Pack, Pad, Range, Reshape, Slice, Squeeze,
    StridedSlice, Tile, Transpose, Unpack,
)
from .data_flow_ops import (
    BroadcastTo, DynamicStitch, Gather, Select, StopGradient,
)
from .generic_ops import Const, OnesLike, Rank, Shape, Size, ZerosLike
from .math_ops import (
    Add, BatchMatMul, Cumprod, Cumsum, Equal, Exp, FloorDiv, FloorMod, Greater,
    GreaterEqual, Less, LessEqual, Log, Log1p, MatMul, Maximum, Mean, Minimum,
    Mul, Neg, NotEqual, Prod, RealDiv, Reciprocal, Rsqrt, Sqrt, Square,
    SquaredDifference, Sub, Sum,
)
from .nn_ops import (
    AvgPool2D, Conv2D, Conv2DBackpropInput, LeakyRelu, LogSoftmax, MaxPool2D,
    Relu, Sigmoid, Softmax, SoftmaxCrossEntropyWithLogits, Tanh,
)
from .random_ops import RandomStandardNormal, RandomUniform
from .resource_ops import Placeholder

_FLOAT_TYPES = np.float16, np.float32, np.float64, np.float128, np.float_, float
_INT_TYPES = np.int0, np.int8, np.int16, np.int32, np.int64, np.int_, int,

_NUMERIC_TYPES = _FLOAT_TYPES + _INT_TYPES


def _tensorize_input(argpositions=[], argkeys=[], all_posargs=False):
  """Convert positional and kwargs of decorated functions to tensors (if they
  are not yet).

  Args:
    argpositions (List[int]): indices of positional args to be tensorized.
    argkeys (List[str]): list of keys of kwargs to be tensorized.
    all_posargs (bool): whether to tensorize all positional args. If True, all
      positional args will be tensorized.

  Returns:
    parameterized_wrapper (callable): decorated function.
  """
  from .tensor import Tensor

  def parameterized_wrapper(func):

    def wrapper(*args, **kwargs):
      new_args = []
      new_kwargs = dict()
      for i, arg in enumerate(args):
        if not isinstance(arg, Tensor) and (all_posargs or i in argpositions):
          arg = np.asarray(arg)
          if arg.dtype in _FLOAT_TYPES:
            arg = arg.astype("float32")
          elif arg.dtype in _INT_TYPES:
            arg = arg.astype("int32")

          assert arg.dtype in _NUMERIC_TYPES
          new_args.append(Const(value=arg).output(0))
        else:
          new_args.append(arg)
      for k, v in kwargs.items():
        if not isinstance(v, Tensor) and k in argkeys:
          v = np.asarray(v)
          if v.dtype in _FLOAT_TYPES:
            v = v.astype("float32")
          elif v.dtype in _INT_TYPES:
            v = v.astype("int32")

          assert v.dtype in _NUMERIC_TYPES
          new_kwargs[k] = Const(value=v).output(0)
        else:
          new_kwargs[k] = v
      return func(*new_args, **new_kwargs)

    return wrapper

  return parameterized_wrapper


@_tensorize_input(argpositions=(0,), argkeys=("value",))
def constant(value, name=None):
  return value


@_tensorize_input(argpositions=(0,), argkeys=("inputs",))
def zeros_like(inputs, name=None):
  tensor = ZerosLike(input_list=[inputs], name=name).output(0)
  return tensor


@_tensorize_input(argpositions=(0,), argkeys=("inputs",))
def ones_like(inputs, name=None):
  tensor = OnesLike(input_list=[inputs], name=name).output(0)
  return tensor


@_tensorize_input(argpositions=(0,), argkeys=("shape",))
def zeros(shape, name=None):
  zero_scalar = Const(value=np.asarray(0, dtype="float32")).output(0)
  tensor = Fill(input_list=[shape, zero_scalar], name=name).output(0)
  return tensor


@_tensorize_input(argpositions=(0,), argkeys=("shape",))
def ones(shape, name=None):
  one_scalar = Const(value=np.asarray(1, dtype="float32")).output(0)
  tensor = Fill(input_list=[shape, one_scalar], name=name).output(0)
  return tensor


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def shape(tensor, name=None):
  tensor_shape = Shape(input_list=[tensor], name=name).output(0)
  return tensor_shape


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def rank(tensor, name=None):
  tensor_rank = Rank(input_list=[tensor], name=name).output(0)
  return tensor_rank


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def size(tensor, name=None):
  tensor_size = Size(input_list=[tensor], name=name).output(0)
  return tensor_size


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "shape"))
def reshape(tensor, shape, name=None):
  reshaped = Reshape(input_list=[tensor, shape], name=name).output(0)
  return reshaped


def transpose(tensor, perm=None, name=None):
  if perm is None:
    if tensor.shape.level > 0:
      perm = np.arange(0, tensor.shape.ndims)[::-1]
    else:
      minus_one_scalar = Const(value=np.asarray(-1, dtype="int32")).output(0)
      rank = tensor.op.get_rank_tensor(tensor_index=tensor.tensor_index)
      perm = Range(
          input_list=[
              Add(input_list=[rank, minus_one_scalar]).output(0),
              minus_one_scalar,
              minus_one_scalar,
          ],
      ).output(0)
  return _transpose(tensor, perm, name=name)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "perm"))
def _transpose(tensor, perm, name=None):
  transposed = Transpose(input_list=[tensor, perm], name=name).output(0)
  return transposed


@_tensorize_input(argpositions=(0, 1, 2), argkeys=("start", "limit", "delta"))
def _range(start, limit, delta, name=None):
  range_ = Range(input_list=[start, limit, delta], name=name).output(0)
  return range_


def range(start, limit=None, delta=1, name=None):
  if limit is None:
    limit = start
    start = 0
  return _range(start, limit, delta, name=name)


@_tensorize_input(all_posargs=True)
def _stack(*tensors, axis=0, name=None):
  return Pack(input_list=tensors, axis=axis, name=name).output(0)


def stack(tensors, axis=0, name=None):
  assert isinstance(tensors, (tuple, list))
  return _stack(*tensors, axis=axis, name=name)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def unstack(tensor, num=None, axis=0, name=None):
  if num is None:
    if tensor.shape.level == 0 or tensor.shape[axis] is None:
      raise ValueError(f"Cannot infer `num` from shape {tensor.shape}")
    num = tensor.shape[axis]

  op = Unpack(input_list=[tensor], num=num, axis=axis, name=name)
  return [op.output(i) for i in np.arange(num)]


@_tensorize_input(argpositions=(0, 1), argkeys=("inputs", "multiples"))
def tile(inputs, multiples, name=None):
  return Tile(input_list=[inputs, multiples], name=name).output(0)


@_tensorize_input(
    argpositions=(0, 1, 2, 3),
    argkeys=("inputs", "begin", "end", "strides"),
)
def strided_slice(inputs, begin, end, strides, name=None):
  return StridedSlice(
      input_list=[inputs, begin, end, strides],
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0, 1, 2), argkeys=("inputs", "begin", "size"))
def slice(inputs, begin, size, name=None):
  return Slice(input_list=[inputs, begin, size], name=name).output(0)


def concat(values, axis, name=None):
  assert isinstance(values, (list, tuple))
  inputs = [axis] + list(values)
  return _concat(*inputs, name=name)


@_tensorize_input(all_posargs=True)
def _concat(*inputs, name=None):
  return Concat(input_list=inputs, name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "paddings"))
def pad(tensor, paddings, constant_values=0, name=None):
  return Pad(
      input_list=[tensor, paddings],
      constant_values=constant_values,
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def expand_dims(tensor, axis, name=None):
  return ExpandDims(input_list=[tensor, axis], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def squeeze(tensor, axis=[], name=None):
  return Squeeze(input_list=[tensor], axis=axis, name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("dims", "value"))
def fill(dims, value, name=None):
  return Fill(input_list=[dims, value], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def add(x, y, name=None):
  return Add(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def subtract(x, y, name=None):
  return Sub(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def multiply(x, y, name=None):
  return Mul(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def divide(x, y, name=None):
  return ReadDiv(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("x",))
def negative(x, name=None):
  return Neg(input_list=[x], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def floordiv(x, y, name=None):
  return FloorDiv(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def floormod(x, y, name=None):
  return FloorMod(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def maximum(x, y, name=None):
  return Maximum(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def minimum(x, y, name=None):
  return minimum(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def divide_no_nan(x, y, name=None):
  return DivNoNan(input_list=[x, y], name=name).output(0)


def add_n(inputs, name=None):
  assert isinstance(inputs, (list, tuple))
  return _add_n(*inputs)


@_tensorize_input(all_posargs=True)
def _add_n(*inputs, name=None):
  return AddN(input_list=inputs, name=name).output(0)


def reduce_mean(tensor, axis=None, keepdims=False, name=None):
  if axis is None:
    if tensor.shape.level > 0:
      axis = np.arange(0, tensor.shape.ndims).tolist()
    else:
      zero_scalar = Const(value=np.asarray(0, dtype="int32")).output(0)
      one_scalar = Const(value=np.asarray(1, dtype="int32")).output(0)
      rank = tensor.op.get_rank_tensor(tensor_index=tensor.tensor_index)
      axis = Range(input_list=[zero_scalar, rank, one_scalar]).output(0)
  return _reduce_mean(tensor, axis, keepdims)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def _reduce_mean(tensor, axis, keepdims=False, name=None):
  return Mean(input_list=[tensor, axis], keepdims=keepdims, name=name).output(0)


def reduce_sum(tensor, axis=None, keepdims=False, name=None):
  if axis is None:
    if tensor.shape.level > 0:
      axis = np.arange(0, tensor.shape.ndims).tolist()
    else:
      zero_scalar = Const(value=np.asarray(0, dtype="int32")).output(0)
      one_scalar = Const(value=np.asarray(1, dtype="int32")).output(0)
      rank = tensor.op.get_rank_tensor(tensor_index=tensor.tensor_index)
      axis = Range(input_list=[zero_scalar, rank, one_scalar]).output(0)
  return _reduce_sum(tensor, axis, keepdims)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def _reduce_sum(tensor, axis, keepdims=False, name=None):
  return Sum(input_list=[tensor, axis], keepdims=keepdims, name=name).output(0)


def reduce_prod(tensor, axis=None, keepdims=False, name=None):
  if axis is None:
    if tensor.shape.level > 0:
      axis = np.arange(0, tensor.shape.ndims).tolist()
    else:
      zero_scalar = Const(value=np.asarray(0)).output(0)
      one_scalar = Const(value=np.asarray(1)).output(0)
      rank = tensor.op.get_rank_tensor(tensor_index=tensor.tensor_index)
      axis = Range(input_list=[zero_scalar, rank, one_scalar]).output(0)
  return _reduce_prod(tensor, axis, keepdims)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def _reduce_prod(tensor, axis, keepdims=False, name=None):
  return Prod(input_list=[tensor, axis], keepdims=keepdims, name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
  if x.shape.level > 0 and x.shape.ndims == 2 and y.shape.level > 0 and y.shape.ndims == 2:
    return MatMul(
        input_list=[x, y],
        transpose_x=transpose_x,
        transpose_y=transpose_y,
    ).output(0)
  else:
    return BatchMatMul(
        input_list=[x, y],
        transpose_x=transpose_x,
        transpose_y=transpose_y,
    ).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def squared_difference(x, y, name=None):
  return SquaredDifference(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def square(tensor, name=None):
  return Square(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def greater_equal(x, y, name=None):
  return GreaterEqual(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def greater(x, y, name=None):
  return Greater(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def less_equal(x, y, name=None):
  return LessEqual(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def less(x, y, name=None):
  return Less(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def equal(x, y, name=None):
  return Equal(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("x", "y"))
def not_equal(x, y, name=None):
  return NotEqual(input_list=[x, y], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def cumsum(tensor, axis=0, exclusive=False, reverse=False, name=None):
  return Cumsum(
      input_list=[tensor, axis],
      exclusive=exclusive,
      reverse=reverse,
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "axis"))
def cumprod(tensor, axis=0, exclusive=False, reverse=False, name=None):
  return Cumprod(
      input_list=[tensor, axis],
      exclusive=exclusive,
      reverse=reverse,
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def exp(tensor, name=None):
  return Exp(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def log1p(tensor, name=None):
  return Log1p(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def log(tensor, name=None):
  return Log(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def reciprocal(tensor, name=None):
  return Reciprocal(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def rsqrt(tensor, name=None):
  return Rsqrt(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def sqrt(tensor, name=None):
  return Sqrt(input_list=[tensor], name=name).output(0)


def dynamic_stitch(indices, data, name=None):
  assert isinstance(indices, list)
  assert isinstance(data, list)
  inputs = indices + data
  return _dyanmic_stitch(*inputs, name=name)


@_tensorize_input(all_posargs=True)
def _dynamic_stitch(*inputs, name=None):
  return DynamicStich(input_list=inputs, name=name).output(0)


@_tensorize_input(argpositions=(0, 1, 2), argkeys=("params", "indices", "axis"))
def gather(params, indices, axis=0, name=None):
  return Gather(input_list=[params, indices, axis], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("tensor", "target_shape"))
def broadcast_to(tensor, target_shape, name=None):
  return BroadcastTo(input_list=[tensor, target_shape], name=name).output(0)


@_tensorize_input(argpositions=(0, 1, 2), argkeys=("cond", "x", "y"))
def where(cond, x, y, name=None):
  return Select(input_list=[cond, x, y], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def stop_gradient(tensor, name=None):
  return StopGradient(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0, 1), argkeys=("inputs", "filters"))
def conv2d(inputs, filters, strides, padding, name=None):
  return Conv2D(
      input_list=[inputs, filters],
      strides=strides,
      padding=padding,
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0, 1, 2), argkeys=("inputs", "filters"))
def conv2d_transpose(
    inputs,
    filters,
    output_shape,
    strides,
    padding,
    name=None,
):
  return Conv2DBackpropInput(
      input_list=[filters, inputs, output_shape],
      strides=strides,
      padding=padding,
      name=name,
  ).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("inputs",))
def max_pool2d(inputs, filters_size, strides, padding, name=None):
  return MaxPool2D(
      input_list=[inputs],
      strides=strides,
      filters_size=filters_size,
      padding=padding,
  ).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("inputs",))
def avg_pool2d(inputs, filters_size, strides, padding, name=None):
  return AvgPool2D(
      input_list=[inputs],
      strides=strides,
      filters_size=filters_size,
      padding=padding,
  ).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def sigmoid(tensor, name=None):
  return Sigmoid(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def tanh(tensor, name=None):
  return Tanh(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def relu(tensor, name=None):
  return Relu(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def leaky_relu(tensor, alpha=0.2, name=None):
  return LeakyRelu(alpha=alpha, input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def log_softmax(tensor, name=None):
  return LogSoftmax(input_list=[tensor], name=name).output(0)


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def softmax(tensor, name=None):
  return Softmax(input_list=[tensor], name=name).output(0)


def random_uniform(shape, minval=0.0, maxval=1.0, name=None):
  return _random_uniform(shape, minval, maxval, name)


@_tensorize_input(
    argpositions=(
        0,
        1,
        2,
    ),
    argkeys=(
        "shape",
        "minval",
        "maxval",
    ),
)
def _random_uniform(shape, minval=0.0, maxval=1.0, name=None):
  ru = RandomUniform(input_list=[shape], name=name).output(0)
  sub = Sub(input_list=[maxval, minval]).output(0)
  mul = Mul(input_list=[sub, ru]).output(0)
  add = Add(input_list=[mul, minval]).output(0)
  sg = StopGradient(input_list=[add]).output(0)
  return sg


def random_normal(shape, mean=0.0, stddev=1.0, name=None):
  return _random_normal(shape, mean, stddev, name)


@_tensorize_input(
    argpositions=(
        0,
        1,
        2,
    ),
    argkeys=(
        "shape",
        "mean",
        "stddev",
    ),
)
def _random_normal(shape, mean, stddev, name=None):
  rsn = RandomStandardNormal(input_list=[shape], name=name).output(0)
  mul = Mul(input_list=[stddev, rsn]).output(0)
  add = Add(input_list=[mul, mean]).output(0)
  sg = StopGradient(input_list=[add]).output(0)
  return sg


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def dropout(tensor, rate, name=None):
  assert isinstance(rate, _NUMERIC_TYPES)

  if tensor.shape.level == 2:
    shape = Const(
        value=np.asarray(tensor.shape.raw_shape).astype("int32"),
    ).output(0)
  else:
    shape = tensor.get_shape_op(tensor_index=tensor.tensor_index)

  const = Const(value=np.asarray(rate).astype("float32")).output(0)
  ru = RandomUniform(input_list=[shape]).output(0)
  ge = GreaterEqual(input_list=[ru, const]).output(0)

  scalar = Const(value=np.asarray(1 / (1 - rate))).output(0)

  zero_scalar = Const(value=np.asarray(0).astype("float32")).output(0)

  mul = Mul(input_list=[tensor, scalar]).output(0)

  select = Select(input_list=[ge, mul, zero_scalar]).output(0)
  return select


@_tensorize_input(
    argpositions=(0, 1, 2, 3, 4, 5),
    argkeys=(
        "tensor",
        "mean",
        "variance",
        "offset",
        "scale",
        "variance_epsilon",
    ),
)
def batch_normalization(
    tensor,
    mean,
    variance,
    offset,
    scale,
    variance_epsilon=0.0001,
    name=None,
):

  add = Add(input_list=[variance, variance_epsilon]).output(0)

  rsqrt = Rsqrt(input_list=[add]).output(0)
  mul = Mul(input_list=[rsqrt, scale]).output(0)
  mul1 = Mul(input_list=[mul, tensor]).output(0)
  mul2 = Mul(input_list=[mul, mean]).output(0)
  sub = Sub(input_list=[offset, mul2]).output(0)

  bn = Add(input_list=[mul1, sub]).output(0)
  return bn


@_tensorize_input(argpositions=(0,), argkeys=("tensor",))
def moments(tensor, axes, keepdims=False):
  c = Const(value=np.asarray(axes, dtype="int32")).output(0)
  m = Mean(input_list=[tensor, c], keepdims=True).output(0)
  sd = SquaredDifference(input_list=[tensor, m]).output(0)
  m1 = Mean(input_list=[sd, c], keepdims=True).output(0)
  if keepdims:
    mean = m
    variance = m1
  else:
    mean = Squeeze(input_list=[m], axis=axes).output(0)
    variance = Squeeze(input_list=[m1], axis=axes).output(0)
  return mean, variance


@_tensorize_input(argpositions=(0, 1), argkeys=("labels", "logits"))
def softmax_cross_entropy_with_logits(labels, logits, name=None):

  zero_scalar = Const(value=np.asarray(0, dtype="int32")).output(0)
  one_scalar = Const(value=np.asarray(1, dtype="int32")).output(0)
  one_array = Const(value=np.asarray([1], dtype="int32")).output(0)
  zero_array = Const(value=np.asarray([0], dtype="int32")).output(0)
  minus_one_array = Const(value=np.asarray([-1], dtype="int32")).output(0)

  def _flat_tensor(tensor):
    if tensor.shape.level == 0:
      tensor_rank = Const(
          value=np.asarray(tensor.shape.ndims, dtype="int32"),
      ).output(0)
    else:
      tensor_rank = tensor.op.get_rank_tensor(tensor.tensor_index)

    sub = Sub(input_list=[tensor_rank, one_scalar], name=name).output(0)
    pack = Pack(input_list=[sub], axis=0).output(0)
    tensor_shape = tensor.op.get_shape_tensor(tensor.tensor_index)
    slice0 = Slice(input_list=[tensor_shape, pack, one_array]).output(0)
    concat = Concat(input_list=[zero_scalar, minus_one_array, slice0]).output(0)
    reshape = Reshape(input_list=[tensor, concat]).output(0)
    return reshape, tensor_shape, pack

  reshaped_logits, logits_shape, pack = _flat_tensor(logits)
  reshaped_labels, _, _ = _flat_tensor(labels)

  ce = SoftmaxCrossEntropyWithLogits(
      input_list=[reshaped_logits, reshaped_labels],
      name=name,
  ).output(0)
  slice2 = Slice(input_list=[logits_shape, zero_array, pack]).output(0)
  loss = Reshape(input_list=[ce, slice2]).output(0)

  return loss


@_tensorize_input(argpositions=(0, 1), argkeys=("labels", "logits"))
def sigmoid_cross_entropy_with_logits(labels, logits, name=None):
  mul = Mul(input_list=[logits, labels]).output(0)
  neg = Neg(input_list=[logits]).output(0)

  zeros = logits.op.get_zeros_tensor()
  ge = GreaterEqual(input_list=[logits, zeros]).output(0)

  select = Select(input_list=[ge, logits, zeros]).output(0)
  select1 = Select(input_list=[ge, neg, logits]).output(0)

  exp = Exp(input_list=[select1]).output(0)
  log1p = Log1p(input_list=[exp]).output(0)
  sub = Sub(input_list=[select, mul]).output(0)
  add = Add(input_list=[sub, log1p]).output(0)

  return add


def placeholder(shape):
  return Placeholder(shape=shape).output(0)
