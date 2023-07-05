import numpy as np


class Tensor(object):
  """Immutable tensor
  """
  def __init__(self, op, tensor_index, shape):
    from operation import Operation
    if not isinstance(op, Operation):
      raise ValueError(f"op must be an Operation, got {type(op)}")

    self._op = op
    self._tensor_index = tensor_index
    self._shape = shape

  @property
  def op(self):
    return self._op

  @property
  def tensor_index(self):
    return self._tensor_index

  @property
  def shape(self):
    return self._shape

  @property
  def name(self):
    return f"{self.op.name}:{self.tensor_index}"

  #@property
  #def id(self):
  #  return self.op.id, self.tensor_index

  def __repr__(self):
    repstr = f"<Tensor '{self.op.name}:{self.tensor_index}', shape={self.shape.raw_shape}>"
    return repstr

  def _convert_arithmetic_operand(self, other):
    from generic_ops import Const
    if not isinstance(other, Tensor):
      try:
        other = Const(value=np.asarray(other)).output(0)
      except Exception:
        raise TypeError(
            'other must be a Tensor or convertable to numpy array.')
    return other

  def __add__(self, other):
    from math_ops import Add
    other = self._convert_arithmetic_operand(other)
    return Add(input_list=[self, other]).output(0)

  def __radd__(self, other):
    from math_ops import Add
    other = self._convert_arithmetic_operand(other)
    return Add(input_list=[self, other]).output(0)

  def __mul__(self, other):
    from math_ops import Mul
    other = self._convert_arithmetic_operand(other)
    return Mul(input_list=[self, other]).output(0)

  def __rmul__(self, other):
    from math_ops import Mul
    other = self._convert_arithmetic_operand(other)
    return Mul(input_list=[self, other]).output(0)

  def __sub__(self, other):
    from math_ops import Sub
    other = self._convert_arithmetic_operand(other)
    return Sub(input_list=[self, other]).output(0)

  def __rsub__(self, other):
    from math_ops import Sub
    other = self._convert_arithmetic_operand(other)
    return Sub(input_list=[other, self]).output(0)

  def __div__(self, other):
    from math_ops import RealDiv
    other = self._convert_arithmetic_operand(other)
    return RealDiv(input_list=[self, other]).output(0)

  def __rdiv__(self, other):
    from math_ops import RealDiv
    other = self._convert_arithmetic_operand(other)
    return RealDiv(input_list=[other, self]).output(0)

  def __pos__(self):
    return self

  def __neg__(self):
    from math_ops import Neg
    return Neg(input_list=[self]).output(0)

  def __getitem__(self, slice_specs):
    from generic_ops import Const
    from array_ops import Reshape, StridedSlice, Unpack

    ndims = None
    orig_shape = None
    if self.shape.level > 0:
      ndims = self.shape.ndims
      orig_shape = list(self.shape.raw_shape)
      if self.shape.level == 1:
        shape = self.op.get_shape_tensor(tensor_index=self.tensor_index)
        unpack_op = Unpack(input_list=[shape], num=ndims, axis=0)
        for i in np.arange(ndims):
          s = unpack_op.output(i)
          if orig_shape[i] is None:
            orig_shape[i] = s
  
    new_shape = []

    if not isinstance(slice_specs, tuple):
      slice_specs = (slice_specs,)

    begin = []
    end = []
    strides = []
    ellipsis_index = None
    new_axis_count = 0

    for index, ss in enumerate(slice_specs):
      if isinstance(ss, int):
        begin.append(ss)
        end.append(ss + 1)
        strides.append(1)
        new_shape.append(orig_shape[index - new_axis_count])
  
      elif isinstance(ss, slice):
        begin.append(0 if ss.start is None else ss.start)
        end.append(orig_shape[index - new_axis_count] if ss.stop is None else ss.stop) 
        strides.append(1 if ss.step is None else ss.step) 
        new_shape.append(orig_shape[index - new_axis_count])

      elif ss is None:
        begin.append(0)
        end.append(1)
        strides.append(1)
        new_shape.append(1) 
        new_axis_count += 1
      elif ss is Ellipsis:
        raise NotImplementedError("Ellipsis is currently not supported for slicing.")

    pad_size = len(orig_shape) - (index + 1 - new_axis_count)
    begin = begin + [0] * pad_size
    end = end + orig_shape[index + 1 - new_axis_count: len(orig_shape)]
    strides = strides + [1] * pad_size
    new_shape = new_shape + orig_shape[index + 1 - new_axis_count: len(orig_shape)]

    new_shape = _build_vector_from_mixed(new_shape)

    tensor = Reshape(input_list=[self, new_shape]).output(0)

    begin = Const(value=np.asarray(begin)).output(0)
    end = _build_vector_from_mixed(end) 
    strides = Const(value=np.asarray(strides)).output(0)

    tensor = StridedSlice(input_list=[tensor, begin, end, strides]).output(0)
    return tensor


def _build_vector_from_mixed(mixed):
  from generic_ops import Const
  from array_ops import Pack

  tensorided = [i if isinstance(i, Tensor) else Const(value=np.asarray(i)).output(0) for i in mixed]
  vector = Pack(input_list=tensorided, axis=0).output(0)
  return vector

