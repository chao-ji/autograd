import wrappers


class Tensor(object):
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

  def __repr__(self):
    repstr = f"<Tensor '{self.op.name}:{self.tensor_index}', shape={self.shape.raw_shape}>"
    return repstr

  def _convert_arithmetic_operand(self, other):

    if not isinstance(other, Tensor):
      try:
        other = wrappers.constant(other)
      except Exception:
        raise TypeError(
            'other must be a Tensor or convertable to numpy array.')
    return other

  def __add__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.add(self, other)

  def __radd__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.add(other, self)

  def __mul__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.multiply(self, other)

  def __rmul__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.multiply(other, self)

  def __sub__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.subtract(self, other)

  def __rsub__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.subtract(other, self)

  def __div__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.divide(self, other)

  def __rdiv__(self, other):
    other = self._convert_arithmetic_operand(other)
    return wrappers.divide(other, self)

  def __pos__(self):
    return self

  def __neg__(self):
    return wrappers.negative(self)

  def __getitem__(self, slice_specs):

    ndims = None
    orig_shape = None
    if self.shape.level > 0:
      ndims = self.shape.ndims
      orig_shape = list(self.shape.raw_shape)
      if self.shape.level == 1:
        shape = self.op.get_shape_tensor(tensor_index=self.tensor_index)
        for i, s in enumerate(wrappers.unstack(shape, axis=0, num=ndims)):
          if orig_shape[i] is None:
            orig_shape[i] = s
    new_shape = []

    if not isinstance(slice_specs, tuple):
      slice_specs = (slice_specs,)

    # Ellipsis
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
      else:
        assert ss is Ellipsis

    

    pad_size = len(orig_shape) - (index + 1 - new_axis_count)
    begin = begin + [0] * pad_size
    end = end + orig_shape[index + 1 - new_axis_count: len(orig_shape)]
    strides = strides + [1] * pad_size
    new_shape = new_shape + orig_shape[index + 1 - new_axis_count: len(orig_shape)]

    print("begin")
    print(begin)
    print("end")
    print(end)
    print("strides")
    print(strides)

    print("orig_shape")
    print(orig_shape)
    print("new_shape")
    print(new_shape)
    print("index")
    print(index)
    print("new_axis_count")
    print(new_axis_count)  
    tensor = wrappers.reshape(self, new_shape)
    tensor = wrappers.strided_slice(tensor, begin, end, strides)

    return tensor

    """
    c = len(orig_shape) + new_axis_count - index

    begin = begin + [0] * c #(index + 1 - new_axis_count)
    end = end + orig_shape[-c:] #orig_shape[-(index + 1 - new_axis_count):]
    strides = strides + [1] * c#(index + 1 - new_axis_count)
    new_shape = new_shape + orig_shape[-c:] #orig_shape[-(index + 1 - new_axis_count):]

    print("begin")
    print(begin)
    print("end")
    print(end)
    print("strides")
    print(strides)

    print("orig_shape")
    print(orig_shape)
    print("new_shape")
    print(new_shape)

    print("index")
    print(index)
    print("new_axis_count")
    print(new_axis_count)
    """
