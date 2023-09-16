"""Defines TensorShape class."""
import numpy as np


class TensorShape(object):
  """`TensorShape` is the wrapper of a tuple of positive integers (or Nones)
  that specify the shape of a symbolic Tensor at graph construction time.

  Integers indicate concrete size of a dimension, while None means unknown size.

  The tuple can also be None, which means the rank (number of dimensions) of the
  tensor is not known.

  The "level" of a `TensorShape` indicates how much we know about the shape of
  the `Tensor`. See method `level`.
  """

  def __init__(self, raw_shape):
    """Constructor.

    Args:
      raw_shape (List[int]): a list of positive integers or Nones, the shape of
        a `Tensor`.
    """
    # rank is not known
    if raw_shape is None:
      self._raw_shape = None
      self._ndims = None
    # rank is known
    else:
      self._raw_shape = tuple(raw_shape)
      self._ndims = len(raw_shape)

  @property
  def ndims(self):
    """number of dimensions (rank)."""
    return self._ndims

  @property
  def raw_shape(self):
    """"""
    return self._raw_shape

  @property
  def level(self):
    """Level of the TensorShape:

    0: unknown rank (num of dimensions)
    1: rank is known (but size of some dimension is unknown)
    2: full shape is known
    """
    if self.ndims is None:
      return 0
    elif all([s is not None for s in self.raw_shape]):
      return 2
    else:
      return 1

  def _partial_size(self):
    """Compute the partial size.

    The partial size is total number of elements from the dimensions whose size
    is known.
    """
    if self.level == 0:
      return -1
    else:
      sizes = [s for s in self.raw_shape if s is not None]
      if len(sizes):
        return np.prod(sizes).astype("int32").item()
      else:
        return -1

  def __repr__(self):
    if self._raw_shape is None:
      return "TensorShape(None)"
    else:
      return 'TensorShape([%s])' % ', '.join(map(str, self._raw_shape))

  def _compatible_with(self, tensor_shape):
    """Checks if the `TensorShape` is compatible with another shape. Example:

    `Shape(None, 1, 2, 3)` is compatible with `Shape(1, 1, 2, 3)`, while not
    compatible with `Shape(1, 2, 2, 3)`.

    Args:
      tensor_shape: raw shape, i.e. a list (or tuple) of integers (or None),
        or a `TensorShape` instance.
    """
    if self.ndims is None or tensor_shape.ndims is None:
      return True

    if self.ndims != tensor_shape.ndims:
      return False

    return all([
        d1 is None or d2 is None or d1 == d2
        for d1, d2 in zip(self.raw_shape, tensor_shape.raw_shape)
    ])

  def _broadcastable_with(self, tensor_shape):
    """Returns whether this `TensorShape` is broadcastable with `tensor_shape`.

    Two shapes are broadcastable if for all pairs of corresponding of sizes
    starting from the lowest dimension
    * they are equal
    * one of them is 1
    * one of them is None (unknown)
    """
    if (
        self.ndims is None or tensor_shape.ndims is None or self.ndims == 0 or
          tensor_shape.ndims == 0
    ):
      return True

    return all([
        d1 is None or d2 is None or d1 == 1 or d2 == 1 or d1 == d2
        for d1, d2 in zip(self.raw_shape[::-1], tensor_shape.raw_shape[::-1])
    ])

  def _merge(self, tensor_shape, skip=[]):
    """Replace unknown dimension sizes in this `TensorShape` with concrete
    sizes in `tensor_shape`.

    Args:
      tensor_shape (TensorShape): the other `TensorShape`.
      skip (List[int]): list of dimension indices to skip.
    """

    if self._compatible_with(tensor_shape):
      if self.ndims is not None and tensor_shape.ndims is not None:
        raw_shape = list(self._raw_shape)
        for i, s in enumerate(raw_shape):
          if i in skip:
            continue
          if raw_shape[i] is None and tensor_shape.raw_shape[i] is not None:
            raw_shape[i] = tensor_shape._raw_shape[i]
        self._raw_shape = tuple(raw_shape)
    else:
      raise ValueError(
          f"Attempting to merge incompatible shapes: {self}, {tensor_shape}",
      )

  def _diff_at(self, tensor_shape):
    """Returns the indices of dimensions at which this `TensorShape` differs
    with `tensor_shape`.
    """
    axes = []
    assert self._compatible_with(tensor_shape)
    if self.ndims is not None and tensor_shape.ndims is not None:
      for i, (d1, d2) in enumerate(zip(self, tensor_shape)):
        if d1 is not None and d2 is not None and d1 != d2:
          axes.append(i)
    return axes

  def __getitem__(self, k):
    """Allows for indexing and slicing. Example:

    `Shape(1, 2, None)[1] == 2`, and
    `Shape(1, 2, None)[1:]` == TensorShape([2, None])`.
    """
    assert self.ndims is not None
    if isinstance(k, slice):
      return TensorShape(self.raw_shape[k])
    else:
      return self._raw_shape[k]
