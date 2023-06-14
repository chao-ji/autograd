"""Neural network related Operations."""
import numpy as np

from operation import Operation
from origin_ops import Const
from tensor import Tensor


class _Filters2DBase(Operation):
  """Base class for 2-D filters-based Operations (Conv2D, MaxPool2D, AvgPool2D,
  etc.).
  """

  def __init__(self, strides, padding, input_list, graph=None, name=None):
    """Constructor.

    Args:
      strides (tuple): strides in height and width dimension.
      padding (string): padding scheme (either "SAME" or "VALID").
    """
    super(_Filters2DBase, self).__init__(
        graph=graph, input_list=input_list, name=name)
    self._strides = strides
    self._padding = padding

  def _get_shapes(self, inputs_shape, filters_size):
    """Compute the spatial dimensions of the outputs tensor, and the padding
    sizes according to the padding scheme.

    Padding sizes are computed according to

    https://www.tensorflow.org/versions/r1.3/api_guides/python/nn#Convolution

    Args:
      inputs_shape (tuple): 4-tuple storing shape of the input tensor in [
        batch_size, height, width, in_channels].
      filters_size (tuple): filters size in height and width dimension.

    Returns:
      out_size (tuple): 2-tuple storing height and width of the outputs tensor.
      padding (tuple): 4-tuple storing the padding sizes in height dimension (
        "pad_top" and "pad_bottom") and width dimension ("pad_left" and
        "pad_right").
    """
    strides_height, strides_width = self._strides
    filters_height, filters_width = filters_size
    height, width = inputs_shape[1:3]

    if self._padding == "SAME":
      out_height = int(np.ceil(float(height) / strides_height))
      out_width = int(np.ceil(float(width) / strides_width))

      pad_height = (max(filters_height - strides_height, 0)
          if height % strides_height == 0
          else max(filters_height - height % strides_height, 0))
      pad_width = (max(filters_width - strides_width, 0)
          if width % strides_width== 0
          else max(filters_width - width % strides_width, 0))
      padding = (pad_height // 2,
                 pad_height - pad_height // 2,
                 pad_width // 2,
                 pad_width - pad_width // 2)
    elif self._padding == "VALID":
      out_height = int(
            np.ceil(float(height - filters_height + 1) / strides_height))
      out_width = int(
          np.ceil(float(width - filters_width + 1) / strides_width))
      padding = 0, 0, 0, 0
    else:
      raise ValueError(f"Invalid padding scheme: {padding}")
    out_size = out_height, out_width

    return out_size, padding

  def _get_img_col_index(self, pad_size, filters_size):
    """Compute the height and width coordinates of the upper left pixel of all
    image patches that match the size of the filter. Example:

    Given the 4-by-4 image below,
    0,0  0,1  0,2  0,3
    1,0  1,1  1,2  1,3
    2,0  2,1  2,2  2,3
    3,0  3,1  3,2  3,3

    and a 2-by-2 filter with strides [2, 2], the output is like this:
    
    [(h, w, h_index w_index)] =
      [(0,  0,  0,  0),
       (0,  2,  0,  1),
       (2,  0,  1,  0),
       (2,  2,  1,  1)]

    Args:
      pad_size (tuple): height and width of the padded inputs tensor.
      filters_size (tuple): filters size in height and width dimension.

    Returns:
      img_col_index (ndarray): numpy array of shape [out_height * out_width, 4],
        where each row holds a tuple of 4 integers [h, w, h_index, w_index]. `h`
        and `w` correspond to the height and width coordinates of the upper left
        pixel of each patch that match the size of the filter; `h_index` and
        `w_index` correspond to the height and width indices of each path.
    """
    pad_height, pad_width = pad_size
    strides_height, strides_width = self._strides
    filters_height, filters_width = filters_size
    h_col_indices = np.arange(
        0, pad_height - filters_height + 1, strides_height)
    w_col_indices = np.arange(
        0, pad_width - filters_width + 1, strides_width)
    w_grid, h_grid = np.meshgrid(w_col_indices, h_col_indices)
    w_index_grid, h_index_grid = np.meshgrid(
        np.arange(w_col_indices.shape[0]), np.arange(h_col_indices.shape[0]))
    img_col_index = np.vstack([h_grid.ravel(),
                               w_grid.ravel(),
                               h_index_grid.ravel(),
                               w_index_grid.ravel()]).T
    return img_col_index

  def _pad(self, inputs, padding):
    """Pad the inputs tensor according to padding sizes.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be padded.
      padding (tuple): 4-tuple storing the padding sizes in height dimension (
        "pad_top" and "pad_bottom") and width dimension ("pad_left" and
        "pad_right").

    Returns:
      inputs_pad (tensor): 4D tensor of shape [batch_size, pad_height, pad_width
        , in_channels], the padded inputs tensor.
    """
    pad_value = self._pad_value if hasattr(self, "_pad_value") else 0
    inputs_pad = np.pad(
        inputs,
        [[0, 0], padding[:2], padding[2:], [0, 0]],
        mode="constant",
        constant_values=pad_value
    )
    return inputs_pad

  def _matrixize_inputs_tensor(self, inputs, padding, filters_size):
    """Transform 4D inputs tensor to a 2D matrix layout such that it can be
    dot-producted with the matrixized filters tensor.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be padded.
      padding (tuple): 4-tuple storing the padding sizes in height dimension (
        "pad_top" and "pad_bottom") and width dimension ("pad_left" and
        "pad_right").
      filters_size (tuple): filters size in height and width dimension.
  
    Returns:
      inputs_mat (tensor): 2D tensor of shape [out_height * out_width *
        batch_size, in_channels * filters_height * filters_ width], the inputs
        tensor in 2D matrix format.
    """
    batch_size = inputs.shape[0]
    filters_height, filters_width = filters_size

    # [out_height * out_width, 4]
    img_col_index = self._get_img_col_index(
        (inputs.shape[1] + padding[0] + padding[1],
         inputs.shape[2] + padding[2] + padding[3]),
        (filters_height, filters_width)
    )

    inputs = self._pad(inputs, padding)
    def _func(indices):
      """Slice a patch from the 4D inputs tensor in dim1 (height) and dim2
      (width) of the size of the filters tensor.

      Returns tensor of shape [batch_size, in_channels * filters_height *
      filters_width].
      """
      h, w = indices
      return inputs[:, h:h+filters_height, w:w+filters_width, :
          ].transpose(0, 3, 1, 2).reshape((batch_size, -1))
    inputs_mat = np.vstack(np.apply_along_axis(_func, 1, img_col_index[:, :2]))

    return inputs_mat

  def _flat_channels_dim(self, inputs_mat, out_size, in_channels, filters_size):
    """Reshape the inputs tensor in 2D matrix format by absorbing the
    "in_channels" dim from dim-1 into dim-0.

    Args:
      inputs_mat (tensor): 2D tensor of shape [out_height * out_width *
        batch_size, in_channels * filters_height * filters_width], the inputs
        tensor in 2D matrix format.
      out_size (tuple): height and width of the outputs tensor.
      in_channels (int): num of the input channels.
      filters_size (tuple): filters size in height and width dimension.

    Returns:
      inputs_mat (tensor): 2D tensor of shape [out_height * out_width *
        batch_size * in_channels, filters_height * filters_ width], the reshaped
        inputs tensor in 2D matrix format.
    """
    out_height, out_width = out_size
    filters_height, filters_width = filters_size
    inputs_mat = inputs_mat.reshape((
        out_height, out_width, -1, in_channels, filters_height, filters_width
    )).reshape(-1, filters_height * filters_width)
    return inputs_mat

  def _matrixize_filters_tensor(self, filters):
    """Reshape the 4D filters tensor to 2D matrix format.

    Args:
      filters (tensor): 4D tensor of shape [filters_height, filters_width,
        in_channels, out_channels], the filters tensor.

    Returns:
      filters_mat (tensor): 2D tensor of shape [in_channels * filters_height *
        filters_width, out_channels], the filters tensor in 2D matrix format.
    """
    out_channels = filters.shape[3]
    filters_mat = filters.transpose(2, 0, 1, 3).reshape(-1, out_channels)
    return filters_mat

  def _unpad(self, inputs, padding):
    """Strip padding from the padded inputs tensor.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, pad_height, pad_width,
        in_channels], the padded inputs tensor.
      padding (tuple): 4-tuple storing the padding sizes in height dimension (
        "pad_top" and "pad_bottom") and width dimension ("pad_left" and
        "pad_right").

    Returns:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the unpadded inputs tensor.
    """
    slice_height = slice(
        padding[0], -padding[1]) if padding[1] > 0 else slice(padding[0], None
    )
    slice_width = slice(
        padding[2], -padding[3]) if padding[3] > 0 else slice(padding[3], None
    )
    inputs = inputs[:, slice_height, slice_width]
    return inputs

  def _tensorize_grads_matrix(
      self,
      inputs_grads_mat,
      inputs_shape,
      padding,
      out_size,
      filters_size,
    ):
    """Transform gradients w.r.t inputs tensor in 2D matrix layout to 4D tensor
    format.

    Args:
      inputs_grads_mat (tensor): 2D tensor of shape [out_height * out_width *
        batch_size, in_channels * filters_height * filters_width], gradients
        w.r.t. inputs tensor in 2D matrix format.
      inputs_shape (tuple): 4-tuple storing shape of the input tensor in [
        batch_size, height, width, in_channels].
      padding (tuple): 4-tuple storing the padding sizes in height dimension (
        "pad_top" and "pad_bottom") and width dimension ("pad_left" and
        "pad_right").
      out_size (tuple): height and width of the outputs tensor.
      filters_size (tuple): filters size in height and width dimension.

    Returns:
      inputs_grads (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], gradients w.r.t inputs tensor.
    """
    filters_height, filters_width = filters_size
    out_height, out_width = out_size
    in_channels = inputs_shape[3]
    pad_height = inputs_shape[1] + padding[0] + padding[1]
    pad_width = inputs_shape[2] + padding[2] + padding[3]

    # [out_height * out_width, 4]
    img_col_index = self._get_img_col_index(
        (pad_height, pad_width),
        (filters_height, filters_width),
    )

    # [out_height, out_width, batch_size, filters_height, filters_width,
    # in_channels]
    inputs_grads_tmp = inputs_grads_mat.reshape((
        out_height, out_width, -1, in_channels, filters_height, filters_width
        )).transpose(0, 1, 2, 4, 5, 3)

    def _func(indices):
      """Route gradients from `inputs_grads_tmp` to slices in `inputs_grads`."""
      h, w, h_index, w_index = indices
      inputs_grads = np.zeros(
        (inputs_shape[0], pad_height, pad_width, inputs_shape[3]
      ), dtype="float32")
      inputs_grads[:, h:h+filters_height, w:w+filters_width, :
          ] = inputs_grads_tmp[h_index, w_index]
      return inputs_grads

    # [out_height * out_width, batch_size, pad_height, pad_width, in_channels]
    inputs_grads = np.apply_along_axis(_func, 1, img_col_index)

    # [batch_size, pad_height, pad_width, in_channels]
    # sum the gradients over all `out_height * out_width` slices
    inputs_grads = inputs_grads.sum(axis=0)

    # [batch_size, height, width, in_channels]
    inputs_grads = self._unpad(inputs_grads, padding)
    return inputs_grads


class Conv2D(_Filters2DBase):
  """Regular 2D convolution."""

  def _run(self, inputs, filters):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be convolved with the filters tensor.
      filters (tensor): 4D tensor of shape [filters_height, filters_width,
        in_channels, out_channels], the filters tensor.

    Returns:
      outputs (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        out_channels], the result of convolution.
    """
    filters_height, filters_width = filters.shape[:2]
    (out_height, out_width), padding = self._get_shapes(
        inputs.shape, (filters_height, filters_width)
    )
    batch_size, out_channels = inputs.shape[0], filters.shape[3]

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(
        inputs, padding, (filters_height, filters_width)
    )

    #[in_channels*filters_height*filters_width, out_channels]
    filters_mat = self._matrixize_filters_tensor(filters)

    #[out_height*out_width*batch_size, out_channels]
    outputs = np.dot(inputs_mat, filters_mat).reshape(
        out_height,
        out_width,
        batch_size,
        out_channels
    ).transpose(2, 0, 1, 3)
    return outputs

  def _grad_func(self, in_grad_tensors):
    """
    Args:
      in_grad_tensors: list of (Op, tensor_index)
    """
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = Conv2DBackpropInput(
          strides=self._strides,
          padding=self._padding,
          input_list=[
              self._input_list[1],
              in_grad_tensors[0],
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      op, tensor_index = self._input_list[1].op, self._input_list[1].tensor_index
      bp_filters = Conv2DBackpropFilter(
          strides=self._strides,
          padding=self._padding,
          input_list=[
              self._input_list[0],
              in_grad_tensors[0],
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0), Tensor(bp_filters, 0)]
    return out_grad_tensors


class Conv2DBackpropInput(_Filters2DBase):
  """Backprop the gradients from the outputs of `Conv2D` to the input argument
  `inputs`.
  """

  def _run(self, filters, grads, inputs_shape):
    """Execute the Operation.

    Args:
      filters (tensor): 4D tensor of shape [filters_height, filters_width,
        in_channels, out_channels], the filters tensor.
      grads (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        out_channels], gradients w.r.t. the outputs tensor.
      inputs_shape (tuple): 4-tuple storing shape of the inputs tensor as [
        batch_size, height, width, in_channels].

    Returns:
      inputs_grads (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], gradients w.r.t. the inputs tensor.
    """
    filters_height, filters_width = filters.shape[:2]
    out_size, padding = self._get_shapes(inputs_shape, (filters_height, filters_width))
    out_channels = filters.shape[3]

    #[out_height*out_width*batch_size, out_channels]
    grads_mat = grads.transpose(1, 2, 0, 3).reshape((-1, out_channels))

    #[in_channels*filters_height*filters_width, out_channels]
    filters_mat = self._matrixize_filters_tensor(filters)

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_grads_mat = np.dot(grads_mat, filters_mat.T)

    inputs_grads = self._tensorize_grads_matrix(
      inputs_grads_mat, inputs_shape, padding, out_size, (filters_height, filters_width))
    return inputs_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_filters = Conv2DBackpropFilter(
          strides=self._strides,
          padding=self._padding,
          input_list=[
              in_grad_tensors[0],
              self._input_list[1],
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      bp_grads = Conv2D(
          strides=self._strides,
          padding=self._padding,
          input_list=[in_grad_tensors[0], self._input_list[0]]
      )
      out_grad_tensors = [Tensor(bp_filters, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0, 1]


class Conv2DBackpropFilter(_Filters2DBase):
  """Backprop the gradients from the outputs of `Conv2D` to the input argument
  `filters`.
  """

  def _run(self, inputs, grads, filters_shape):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be convolved with the filters tensor.
      grads (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        out_channels], gradients w.r.t. the outputs tensor.
      filters_shape (tuple): 4-tuple storing shape of the filters tensor as [
        filters_height, filters_width, in_channels, out_channels].

    Returns:
      filters_grads (tensor): 4D tensor of shape [filters_height, filters_width,
        in_channels, out_channels], gradients w.r.t the filters tensor.
    """
    filters_height, filters_width, in_channels, out_channels = filters_shape
    padding = self._get_shapes(inputs.shape, (filters_height, filters_width))[1]

    #[out_height*out_width*batch_size, out_channels]
    grads_mat = grads.transpose(1, 2, 0, 3).reshape((-1, out_channels))

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(
        inputs, padding, (filters_height, filters_width)
    )

    #[in_channels*filters_height*filters_width, out_channels]
    filters_grads_mat = np.dot(inputs_mat.T, grads_mat)

    filters_grads = filters_grads_mat.reshape((
        in_channels, filters_height, filters_width, out_channels
        )).transpose(1, 2, 0, 3)
    return filters_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index 
      bp_inputs = Conv2DBackpropInput(
          strides=self._strides,
          padding=self._padding,
          input_list=[
              in_grad_tensors[0],
              self._input_list[1],
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      bp_grads = Conv2D(
          strides=self._strides,
          padding=self._padding,
          input_list=[self._input_list[0], in_grad_tensors[0]]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0, 1]


class _Pooling2DBase(_Filters2DBase):
  """Base class for 2D pooling Operations (MaxPool2D, AvgPool2D, etc.)."""

  def __init__(
      self, strides, filters_size, padding, input_list, graph=None, name=None
    ):
    """Constructor.

    Set `np.nan` as flag of padded value when computing maximum or average.

    Args:
      strides (tuple): strides in height and width dimension.
      filters_size (tuple): filters size in height and width dimension.
      padding (string): padding scheme (either "SAME" or "VALID").
    """
    super(_Pooling2DBase, self).__init__(
        strides=strides,
        padding=padding,
        graph=graph,
        input_list=input_list,
        name=name
    )
    self._filters_size = filters_size
    self._pad_value = np.nan


class MaxPool2D(_Pooling2DBase):
  """Regular 2D max pooling."""

  def _run(self, inputs):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be max-pooled.

    Returns:
      outputs (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        in_channels], the outputs tensor.
    """
    out_size, padding = self._get_shapes(inputs.shape, self._filters_size)
    batch_size, in_channels = inputs.shape[0], inputs.shape[3]

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(inputs, padding, self._filters_size)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_mat = self._flat_channels_dim(inputs_mat, out_size, in_channels, self._filters_size)

    #[out_height*out_width*batch_size*in_channels]
    outputs = np.nanmax(inputs_mat, axis=1)

    outputs = outputs.reshape((
        out_size[0], out_size[1], batch_size, in_channels
    )).transpose(2, 0, 1, 3)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      maxpool2d_grads = MaxPool2DGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=[self._input_list[0], in_grad_tensors[0]]
      )
      out_grad_tensors = [Tensor(maxpool2d_grads, 0)]
    return out_grad_tensors


class MaxPool2DGrad(_Pooling2DBase):
  """Backprop the gradients from the outputs of `MaxPool2D` to the input
  argument `inputs`.
  """

  def _run(self, inputs, grads):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be max-pooled.
      grads (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        in_channels], gradients w.r.t. the outputs tensor.

    Returns:
      inputs_grads (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], gradients w.r.t. the inputs tensor.
    """
    out_size, padding = self._get_shapes(inputs.shape, self._filters_size)
    batch_size, in_channels = inputs.shape[0], inputs.shape[3]

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(inputs, padding, self._filters_size)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_mat = self._flat_channels_dim(inputs_mat, out_size, in_channels, self._filters_size)

    #[out_height*out_width*batch_size*in_channels]
    argmax = np.nanargmax(inputs_mat, axis=1)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    ind_mat = np.zeros_like(inputs_mat, dtype="float32")
    ind_mat[np.arange(ind_mat.shape[0]), argmax] = 1

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    grads_mat = np.tile(
        grads.transpose(1, 2, 0, 3).reshape((-1, 1)), (1, ind_mat.shape[1])
    )

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_grads_mat = ind_mat * grads_mat
    inputs_grads_mat = inputs_grads_mat.reshape((
        out_size[0] * out_size[1] * batch_size,
        in_channels * self._filters_size[0] * self._filters_size[1]
    ))

    # [batch_size, height, width, in_channels]
    inputs_grads = self._tensorize_grads_matrix(
      inputs_grads_mat, inputs.shape, padding, out_size, self._filters_size)
    return inputs_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[1].op, self._input_list[1].tensor_index
      bp_inputs = MaxPool2DGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list= [
              self._input_list[0],
              Tensor(op.get_zeros_op(tensor_index=tensor_index), 0)
          ]
      )
      bp_grads = MaxPool2DGradGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=[self._input_list[0], in_grad_tensors[0]]
      )

      out_grad_tensors = [Tensor(bp_inputs, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors


class MaxPool2DGradGrad(_Pooling2DBase):
  """Backprop the gradients from the outputs of `MaxPool2DGrad` to the input
  argument `grads`.
  """

  def _run(self, inputs, inputs_grads_grads):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be max-pooled. 
      inputs_grads_grads (tensor): 4D tensor of shape [batch_size, height, width
        , in_channels], gradients w.r.t. the input argument `inputs_grads` of
        `MaxPool2DGrad`.

    Returns:
      grads_grads (tensor): 4D tensor of shape [batch_size, out_height,
        out_width, in_channels], gradients w.r.t. the input argument `grads`
        of `MaxPool2DGrad`.
    """
    out_size, padding = self._get_shapes(inputs_grads_grads.shape, self._filters_size)
    batch_size, in_channels = (
        inputs_grads_grads.shape[0], inputs_grads_grads.shape[3])

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_grads_grads_mat = self._matrixize_inputs_tensor(
        inputs_grads_grads, padding, self._filters_size)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_grads_grads_mat = self._flat_channels_dim(
        inputs_grads_grads_mat, out_size, in_channels, self._filters_size)

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(inputs, padding, self._filters_size)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_mat = self._flat_channels_dim(inputs_mat, out_size, in_channels, self._filters_size)

    #[out_height*out_width*batch_size*in_channels]
    argmax = np.nanargmax(inputs_mat, axis=1)

    #[out_height*out_width*batch_size*in_channels] 
    grads_grads_mat = inputs_grads_grads_mat[
        np.arange(inputs_grads_grads_mat.shape[0]), argmax]

    grads_grads = grads_grads_mat.reshape(
        out_size[0], out_size[1], batch_size, in_channels
    ).transpose(2, 0, 1, 3)
    return grads_grads


  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = in_grad_tensors[0].op, in_grad_tensors[0].tensor_index
      bp_inputs = MaxPool2DGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=[
              self._input_list[0],
              Tensor(op.get_zeros_op(tensor_index=tensor_index), 0)
          ]
      )
      bp_inputs_grads_grads = MaxPool2DGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=[self._input_list[0], in_grad_tensors[0]]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0), Tensor(bp_inputs_grads_grads, 0)]

    return out_grad_tensors


class AvgPool2D(_Pooling2DBase):
  """Regular 2D average pooling."""

  def _run(self, inputs):
    """Execute the Operation.

    Args:
      inputs (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], the inputs tensor to be average-pooled.

    Returns:
      outputs (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        in_channels], the outputs tensor.
    """
    out_size, padding = self._get_shapes(inputs.shape, self._filters_size)
    batch_size, in_channels = inputs.shape[0], inputs.shape[3]

    #[out_height*out_width*batch_size, in_channels*filters_height*filters_width]
    inputs_mat = self._matrixize_inputs_tensor(inputs, padding, self._filters_size)

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_mat = self._flat_channels_dim(inputs_mat, out_size, in_channels, self._filters_size)

    #[out_height*out_width*batch_size*in_channels]
    outputs = np.nanmean(inputs_mat, axis=1)

    outputs = outputs.reshape((
        out_size[0], out_size[1], batch_size, in_channels
    )).transpose(2, 0, 1, 3)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_inputs = AvgPool2DGrad(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=[
              in_grad_tensors[0],
              Tensor(op.get_shape_op(tensor_index=tensor_index), 0)
          ]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class AvgPool2DGrad(_Pooling2DBase):
  """Backprop the gradients from the outputs of `AvgPool2D` to the input
  argument `inputs`."""

  def _run(self, grads, inputs_shape):
    """Execute the Operation.

    Args:
      grads (tensor): 4D tensor of shape [batch_size, out_height, out_width,
        in_channels], gradients w.r.t. the outputs tensor.
      inputs_shape (tuple): 4-tuple storing shape of the inputs tensor as [
        batch_size, height, width, in_channels]. 

    Returns:
      inputs_grads (tensor): 4D tensor of shape [batch_size, height, width,
        in_channels], gradients w.r.t. the inputs tensor.
    """
    out_size, padding = self._get_shapes(inputs_shape, self._filters_size)
    batch_size, in_channels = inputs_shape[0], inputs_shape[3]

    pad_height = inputs_shape[1] + padding[0] + padding[1]
    pad_width = inputs_shape[2] + padding[2] + padding[3]

    strides_height, strides_width = self._strides
    filters_height, filters_width = self._filters_size
    h_col_indices = np.arange(
        0, pad_height - filters_height + 1, strides_height)
    w_col_indices = np.arange(
        0, pad_width - filters_width + 1, strides_width)
    w_grid, h_grid = np.meshgrid(w_col_indices, h_col_indices)

    def _func(indices):
      """Count the non-nan entries of each patch."""
      h, w = indices
      if h < padding[0]:
        if h + filters_height <= padding[0]:
          patch_height = 0
        else:
          patch_height = min(h + filters_height, inputs_shape[1] + padding[0]
              ) - padding[0]
      else:
        patch_height = min(filters_height, inputs_shape[1] + padding[0] - h)

      if w < padding[2]:
        if w + filters_width <= padding[2]:
          patch_width = 0
        else:
          patch_width = min(w + filters_width, inputs_shape[2] + padding[2]
              ) - padding[2]
      else:
        patch_width = min(filters_width, inputs_shape[2] + padding[2] - w)
      return patch_height * patch_width

    # [out_height * out_width, 2]
    img_col_index = np.vstack([h_grid.ravel(), w_grid.ravel()]).T

    # [out_height * out_width]
    divisor = np.vstack(np.apply_along_axis(_func, 1, img_col_index))

    #[out_height*out_width, batch_size*in_channels*filters_height*filters_width]
    ind_mat = np.ones((
        out_size[0] * out_size[1],
        batch_size * in_channels * filters_height * filters_width
    ), dtype="float32") / divisor

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    ind_mat = ind_mat.reshape(
        (out_size[0] * out_size[1] * batch_size * in_channels, -1)
    )

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    grads_mat = np.tile(
        grads.transpose(1, 2, 0, 3).reshape((-1, 1)), (1, ind_mat.shape[1])
    )

    #[out_height*out_width*batch_size*in_channels, filters_height*filters_width]
    inputs_grads_mat = ind_mat * grads_mat
    inputs_grads_mat = inputs_grads_mat.reshape((
        out_size[0] * out_size[1] * batch_size,
        in_channels * filters_height * filters_width
    ))

    #[batch_size, height, width, in_channels]
    inputs_grads = self._tensorize_grads_matrix(
      inputs_grads_mat, inputs_shape, padding, out_size, self._filters_size)
    return inputs_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_grads = AvgPool2D(
          strides=self._strides,
          filters_size=self._filters_size,
          padding=self._padding,
          input_list=in_grad_tensors,
      )
      out_grad_tensors = [Tensor(bp_grads, 0)]
    return out_grad_tensors

  def _get_bp_indices(self):
    return [0]


class FusedBatchNorm(Operation):
  """"""
  
  def __init__(self, epsilon=0.001, momentum=0.999, training=False):
    self._epsilon = epsilon
    self._momentum = momentum
    self._training = training

  def run(self, inputs, offset, scale, moving_mean=None, moving_variance=None):
    """"""

    if self._training:
      mean, variance, variance_ddof1 = self._get_batch_stats(inputs) 
      #moving_mean, moving_variance = self._update_moving_stats(
      #    mean, variance_ddof1, moving_mean, moving_variance)
    else:
      mean, variance = moving_mean, moving_variance

    standard_inputs = self._get_standard_inputs(
        inputs, mean, variance)

    outputs = standard_inputs * scale + offset 
    return outputs

  def _get_batch_stats(self, inputs):
    dims = tuple(range(inputs.ndim - 1))
    mean = inputs.mean(axis=dims)
    variance = inputs.var(axis=dims)
    variance_ddof1 = inputs.var(axis=dims, ddof=1)
    return mean, variance, variance_ddof1

  def _get_standard_inputs(self, inputs, mean, variance):
    standard_inputs = (inputs - mean) / np.sqrt(variance + self._epsilon)
    return standard_inputs

  def _update_moving_stats(self, mean, variance):
    moving_variance = moving_variance * self._momentum + variance * (1 - self._momentum)
    moving_mean = moving_mean * self._momentum + mean * (1 - self._momentum)
    return moving_mean, moving_variance   


class FusedBatchNormBackpropInputs(Operation):
  def __init__(self, epsilon=0.001, momentum=0.999,): 
    self._epsilon = epsilon
    self._momentum = momentum

  def run(self, inputs, scale, grads):
    dims = tuple(range(inputs.ndim - 1))

    mean, variance = self._get_batch_stats(inputs)[:2]
    standard_inputs_grads = grads * scale

    mean_grads = standard_inputs_grads * np.power(
        variance + self._epsilon, -0.5)
    mean_grads = -mean_grads.sum(axis=dims)
    variance_grads = standard_inputs_grads * (inputs - mean) * np.power(
        variance + self._epsilon, -1.5)
    variance_grads = -0.5 * variance_grads.sum(axis=dims)

    variance_grads = variance_grads * 2 * (inputs - mean)
    inputs_grads = (standard_inputs_grads * np.power(
        variance + self._epsilon, -0.5)) + (
        mean_grads + variance_grads) / np.prod(
        np.array(inputs.shape)[np.array(dims)])
    return inputs_grads

  def _get_batch_stats(self, inputs):
    dims = tuple(range(inputs.ndim - 1))
    mean = inputs.mean(axis=dims)
    variance = inputs.var(axis=dims)
    variance_ddof1 = inputs.var(axis=dims, ddof=1)
    return mean, variance, variance_ddof1


  

class Sigmoid(Operation):
  def _run(self, inputs):
    outputs = 1 / (1 + np.exp(-inputs)) 
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = SigmoidGrad(
          input_list=[Tensor(self, 0), in_grad_tensors[0]]
      )
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class SigmoidGrad(Operation):
  def _run(self, outputs, grads):
    outputs_inputs_grads = outputs * (1 - outputs) * grads
    return outputs_inputs_grads

  def _grad_func(self, in_grad_tensors):
    from math_ops import Mul, Sub

    with self._graph.as_default_graph():
      mul = Mul(input_list=[in_grad_tensors[0], self._input_list[1]])
      mul1 = Mul(
          input_list=[
              Tensor(Const(value=np.asarray(2)), 0),
              Tensor(mul, 0)
        ]
      )
      mul2 = Mul(input_list=[self._input_list[0], Tensor(mul1, 0)]) 
      bp_outputs = Sub(input_list=[Tensor(mul, 0), Tensor(mul2, 0)])

      bp_grads = SigmoidGrad(input_list=[self._input_list[0], in_grad_tensors[0]])
    out_grad_tensors = [Tensor(bp_outputs, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors


class Tanh(Operation):
  def _run(self, inputs):
    outputs = np.tanh(inputs)
    return outputs

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = TanhGrad(input_list=[Tensor(self, 0), in_grad_tensors[0]])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class TanhGrad(Operation):
  def _run(self, outputs, grads):
    outputs_inputs_grads = (1 - outputs * outputs) * grads
    return outputs_inputs_grads    

  def _grad_func(self, in_grad_tensors):
    from math_ops import Mul

    with self._graph.as_default_graph():
      mul = Mul(
          input_list=[
              Tensor(Const(value=np.asarray(-2)), 0),
              in_grad_tensors[0]
          ]
      )
      mul1 = Mul(input_list=[Tensor(mul, 0), self._input_list[1]])
      bp_outputs = Mul(input_list=[Tensor(mul1, 0), self._input_list[0]])
      bp_grads = TanhGrad(input_list=[self._input_list[0], in_grad_tensors[0]])
      out_grad_tensors = [Tensor(bp_outputs, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors


class Relu(Operation):
  def _run(self, inputs):
    outputs = np.maximum(0, inputs)
    return outputs 

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      bp_inputs = ReluGrad(input_list=[Tensor(self, 0), in_grad_tensors[0]])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class ReluGrad(Operation):
  def _run(self, outputs, grads):
    outputs_inputs_grads = np.where(outputs <= 0, 0, grads)
    return outputs_inputs_grads

  def _grad_func(self, in_grad_tensors):
    with self._graph.as_default_graph():
      op, tensor_index = self._input_list[0].op, self._input_list[0].tensor_index
      bp_outputs = op.get_zeros_op(tensor_index=tensor_index)
      bp_grads = ReluGrad(input_list=[self._input_list[0], in_grad_tensors[0]]) 
      out_grad_tensors = [Tensor(bp_outputs, 0), Tensor(bp_grads, 0)]
    return out_grad_tensors 


class SoftmaxCrossEntropyWithLogits(Operation):

  def _run(self, logits, labels):
    exp_logits = np.exp(logits - np.max(logits))
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True) 
    loss = np.sum(-np.log(softmax) * labels, axis=-1)
    logits_grads = np.expand_dims(np.ones_like(loss), -1) * (softmax - labels)
    return loss, logits_grads

  def _grad_func(self, in_grad_tensors):
    from array_ops import ExpandDims, Squeeze
    from math_ops import BatchMatMul, Mul, Sub, Neg, Add

    with self._graph.as_default_graph():
      softmax = Softmax(input_list=[self._input_list[0]])
      log_softmax = LogSoftmax(input_list=[self._input_list[0]]) 

      ed = ExpandDims(input_list=[
          in_grad_tensors[0],
          Tensor(Const(value=np.asarray(-1)), 0)
        ])
      ed1 = ExpandDims(input_list=[
          in_grad_tensors[1],
          Tensor(Const(value=np.asarray(1)), 0)
        ])
      ed2 = ExpandDims(input_list=[
          Tensor(softmax, 0),
          Tensor(Const(value=np.asarray(2)), 0)
        ])
    
      mul = Mul(input_list=[Tensor(self, 1), Tensor(ed, 0)])
      neg = Neg(input_list=[Tensor(log_softmax, 0)])
      bp_labels = Mul(input_list=[Tensor(neg, 0), Tensor(ed, 0)])

      bmm = BatchMatMul(input_list=[Tensor(ed1, 0), Tensor(ed2, 0)])
      squeeze = Squeeze(input_list=[Tensor(bmm, 0)], axis=[1])
      sub = Sub(input_list=[
          in_grad_tensors[1],
          Tensor(squeeze, 0),
          ])
      mul1 = Mul(input_list=[Tensor(sub, 0), Tensor(softmax, 0)])
      bp_logits = Add(input_list=[Tensor(mul1, 0), Tensor(mul, 0)])
      out_grad_tensors = [Tensor(bp_logits, 0), Tensor(bp_labels, 0)]
    return out_grad_tensors


class LogSoftmax(Operation):
  """
  tf.raw_ops.LogSoftmax(logits, name=None)
  """

  def _run(self, logits):
    logits = np.exp(logits - np.max(logits))
    softmax = logits / np.sum(logits, axis=-1, keepdims=True)
    outputs = np.log(softmax)
    return outputs

  def _grad_func(self, in_grad_tensors):
    from math_ops import Exp, Sum, Mul, Sub

    with self._graph.as_default_graph():
      exp = Exp(input_list=[Tensor(self, 0)])  
      sum0 = Sum(
          keepdims=True,
          input_list=[
              in_grad_tensors[0],
              Tensor(Const(value=np.asarray([-1])), 0)
          ]
      )
      mul = Mul(input_list=[Tensor(sum0, 0), Tensor(exp, 0)])
      bp_inputs = Sub(input_list=[in_grad_tensors[0], Tensor(mul, 0)])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


class Softmax(Operation):
  def _run(self, inputs):
    logits = np.exp(inputs - np.max(inputs))
    softmax = logits / np.sum(logits, axis=-1, keepdims=True)
    return softmax

  def _grad_func(self, in_grad_tensors):
    from math_ops import Sum, Mul, Sub

    with self._graph.as_default_graph(): 
      mul = Mul(input_list=[in_grad_tensors[0], Tensor(self, 0)])
      sum0 = Sum(
          keepdims=True,
          input_list=[
              Tensor(mul, 0),
              Tensor(Const(value=np.asarray(-1)), 0)
          ]
      )
      sub = Sub(input_list=[in_grad_tensors[0], Tensor(sum0, 0)])
      bp_inputs = Mul(input_list=[Tensor(sub, 0), Tensor(self, 0)])
      out_grad_tensors = [Tensor(bp_inputs, 0)]
    return out_grad_tensors


