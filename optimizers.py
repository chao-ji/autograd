import numpy as np

from .operation import backprop as _backprop


class Optimizer(object):
  """Base class of all Optimizers.

  Subclasses must implement abstract method `apply_gradients`.
  """

  def __init__(self, **params):
    """Constructor.

    Args:
      params: a dict mapping from parameter names to parameters.
    """
    self._params = params
    self._params_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items()])

  def __repr__(self):
    """Displays the initializer name and list of parameter name and value pairs.
    """
    if self._params_str:
      return '<%s:%s>' % (type(self).__name__, self._params_str)
    else:
      return '<%s>' % type(self).__name__

  def compute_gradients(self, tensor, variables):
    variables = [(v.weight, v.handle) for v in variables if v.trainable]
    var_weights = list(list(zip(*variables))[0])
    var_handles = list(list(zip(*variables))[1])

    gradients = _backprop([tensor], var_weights)
    grads_and_vars = list(zip(gradients, var_handles))
    return grads_and_vars


class GradientDescentOptimizer(Optimizer):
  """The Vanilla Gradient Descent Optimizer."""

  def apply_gradients(self, grads_and_vars, reset_runtime=True):
    """Apply the computed gradient w.r.t. trainable variables.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
      reset_runtime (bool): whether to reset runtime after variables are updated
        . Defaults to True.
    """
    runtime = grads_and_vars[0][0].op.graph.runtime
    for grad, var in grads_and_vars:
      var_id = var.eval().item().id
      var_value = runtime.get_variable_value(var_id).astype("float32")
      grad_value = grad.eval().astype("float32")

      runtime.set_variable_value(
          var_id, var_value - self._params["alpha"] * grad_value,
      )
    if reset_runtime:
      runtime.reset()


class AdamOptimizer(Optimizer):
  """Adam optimizer"""

  def __init__(self, **params):
    """Constructor.

    Args:
      params: a dict mapping from parameter names to parameters.
    """
    self._params = params
    self._params_str = ', '.join([
        '%s=%s' % (k, v)
        for k, v in params.items()
        if k in ('alpha', 'beta1', 'beta2', 'epsilon')
    ])

    self._t = 0
    self._m = dict()
    self._v = dict()

  def apply_gradients(self, grads_and_vars, reset_runtime=True):
    """Apply the computed gradient w.r.t. trainable variables.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
      reset_runtime (bool): whether to reset runtime after variables are updated
        . Defaults to True.
    """
    alpha, beta1, beta2, epsilon = (
        np.asarray(
            self._params['alpha'],
            "float32",
        ), np.asarray(self._params['beta1'], "float32"),
        np.asarray(
            self._params['beta2'],
            "float32",
        ), np.asarray(self._params['epsilon'], "float32"),
    )
    t = self._t + 1
    m = self._m
    v = self._v
    alpha_t = alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t))
    alpha_t = alpha_t.astype('float32')

    runtime = grads_and_vars[0][0].op.graph.runtime
    for grad, var in grads_and_vars:
      var_id = var.eval().item().id
      var_shape = var.eval().item().shape
      var_value = runtime.get_variable_value(var_id).astype("float32")
      grad_value = grad.eval().astype("float32")

      m[var_id] = beta1 * m.get(
          var_id, np.zeros(var_shape, dtype="float32"),
      ) + (1 - beta1) * grad_value
      v[var_id] = beta2 * v.get(
          var_id, np.zeros(var_shape, dtype="float32"),
      ) + (1 - beta2) * grad_value * grad_value
      runtime.set_variable_value(
          var_id,
          var_value - alpha_t * m[var_id] / (np.sqrt(v[var_id]) + epsilon),
      )

    self._m = m
    self._v = v
    self._t = t
    if reset_runtime:
      runtime.reset()


class RMSPropOptimizer(Optimizer):
  """RMSProp Optimizer"""

  def __init__(self, **params):
    """Constructor.

    Args:
      params: a dict mapping from parameter names to parameters.
    """
    self._params = params
    self._params_str = ', '.join([
        '%s=%s' % (k, v)
        for k, v in params.items()
        if k in ('alpha', 'rho', 'momentum', 'epsilon')
    ])

    self._mean_square = dict()
    self._moment = dict()

  def apply_gradients(self, grads_and_vars, reset_runtime=True):
    """Apply the computed gradient w.r.t. trainable variables.

    Args:
      grads_and_vars: a list of (gradient, variable) pairs, where gradient is
        numpy array, and variable is a Node instance.
      reset_runtime (bool): whether to reset runtime after variables are updated
        . Defaults to True.
    """
    alpha, rho, momentum, epsilon = (
        self._params['alpha'], self._params['rho'], self._params['momentum'],
        self._params['epsilon'],
    )

    mean_square = self._mean_square
    moment = self._moment

    runtime = grads_and_vars[0][0].op.graph.runtime
    for grad, var in grads_and_vars:
      var_id = var.eval().item().id
      var_shape = var.eval().item().shape
      var_value = runtime.get_variable_value(var_id)
      grad_value = grad.eval().astype("float32")

      mean_square[var_id] = (
          rho * mean_square.get(var_id, np.zeros(var_shape)) +
          (1 - rho) * grad_value * grad_value
      )

      moment[var_id] = momentum * moment.get(
          var_id, np.zeros(var_shape),
      ) + alpha * grad_value / (
          np.sqrt(mean_square[var_id]) + epsilon
      )
      runtime.set_variable_value(var_id, var_value - moment[var_id])

    self._mean_square = mean_square
    self._moment = moment
    if reset_runtime:
      runtime.reset()
