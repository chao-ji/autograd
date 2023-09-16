"""`Graph` and `Runtime` class for defining and executing computational graph
."""
import collections
import contextlib
import logging

from .default_stack import _DEFAULT_GRAPH_STACK  # , _DEFAULT_NAME_SCOPE_STACK


def get_default_graph():
  """Returns the default `Graph`."""
  if not len(_DEFAULT_GRAPH_STACK):
    _push_graph(Graph())
  graph = _DEFAULT_GRAPH_STACK[-1]
  return graph


def _push_graph(graph):
  _DEFAULT_GRAPH_STACK.append(graph)


def _pop_graph():
  graph = _DEFAULT_GRAPH_STACK.pop()


class Graph(object):
  """A `Graph` is an object that contains a set of interconnected Ops that form
  a DAG which describes a computational workflow.
  """

  def __init__(self):
    """Constructor."""
    self._runtime = Runtime()

    # dict that maps op ID (int) to Op
    self._ops = dict()
    # dict that maps type name of Operation to its count
    self._op_type_count = collections.defaultdict(int)

    # the following dicts map tensor (op, tensor_index) to a `Shape` op
    self._shape_tensors = dict()
    self._zeroslike_tensors = dict()
    self._oneslike_tensors = dict()
    self._size_tensors = dict()
    self._rank_tensors = dict()

  @property
  def runtime(self):
    return self._runtime

  def get_op(self, id_):
    """Return an Op given its ID."""
    return self._ops[id_]

  def _get_type_based_name(self, type_name):
    """Return the name of an Op based on its type.

    Args:
      type_name (str): type of the Op.

    Returns:
      new_name (str): the name of the Op.
    """
    # return the type name as the name of the Op, if it's not been seen
    if type_name not in self._op_type_count:
      new_name = type_name
    # or generate a name in the format of "[type_name]_[count]"
    else:
      new_name = "_".join([type_name, str(self._op_type_count[type_name])])
    return new_name

  def add_op(self, op, name=None):
    """Add a new Op to a graph.

    Args:
      op (Operation): the Op to be added to the `Graph`.
      name (str): (Optional) the name of the Op.
    """

    # `op` is given ID `id_`, which is the number of Ops in the graph before
    # it's added.
    id_ = len(self._ops)

    # type of the Op, e.g. `Add`, `Subtract` etc.
    type_ = type(op).__name__

    if name is not None:
      if name in self._ops:
        new_name = self._get_type_based_name(type_)
        logging.warning(f"Op name {name} already used. Renamed to {new_name}.")
    else:
      name = self._get_type_based_name(type_)
    self._op_type_count[type_] += 1

    op._id = id_
    op._type = type_
    op._name = name

    self._ops[id_] = op

  @contextlib.contextmanager
  def as_default_graph(self):
    """Create a context manager in which all Ops are defined within the scope of
    this `Graph`.
    """
    try:
      _push_graph(self)
      yield self
    finally:
      _pop_graph()


class Runtime(object):
  """A `Runtime` is an object in which a `Graph` is executed. It maintains the
  values of the symbolic `Tensor`s as the Ops are being run.
  """

  def __init__(self):
    """Constructor."""

    # map: OpName -> list of tensor values
    self._values = collections.defaultdict(list)
    self._variable_values = dict()
    self._placeholder_values = dict()

  def get_variable_value(self, creator_id):
    """Retrieve the value of a variable.

    Args
      creator_id (int): the ID of the variable, i.e. the ID of the
        `CreateVariable` Op in the `Graph`.

    Returns
      variable_value (nd.array): the value of the variable.
    """
    variable_value = self._variable_values[int(creator_id)]
    return variable_value

  def set_variable_value(self, creator_id, new_value):
    """Set the value of a variable.

    Args:
      creator_id (int): the ID of the variable, i.e. the ID of the
        `CreateVariable` Op in the `Graph`.
      new_value (nd.array): the new value of the variable.
    """
    self._variable_values[creator_id] = new_value

  def reset(self):
    """Reset the values of the `Tensor`s and `Placeholder`s."""
    self._values = collections.defaultdict(list)
    self._placeholder_values = dict()
