
import collections
import logging 
import contextlib

from default_stack import _DEFAULT_GRAPH_STACK #, _DEFAULT_NAME_SCOPE_STACK


OpInfo = collections.namedtuple("OpInfo", ["id", "op", "type", "name"])


def get_default_graph():
  if not len(_DEFAULT_GRAPH_STACK):
    _push_graph(Graph())
  graph = _DEFAULT_GRAPH_STACK[-1]
  return graph


def _push_graph(graph):
  _DEFAULT_GRAPH_STACK.append(graph)


def _pop_graph():
  graph = _DEFAULT_GRAPH_STACK.pop()


class Graph(object):

  def __init__(self):
    self._runtime = None

    # dictionary that maps op ID (int) to Op 
    self._ops = dict()
    # dictionary that maps type name of Operation to its count
    self._op_type_count = collections.defaultdict(int)

    # maps tensor (op, tensor_index) to a `Shape` op
    self._shape_tensors = dict()

    self._zeroslike_tensors = dict()

    self._oneslike_tensors = dict()

    self._size_tensors = dict()

    self._rank_tensors = dict()


  def get_op(self, id_):
    return self._ops[id_]

  def _get_type_based_name(self, type_name):
    if type_name not in self._op_type_count:
      new_name = type_name
    else:
      new_name = "_".join([type_name, str(self._op_type_count[type_name])])
    return new_name

  def add_op(self, op, name=None):
    id_ = len(self._ops)
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
    try:
      _push_graph(self)
      yield self
    finally:
      _pop_graph()


class Runtime(object):
  def __init__(self):
    # map: OpName -> list of tensor values
    self._values = collections.defaultdict(list)

  def get_tensor_value(self, tensor):
    tensor_value = self._values[tensor.op.id][tensor.tensor_index]
    return tensor_value



