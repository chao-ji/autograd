
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

    # dictionary that maps Operation to OpInfo
    self._ops = dict()
    # dictionary that maps type name of Operation to its count
    self._op_type_count = collections.defaultdict(int)


    # maps tensor (op, tensor_index) to a `Shape` op
    self._shape_ops = dict()

    self._zeros_ops = dict()

    self._ones_ops = dict()

    self._size_ops = dict()

    self._rank_ops = dict()


    self._name_to_op = dict()

    self._id_to_op = dict()


  def get_op(self, name):
    return self._name_to_op[name]

  def _get_type_based_name(self, type_name):
    if type_name not in self._op_type_count:
      new_name = type_name
    else:
      new_name = "_".join([type_name, str(self._op_type_count[type_name])])
    return new_name

  def add_op(self, op, name=None):
    op_id = len(self._ops)
    type_name = type(op).__name__

    if name is not None:
      if name in self._ops:
        new_name = self._get_type_based_name(type_name)
        logging.warning(f"Op name {name} already used. Renamed to {new_name}.")
    else:
      name = self._get_type_based_name(type_name)
    self._op_type_count[type_name] += 1
    self._ops[op] = OpInfo(id=op_id, op=op, type=type_name, name=name)
    self._name_to_op[name] = OpInfo(id=op_id, op=op, type=type_name, name=name)
    self._id_to_op[op_id] = OpInfo(id=op_id, op=op, type=type_name, name=name)

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

