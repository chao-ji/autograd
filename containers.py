
import collections
import warnings


OpInfo = collections.namedtuple("OpInfo", ["id", "op", "type", "name"])


class Graph(object):

  def __init__(self):
    self._runtime = None

    # dictionary that maps Operation to OpInfo
    self._ops = dict()
    # dictionary that maps type name of Operation to its count
    self._op_type_count = collections.defaultdict(int)

    self._shape_ops = dict()

    self._zeros_ops = dict()

    self._ones_ops = dict()

    self._size_ops = dict()

    self._rank_ops = dict()

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
        warnings.warn(f"Op name {name} already used. Ranmed to {new_name}.")
    else:
      name = self._get_type_based_name(type_name)
    self._op_type_count[type_name] += 1
    self._ops[op] = OpInfo(id=op_id, op=op, type=type_name, name=name)


class Runtime(object):
  def __init__(self):
    # map: OpName -> list of tensor values
    self._values = collections.defaultdict(list)

