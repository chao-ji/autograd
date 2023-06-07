from operation import Operation

class Const(Operation):
  def __init__(self, value, graph=None, name=None):
    super(Const, self).__init__(graph=graph, name=name)
    self._value = value

  def _run(self):
    """Returns numpy array."""
    return self._value   


class Placeholder(Operation):
  pass


class Variable(Operation):

  def __init__(self, initializaer, graph=None, name=None):
    pass




