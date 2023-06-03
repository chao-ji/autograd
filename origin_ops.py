from operation import Operation

class Const(Operation):
  def __init__(self, value, graph, name=None):
    super(Const, self).__init__(graph=graph, name=name)
    self._value = value

  def _run(self):
    """Returns numpy array."""
    return self._value   


  def backprop(self, bwval_list):
    """no-op"""
    return


class Placeholder(Operation):

  pass


class Variable(Operation):

  def __init__(self, initializaer, graph, name=None):
    pass


class Fill(Operation):
  pass


