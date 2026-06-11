class Value:
    def __init__(self,data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # checks if other is of type Value - if true returns other if not then wraps other in Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
          self.grad += 1 * out.grad
          other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):   # called when left operand doesn't know how to add
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
          self.grad += other.data * out.grad
          other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):   # called when left operand doesn't know how to add
        return self * other

    def __rmul__(self, other):
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out     

    def __pow__(self, other):
      assert isinstance(other, (int,float))
      x = self.data
      out = Value(x**other, (self,), f'**{other}')
      def _backward():
        self.grad = (other*(x**(other-1))) * out.grad
      out._backward = _backward
      return out

    def __neg__(self):
        return self*(-1)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __truediv__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      return self*(other**-1)        #differntiate and Value wrappibng doesnt needed, as it calls mul and power function

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')      # ye aage waali node O hogyi maan lete hai

        def _backward():
            self.grad += (1 - t**2) * out.grad       # iska formula banta  - 1 - o**2, ab is case me to o but koi bhi ho sakti hai, upar out aage wali node hi hai, out.grad coming from ahead we multiple ahead gradient with local gradient. out.grad is ahead nodes gradient which is 1 here.
        out._backward = _backward
        return out

    def backward(self):
      vis = set()
      topo = []                                #first node being the start point of mathematical expression, last entry will be the final output i.e. neuron
      def node_topo(v):                        #DFS way fo sorting Graph nodes
        if v not in vis:
            vis.add(v)
            for child in v._prev:
              node_topo(child)
            topo.append(v)
      node_topo(self)
      self.grad = 1.0
      for node in reversed(topo):
            node._backward()
