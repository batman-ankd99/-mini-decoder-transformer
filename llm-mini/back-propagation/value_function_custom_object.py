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
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
          self.grad += 1 * out.grad
          other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = 0.0
        for child in self._prev:
            child.zero_grad()
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
          self.grad += other.data * out.grad
          other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')      # ye aage waali node O hogyi maan lete hai

        def _backward():
            self.grad += (1 - t**2) * out.grad       # iska formula banta  - 1 - o**2, ab is case me to o but koi bhi ho sakti hai, upar out aage wali node hi hai, out.grad coming from ahead we multiple ahead gradient with local gradient. out.grad is ahead nodes gradient which is 1 here.
        out._backward = _backward
        return out
