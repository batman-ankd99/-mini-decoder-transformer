import random
import math
class Neuron:
  def __init__(self,nin):
    self.b = Value(random.uniform(-1,1))
    self.w = []
    for i in range(nin):
      self.w.append(Value(random.uniform(-1,1)))
  def __call__(self,x):
    summation = self.b
    for i,j in zip(self.w, x):
      summation += i*j
    out = summation.tanh()
    return out
  def parameters(self):
    return self.w + [self.b]
