import torch
import random
class Neuron:
  def __init__(self, nin):
    self.b = Value(random.uniform(-1,1))
    self.w = []
    for i in range(nin):
      self.w.append(Value(random.uniform(-1,1)))

  def __call__(self, x):
    summation = self.b
    l=[]
    for xi,wi in zip(self.w, x):
      l.append(xi*wi)
    for i in l:
      summation += i
    return summation

x = [1.4, 4.7]
n1 = Neuron(len(x))
