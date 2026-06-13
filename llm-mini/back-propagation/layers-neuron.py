class Layers:
  def __init__(self,nin,nout):
    self.neurons = []
    for i in range(nout):
      self.neurons.append(Neuron(nin))

  def __call__(self,x):
    outs = []
    for neuro in self.neurons:
      outs.append(neuro(x))
    return outs
