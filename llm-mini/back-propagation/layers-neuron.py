class Layers:
  def __init__(self,nin,nout):
    self.neurons = []
    for i in range(nout):
      self.neurons.append(Neuron(nin))

  def __call__(self,x):
    outs = []
    for neuro in self.neurons:
      outs.append(neuro(x))
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    out = []
    for neuro in self.neurons:
      out += neuro.paramters()
    return out

          
