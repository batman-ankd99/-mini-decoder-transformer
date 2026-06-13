class MLP:
  def __init__(self,nin,nouts):
    self.layer = []
    sz = [nin] + nouts
    for i in range(len(nouts)):
      self.layer.append(Layers(sz[i],sz[i+1]))

  def __call__(self,x):
    for i in self.layer:
      x = i(x)
    return x

  def parameters(self):
    params = []
    for layer in self.layer:
      params += layer.parameters()
    return params
