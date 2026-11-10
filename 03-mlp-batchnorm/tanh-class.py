class Tanh():
  def __call__(self,x):
    self.out = torch.tanh(x)
    return self.out
