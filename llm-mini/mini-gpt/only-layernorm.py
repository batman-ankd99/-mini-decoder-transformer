class LayerNorm:
  def __init__(self, neurons, eps=1e-05):
    self.eps = eps
    self.track_running_training=True

    self.bngain = torch.ones(neurons, requires_grad=True)
    self.bnbias = torch.zeros(neurons, requires_grad=True)

  def __call__(self, x):

    bnmean = x.mean(1, keepdim=True)
    bnvar = x.var(1, keepdim=True)

    xhat = (x - bnmean)/ torch.sqrt(bnvar + self.eps)
    self.out = self.bngain * xhat + self.bnbias

    return self.out

  def parameters(self):
    return [self.bngain, self.bnbias]
