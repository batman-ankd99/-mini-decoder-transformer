class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.momentum = momentum
    self.eps = eps
    self.training = True
    self.gamma = torch.ones(dim, requires_grad=True)    #bngain
    self.beta = torch.zeros(dim, requires_grad=True)      #bnbias

    self.mean_running = torch.zeros(dim)
    self.var_running = torch.ones(dim)

  def __call__(self,x):
    if self.training:
      xmean = x.mean(0, keepdim=True)
      xvar = x.var(0, keepdim=True)
    else:
      xmean = self.mean_running
      xvar = self.var_running

    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  #normalizing , x is output of linear layer
    self.out = self.gamma * xhat + self.beta

    #updaing running mean and var

    if self.training:
      with torch.no_grad():
        self.mean_running = (1 - self.momentum) * self.mean_running + self.momentum * xmean
        self.var_running = (1 - self.momentum) * self.var_running + self.momentum * xvar

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

 # hpreact = (bngain * (hpreact - hpreact.mean(0, keepdim=True)) / (hpreact.std(0, keepdim=True)) ) + bnbias
     
