class LayerNorm(nn.Module):
  def __init__(self, n_embd, eps=1e-05):
    self.eps = eps
    self.bngain = torch.ones(n_embd, requires_grad=True)
    self.bnbias = torch.zeros(n_embd, requires_grad=True)

  def forward(self, x):
    bnmean = x.mean(1, keepdim=True)
    bnvar = x.var(1, keepdim=True)

    xhat = (x - bnmean)/ torch.sqrt(bnvar + self.eps)
    self.out = self.bngain * xhat + self.bnbias
    return self.out   
