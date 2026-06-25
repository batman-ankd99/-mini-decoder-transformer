import torch
g = torch.Generator().manual_seed(213457)
w = torch.randn((27,27), generator=g)

def neuro(x, y):
  nll_sum = 0

  xenc = torch.nn.functional.one_hot(x, num_classes=27)
  xenc = xenc.float()
  logits = xenc @ w
  count = logits.exp()
  probs = count/count.sum(dim=1,keepdim=True)
  log_likelihood = torch.log(probs)
  nll = -log_likelihood

  for i in range(len(y)):
    nll_sum += nll[i][y[i]]

  out = nll_sum / len(y)
  return out
