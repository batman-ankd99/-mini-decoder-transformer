import torch
g = torch.Generator().manual_seed(213457)
W = torch.randn((27,27), generator=g)

#### creating training data set
xs, ys = [], []
for w in words[:1]:
  chs = ["."] + list(w) + ["."]
  for ch1,ch2 in zip(chs,chs[1:]):
    xs.append(stoi[ch1])
    ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

## creating neuron function

def neuro(x, y):
  nll_sum = 0

  xenc = torch.nn.functional.one_hot(x, num_classes=27)
  xenc = xenc.float()
  logits = xenc @ W
  count = logits.exp()
  probs = count/count.sum(dim=1,keepdim=True)
  log_likelihood = torch.log(probs)
  nll = -log_likelihood

  for i in range(len(y)):
    nll_sum += nll[i][y[i]]

  out = nll_sum / len(y)
  return out

neuro(xs, ys)
