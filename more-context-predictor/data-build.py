#part3 data set creation
import random
def build_dataset(words):
  block_size = 3
  x, y = [],[]
  for w in words:
    context = [0] * block_size
    chs = list(w) + ["."]
    for ch in w + '.':
      ix = stoi[ch]
      x.append(context)
      y.append(ix)
      context = context[1:] + [ix]
  x = torch.tensor(x)
  y = torch.tensor(y)

  return x, y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

#part4 parameters
from __future__ import generators
import torch
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
w1 = torch.randn((6,100), generator=g) #100 neuron
b1 = torch.randn(100, generator=g)

w2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, w1, b1, w2, b2]
for p in parameters:
  p.requires_grad = True
