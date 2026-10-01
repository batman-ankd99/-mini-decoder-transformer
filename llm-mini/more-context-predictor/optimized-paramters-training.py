#part4 parameters
from __future__ import generators
import torch
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,10), generator=g)
w1 = torch.randn((30,300), generator=g) #100 neuron #modified to 300
b1 = torch.randn(300, generator=g)

w2 = torch.randn((300,27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, w1, b1, w2, b2]
for p in parameters:
  p.requires_grad = True
  
#part-7, to train on minibatches to fasten up the approach
#part5 - forward pass
for i in range(20000):
  ix = torch.randint(0,len(Xtr),(32,))
  emb = C[Xtr[ix]]
  h = torch.tanh((emb.view(-1,30) @ w1) + b1)
  logits = (h @ w2) + b2
  #count = logits.exp()
  #prob = count/count.sum(dim=1, keepdim=True)
  #loss = -prob[torch.arange(32),y].log().mean()
  loss = torch.nn.functional.cross_entropy(logits,Ytr[ix])

  #part-6 backward pass
  for p in parameters:
    p.grad = None
  loss.backward()


  lr = 0.1 if i < 10000 else 0.01
  for p in parameters:
    p.data += -lr*p.grad

print(loss.item())
