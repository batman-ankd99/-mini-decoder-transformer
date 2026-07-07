for _ in range(100):
  ## forward pass
  Xenc = C[X]
  Xenc = torch.cat(torch.unbind(Xenc,dim=1),dim=1)
  import torch.nn.functional as F
  h = torch.tanh((Xenc @ W1) + b1)
  logits = (h @ W2) + b2
  #counts = logits.exp()
  #probs = counts/counts.sum(dim=1, keepdim=True)
  loss = F.cross_entropy(logits, Y)

  #grads to None
  for params in parameters:
    params.grad = None

  #backward
  loss.backward()
  #update
  lr = 0.01
  for params in parameters:
    params.data =  params.data + (-params.grad * lr)



print(loss.item())
