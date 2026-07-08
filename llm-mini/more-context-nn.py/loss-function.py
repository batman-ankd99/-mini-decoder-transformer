import torch.nn.functional as F
#def lossfunc(Xbatch, Ybatch):
for _ in range(1000):
  ix = torch.randint(0, len(X), (32,)) #sampling any random 32 indexes from 2 lakh indexes

  ## forward pass
  emb = C[X[ix]]    #(32,3,2)
#    h = torch.cat(torch.unbind(Xenc,dim=1),dim=1)
  h = torch.tanh(emb.view(-1,6) @ W1 + b1)
  logits = (h @ W2) + b2
  #counts = logits.exp()
  #probs = counts/counts.sum(dim=1, keepdim=True)
  loss = F.cross_entropy(logits, Y[ix])

  #backward
  #grads to None
  for p in parameters:
    p.grad = None
  loss.backward()
  #update
  lr = 0.01
  for p in parameters:
    p.data =  p.data + (-p.grad * lr)

print(loss.item())



#2nd function - to get Loss overall complete data

emb = C[X]
h = torch.tanh(emb.view(-1,6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
loss
