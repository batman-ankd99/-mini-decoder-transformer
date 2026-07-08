def lossfunc(Xbatch, Ybatch):
  for _ in range(50):
    ## forward pass
    Xenc = C[Xbatch]
    Xenc = torch.cat(torch.unbind(Xenc,dim=1),dim=1)
    import torch.nn.functional as F
    h = torch.tanh((Xenc @ W1) + b1)
    logits = (h @ W2) + b2
    #counts = logits.exp()
    #probs = counts/counts.sum(dim=1, keepdim=True)
    loss = F.cross_entropy(logits, Ybatch)

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


## sampling 32 data samples

ix = torch.randint(0, len(X), (32,))     #sampling any random 32 indexes from 2 lakh indexes of total sample
Xdata = X[ix]    #not embeded yet
Ydata = Y[ix]
lossfunc(Xdata, Ydata)
