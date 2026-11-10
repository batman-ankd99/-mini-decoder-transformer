#part-7, to train on minibatches to fasten up the approach
#part5 - forward pass
for i in range(10000):
  ix = torch.randint(0,len(Xtr),(32,))
  emb = C[Xtr[ix]]
  h = torch.tanh((emb.view(-1,6) @ w1) + b1)
  logits = (h @ w2) + b2
  #count = logits.exp()
  #prob = count/count.sum(dim=1, keepdim=True)
  #loss = -prob[torch.arange(32),y].log().mean()
  loss = torch.nn.functional.cross_entropy(logits,Ytr[ix])

  #part-6 backward pass
  for p in parameters:
    p.grad = None
  loss.backward()


  lr = 0.1 if i < 500 else 0.01
  for p in parameters:
    p.data += -lr*p.grad

print(loss.item())
