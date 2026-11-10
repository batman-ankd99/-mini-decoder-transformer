#part5 - forward pass
for i in range(110):
  emb = C[Xtr]
  h = torch.tanh((emb.view(-1,6) @ w1) + b1)
  logits = (h @ w2) + b2
  #count = logits.exp()
  #prob = count/count.sum(dim=1, keepdim=True)
  #loss = -prob[torch.arange(32),y].log().mean()
  loss = torch.nn.functional.cross_entropy(logits,Ytr)

  #part-6 backward pass
  for p in parameters:
    p.grad = None
  loss.backward()


  lr = 0.1 if i < 100 else 0.01
  for p in parameters:
    p.data += -lr*p.grad

print(loss.item())

#part-6, to check on dev dataset
emb = C[Xdev]
h = torch.tanh((emb.view(-1,6) @ w1) + b1)
logits = (h @ w2) + b2
loss = torch.nn.functional.cross_entropy(logits,Ydev)
loss
