max_steps=100
lossi=[]
ud = []

for i in range(max_steps):
  ix = torch.randint(0,len(Xtr), (batch_size,), generator=g)
  data = C[Xtr[ix]]
  x = data.view(data.shape[0], -1)

  for layer in layers:
    x = layer(x)

  loss = torch.nn.functional.cross_entropy(x, Ytr[ix])
  for layer in layers:
    layer.out.retain_grad()

  for p in parameters:
    p.grad = None

  loss.backward()
  lr = 0.1 if i<100000 else 0.01
  for p in parameters:
    p.data = p.data - lr*p.grad

  with torch.no_grad():
    ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])

  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  
