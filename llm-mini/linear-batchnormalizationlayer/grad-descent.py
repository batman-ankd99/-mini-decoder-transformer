n_embd = 10
block_size = 3
n_hidden = 100
vocab_size = len(unique_char)
C = torch.randn((vocab_size, n_embd))
batch_size = 32


layer1 = Linear(n_embd*block_size, n_hidden)
layer_tan = Tanh()
layer2 = Linear(n_hidden,vocab_size)

layer_bn = BatchNorm1d(n_hidden)

#part -5 -> batch normalization layer as well
for i in range(10):
  ix = torch.randint(0,len(Xtr),(batch_size,))
  emb = C[Xtr[ix]]
  embcat = emb.view(emb.shape[0], -1)

  #forward pass
  hpreact = layer1(embcat)

 #batch norm layer here
  hpreact = layer_bn(hpreact)
 ## BN layer ends

  h = layer_tan(hpreact)
  logits = layer2(h)

  loss = torch.nn.functional.cross_entropy(logits,Ytr[ix])

 #updating gradients to None
  params = layer1.parameters() + layer_bn.parameters() + layer2.parameters()
  for p in params:
    p.grad = None

  loss.backward()
  for p in params:
    p.data += -0.1*p.grad

  print(loss)
