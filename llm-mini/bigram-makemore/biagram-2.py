P = N.float()                  #to create a tensor with prob distribution
P = P / P.sum(1, keepdim = True)

g = torch.Generator().manual_seed(2147483647)

for i in range(10):

  out = []
  ix = 0
  while True:
    #p = N[ix].float()
    #p = p/p.sum()
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1,replacement=True, generator=g).item()
    out.append(itos[ix])
  #  print(itos[ix])
    if ix == 0:
      break

  print(''.join(out))
