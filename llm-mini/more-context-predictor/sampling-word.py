#part-7, sampling of words from the above trained neural network
for _ in range(10):
  context = [0,0,0]
  out = []
  while True:
    emb = C[torch.tensor(context)]   #converting characters in contexting in 2-D embedding
    h = torch.tanh((emb.view(-1,6) @ w1) + b1)
    logits = (h @ w2) + b2
    count = logits.exp()
    prob = count/count.sum(dim=1, keepdim=True)

    #sample next char
    ix = torch.multinomial(prob,num_samples=1,replacement=True).item()

    if ix == 0:
      break

    out.append(itos[ix])
    context = context[1:] + [ix]
  print("".join(out))  
