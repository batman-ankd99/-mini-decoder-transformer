X, Y = [], []
block_size = 3
context = [0] * block_size
for w in words[:5]:
  for ch in w + ".":
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn(27,2)
W1 = torch.randn(6,100)
b1 = torch.randn(100)
emb = C[X]
#torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], dim=1)
#new_emb = torch.cat(emb.unbind(dim=1))
new_emb = torch.cat(torch.unbind(emb,1), dim =1)
new_emb.shape

h = (new_emb @ W1) + b1
h.shape

W2 = torch.randn(100,27)
b2 = torch.randn(27)

logits = (h @ W2 ) + b2
logits.shape
logits.dtype

count = logits.exp()
count

prob = count / count.sum(dim=1, keepdim=True)
prob
loss = -prob[torch.arange(32), Y].log().mean()
