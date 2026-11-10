from google.colab import drive
drive.mount('/content/drive')

words = open('/content/drive/MyDrive/names.txt', 'r').read().splitlines()
for w in words[:2]:
  print(w)

unique_char = set()
stoi = {}
itos = {}
str1 = '.'.join(words)
for char in str1:
  unique_char.add(char)

for i,j in enumerate(sorted(unique_char)):
  stoi[j] = i
  itos[i] = j  

import  torch
xs, ys = [], []
for w in words[:1]:
  chs = ["."] + list(w) + ["."]
  for ch1,ch2 in zip(chs,chs[1:]):
    xs.append(stoi[ch1])
    ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs, ys)

xenc = torch.nn.functional.one_hot(xs, num_classes=27)
xenc= xenc.float()
yenc = torch.nn.functional.one_hot(ys, num_classes=27)
yenc= yenc.float()
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g)

logits = xenc @ W
count = logits.exp()
probs = count / count.sum(dim=1, keepdim=True)
log_likelihood = torch.log(probs)
nll = -log_likelihood
probs[0]
