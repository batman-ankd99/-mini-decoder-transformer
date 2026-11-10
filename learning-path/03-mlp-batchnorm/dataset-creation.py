#part2
#stoi, itos

import torch
unique_char = set()
stoi = {}
itos = {}

str1 = '.'.join(words)
for char in str1:
  unique_char.add(char)

for i,j in enumerate(sorted(unique_char)):
  stoi[j] = i
  itos[i] = j


#part3 data set creation
import random
def build_dataset(words):
  block_size = 3
  x, y = [],[]
  for w in words:
    context = [0] * block_size
    chs = list(w) + ["."]
    for ch in w + '.':
      ix = stoi[ch]
      x.append(context)
      y.append(ix)
      context = context[1:] + [ix]
  x = torch.tensor(x)
  y = torch.tensor(y)

  return x, y
