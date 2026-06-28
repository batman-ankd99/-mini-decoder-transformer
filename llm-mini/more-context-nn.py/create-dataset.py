import torch
from google.colab import drive
drive.mount('/content/drive')

words = open('/content/drive/MyDrive/names.txt', 'r').read().splitlines()

unique_char = set()
stoi = {}
itos = {}
str1 = '.'.join(words)
for char in str1:
  unique_char.add(char)

for i,j in enumerate(sorted(unique_char)):
  stoi[j] = i
  itos[i] = j

block_size = 3
context = [0] * block_size
X, Y = [], []

for w in words[:5]:
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
X, Y
