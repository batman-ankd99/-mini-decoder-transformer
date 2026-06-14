from google.colab import drive
drive.mount('/content/drive')

words = open('/content/drive/MyDrive/names.txt', 'r').read().splitlines()
for w in words[:2]:
  print(w)

import torch
dict1 = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1,ch2 in zip(chs,chs[1:]):
    if (ch1+ch2) in dict1:
      dict1[ch1+ch2] += 1
    else:
      dict1[ch1+ch2] = 1
print(dict1)

str1 = ' '.join(words)
print(str1)

chars = set()
for char in str1:
  if char.isalpha():
    chars.add(char)

print(len(chars))

stoi = {}

for i,j in enumerate(sorted(chars)):
  stoi[j] = i
print(stoi)

stoi['<S>'] = 26
stoi['<E>'] = 27

##
itos = {}
for i,j in enumerate(sorted(chars)):
    itos[i] = j
itos[26] = '<S>'
itos[27] = '<E>'    

N = torch.zeros((28,28), dtype=torch.int32)

for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1,ch2 in zip(chs,chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1,ix2] += 1

print(N)
