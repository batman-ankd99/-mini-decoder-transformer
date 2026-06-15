from google.colab import drive
drive.mount('/content/drive')

words = open('/content/drive/MyDrive/names.txt', 'r').read().splitlines()
for w in words[:2]:
  print(w)

import torch
dict1 = {}
for w in words:
  chs = ['.'] + list(w) + ['.']
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
  stoi[j] = i + 1 #we are giving . the 0th position
print(stoi)

#stoi['<S>'] = 26
#stoi['<E>'] = 27
stoi['.'] = 0
##
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((27,27), dtype=torch.int32)

for w in words:
  #chs = ['<S>'] + list(w) + ['<E>']
  chs = ['.'] + list(w) + ['.']
  for ch1,ch2 in zip(chs,chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1,ix2] += 1

print(N)
