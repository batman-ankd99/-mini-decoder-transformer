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
