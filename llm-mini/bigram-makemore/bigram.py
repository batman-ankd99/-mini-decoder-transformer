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
