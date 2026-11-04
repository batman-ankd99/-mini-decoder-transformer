#self-attention

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

#Hyperparameters - code starts

batch_size = 4
block_size = 8
n_embd = 32
head_size = 32
max_iters = 3000
eval_iter = 200
uniq_char = list(sorted(set(text)))
vocab_size = len(uniq_char)

#itos, stoi
for i, char in enumerate(uniq_char):
  stoi[char] = i
  itos[i] = char

def encode(str1):
  encode_input = []
  for char in str1:
    encode_input.append(stoi[char])

  return encode_input

def decode(list1):
  decoded_char = []
  for num in list1:
    decoded_char.append(itos[num])
  return ''.join(decoded_char)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train_data' else 'val_data'

  ix = torch.randint(0, len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  targets = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,targets

x_train, y_train = get_batch('train_data')

class Head(nn.Module):
  pass


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)                          #output matrix should emit vocab_size scores

  def forward(self, x, targets=None):
    B, T = x.shape
    tok_emb = self.token_embedding_table(x)                               #B,T,C = n_embd
    pos_emb = self.position_embedding_table(torch.arange(T))     #T, C = n_embd
    x_emb = tok_emb + pos_emb                                             # B, T, C = n_embd      whereas logits to be B, T, vocab_size
    logits = self.lm_head(x_emb)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape                                                  #B, T, vocab_size
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    return logits, loss
