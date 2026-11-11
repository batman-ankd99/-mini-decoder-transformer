#block-multi # Transformer implementation # #
#Hyperparameters - code starts

import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))

batch_size = 64     #how many independent sequences will be trained in parallel, every sequence is of n number of tokens together, n = block_size
block_size = 256 # maximum context length for predictions
n_embd = 384
head_size = 8
max_iters = 5000
eval_interval = 500
uniq_char = list(sorted(set(text)))
vocab_size = len(uniq_char)
eval_iters = 200
head_num = 4
n_layer = 6
n_head = 6 #no. of heads
dropout = 0.2
learning_rate = 3e-4

stoi, itos = {}, {}

for i, char in enumerate(uniq_char):
  stoi[char] = i
  itos[i] = char

#Hyperparameters - code starts

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
  data = train_data if split == 'train' else val_data

  ix = torch.randint(0, len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  targets = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x.to(device),targets.to(device)

x_train, y_train = get_batch('train')

@torch.no_grad
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    loss_int = []
    for i in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      loss_int.append(loss.item())
    out[split] = torch.tensor(loss_int).mean()
  model.train()
  return out

######optimizer ######

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
losses = estimate_loss()

for iter in range(max_iters):
  if iter % eval_interval==0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
  #sample a batch to get loss
  xb, yb = get_batch('train')
  logits, loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
