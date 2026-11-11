#block-multi # Transformer implementation # #
#Hyperparameters - code starts

import torch
from config import Config, device
from model import BigramLanguageModel

config = Config() #config object created to be passed in model classes

print("Using device:", device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))

uniq_char = list(sorted(set(text)))
vocab_size = len(uniq_char)
config.vocab_size = vocab_size

stoi, itos = {}, {}

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

  ix = torch.randint(0, len(data)- config.block_size, (config.batch_size,))
  x = torch.stack([data[i:i+ config.block_size] for i in ix])
  targets = torch.stack([data[i+1:i+ config.block_size+1] for i in ix])
  return x.to(device),targets.to(device)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    loss_int = []
    for i in range(config.eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      loss_int.append(loss.item())
    out[split] = torch.tensor(loss_int).mean().item()
  model.train()
  return out

######optimizer ######

model = BigramLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for step in range(config.max_iters):
  if step % config.eval_interval==0:
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
  #sample a batch to get loss
  xb, yb = get_batch('train')
  logits, loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,config.block_size), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
