import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

#part1 - hyperparams
un_char = set(text)
batch_size = 4
max_iters = 3000
eval_iters = 300
block_size = 8
vocab_size = len(un_char)
n_emdb = 32 #self attention model, bigram 65

#part-2 itos and stoi, encode and decode
stoi, itos = {}, {}
for i,ch in enumerate(un_char):
  stoi[ch] = i
  itos[i] = ch

def encode(str1):
  enc_num = []
  for ch in str1:
    enc_num.append(stoi[ch])
  return enc_num

def decode(list1):
  dec_str = []
  for i in list1:
    dec_str.append(itos[i])
  return ''.join(dec_str)

data = torch.tensor(encode(text), dtype = torch.long)

#part-3
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

#part - 4
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(0, len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  targets = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, targets

#part-5 bigramlanguage class
class BigramLanguageModel(nn.Module):
  def __init__(self):
  #def __init__(self,vocab_size):
    super().__init__()
    #self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) -> bigram model
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #-> self-attention model

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) #-> in bigram they were logits
    #tok_emb = self.token_embedding_table(idx) #-> now just token embeddings
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x, y)
      losses[k] = loss.item()

    out[split] = losses.mean()
  model.train()
  return out
