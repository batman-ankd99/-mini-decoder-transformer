#block-multi # Transformer implementation # #
#Hyperparameters - code starts

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8
n_embd = 32
head_size = 8
max_iters = 200000
eval_interval = 5000
uniq_char = list(sorted(set(text)))
vocab_size = len(uniq_char)
eval_iters = 20
head_num = 4
n_layers = 4
n_head = 4 #no. of heads

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
  return x,targets

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

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.q = nn.Linear(n_embd, head_size)
    self.k = nn.Linear(n_embd, head_size)
    self.v = nn.Linear(n_embd, head_size)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    q_vec = self.q(x)                #4,8,32 = x_emb -> out also 4,8,head_size=32
    k_vec = self.k(x)
    v_vec = self.v(x)

    wei = q_vec @ k_vec.transpose(-2,-1) * head_size**-0.5           #4,8,8 @ 4,8,8 -> wei.shape = (4,8,8)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)                                      #wei.shape - (4,8,8)

    out = wei @ v_vec                                                # 4,8,8 @ 4,8,8 -> out.shape = (4,8,8)
    return out

class MultiHead(nn.Module):
    def __init__(self, head_num, head_size):
      super().__init__()
      self.att_heads = nn.ModuleList([Head(head_size) for _ in range(head_num)])
      self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        outputs = []
        for i in self.att_heads:
            outputs.append(i(x))
        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)                  #linear transformation of outcomes of all heads, combining them together
        return out

#class FeedForward:
 # def __init__(self, n_embd):
  #  super().__init__()
   # self.lin = nn.Linear(n_embd, n_embd)
#    self.relu = nn.Relu()

 # def forward(self, x):
  #  x = self.lin(x)
   # x = self.relu(output)
  #  return x

#ffn with sequential
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd * 4),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, head_num):
      super().__init__()
      head_size = n_embd//head_num
      self.multi = MultiHead(head_num, head_size)
      self.ffn = FeedForward(n_embd)

  def forward(self, x):
    x = x + self.multi(x)
    x = x +self.ffn(x)
    return x

class LayerNorm(nn.Module):
  def __init__(self, n_embd, eps=1e-05):
    self.eps = eps
    self.bngain = torch.ones(n_embd, requires_grad=True)
    self.bnbias = torch.zeros(n_embd, requires_grad=True)

  def forward(self, x):
    bnmean = x.mean(1, keepdim=True)
    bnvar = x.var(1, keepdim=True)

    xhat = (x - bnmean)/ torch.sqrt(bnvar + self.eps)
    self.out = self.bngain * xhat + self.bnbias
    return self.out

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)                          #output matrix should emit vocab_size scores
    #self.att_head = Head()
    #self.att_heads = MultiHead(head_num, head_size)
    #self.block = Block(head_num, n_embd)
    blocks = []
    for _ in range(n_layers):
      blocks.append(*Block(n_embd, head_num=n_head))
    self.blocks = nn.Sequential(*blocks)
    #self.blocks = nn.Sequential(
    #    Block(n_embd, head_num=4),
    #    Block(n_embd, head_num=4),
    #    Block(n_embd, head_num=4),
    #)

  def forward(self, x, targets=None):
    B, T = x.shape
    tok_emb = self.token_embedding_table(x)                               #B,T,C = n_embd
    pos_emb = self.position_embedding_table(torch.arange(T))     #T, C = n_embd = 32
    x_emb = tok_emb + pos_emb                                             # B, T, C = n_embd = 32   whereas logits to be B, T, vocab_size

    #x_att = self.att_head(x_emb)                                 #x_emb - B,T,C = n_embd
    #x_matt = self.att_heads(x_emb)
    x_emb = self.blocks(x_emb)

    logits = self.lm_head(x_emb)                                  #x_att which is 4,8,32 goes in here, logits -> 4,8,65
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape                                                  #B, T, vocab_size
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      #idx= 00000000
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :] #during prediction of last token , consider last token of sequence as bigram model
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
######optimizer ######

model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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

context = torch.zeros((1,block_size), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
