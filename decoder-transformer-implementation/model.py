import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config, device

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.head_size = head_size

    self.q = nn.Linear(config.n_embd, head_size)  #head_size is not hyperparam, calculated by Block and passed to Head
    self.k = nn.Linear(config.n_embd, head_size)
    self.v = nn.Linear(config.n_embd, head_size)

    self.register_buffer(
        'tril',
        torch.tril(torch.ones(config.block_size, config.block_size))
        )
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B, T, C = x.shape
    q_vec = self.q(x)                #4,8,32 = x_emb -> out also 4,8,head_size=32
    k_vec = self.k(x)
    v_vec = self.v(x)

    wei = q_vec @ k_vec.transpose(-2,-1) * self.head_size**-0.5           #4,8,8 @ 4,8,8 -> wei.shape = (4,8,8)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)                                      #wei.shape - (4,8,8)
    wei = self.dropout(wei)     #dropped out few attention scores so that some neurons dont communicate in some pass
    out = wei @ v_vec                                                # 4,8,8 @ 4,8,8 -> out.shape = (4,8,8)
    return out

class MultiHead(nn.Module):
    def __init__(self, config, head_size):
      super().__init__()
      self.att_heads = nn.ModuleList(
          [Head(config, head_size) for _ in range(config.n_head)]
      )
      self.proj = nn.Linear(config.n_embd, config.n_embd)
      self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        outputs = []
        for i in self.att_heads:
            outputs.append(i(x))
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))                #linear transformation of outcomes of all heads, combining them together
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

#ffwd with sequential
class FeedForward(nn.Module):
  def __init__(self, config): #n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(config.n_embd, config.n_embd * 4),
        nn.ReLU(),
        nn.Linear(4 * config.n_embd, config.n_embd),
        nn.Dropout(config.dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, config):
      super().__init__()
      head_size = config.n_embd//config.n_head
      self.multi = MultiHead(config, head_size)  #(head_num, head_size)
      self.ffwd = FeedForward(config)
      self.ln1 = LayerNorm(config.n_embd)
      self.ln2 = LayerNorm(config.n_embd)
      self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
     x = x + self.dropout(self.multi(x)) # dropout applied before output of multi head merges back into residual path. original input
     x = self.ln1(x)
     x = x + self.dropout(self.ffwd(x)) # dropout again
     x = self.ln2(x)
     return x

class LayerNorm(nn.Module):
  def __init__(self, n_embd, eps=1e-05):     #self.ln1 = LayerNorm(config.n_embd)
    super().__init__()
    self.eps = eps
    self.bngain = nn.Parameter(torch.ones(n_embd))
    self.bnbias = nn.Parameter(torch.zeros(n_embd))

  def forward(self, x):
    bnmean = x.mean(-1, keepdim=True)
    bnvar = x.var(-1, keepdim=True, unbiased=False)

    xhat = (x - bnmean)/ torch.sqrt(bnvar + self.eps)
    out = self.bngain * xhat + self.bnbias
    return out

class BigramLanguageModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size)                          #output matrix should emit vocab_size scores
    self.ln_final = nn.LayerNorm(config.n_embd)
    self.block_size = config.block_size
    #self.att_head = Head()
    #self.att_heads = MultiHead(head_num, head_size)
    #self.block = Block(head_num, n_embd)
    blocks = []
    for _ in range(config.n_layer):
      blocks.append(Block(config))    #(n_embd, head_num=n_head))
    self.blocks = nn.Sequential(*blocks)

  def forward(self, x, targets=None):
    B, T = x.shape
    tok_emb = self.token_embedding_table(x)                               #B,T,C = n_embd
    pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))     #T, C = n_embd = 32
    x_emb = tok_emb + pos_emb                                             # B, T, C = n_embd = 32   whereas logits to be B, T, vocab_size

    #x_att = self.att_head(x_emb)                                 #x_emb - B,T,C = n_embd
    #x_matt = self.att_heads(x_emb)
    x_emb = self.blocks(x_emb)                                     #attention , feedforward
    x_emb = self.ln_final(x_emb)                                   # Final LayerNorm
    logits = self.lm_head(x_emb)                                   #x_att which is 4,8,32 goes in here, logits -> 4,8,65
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
      idx_cond = idx[:, -self.block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :] #during prediction of last token , consider last token of sequence as bigram model
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
