#stoi, itos, dataset creation

import torch
import random
block_size = 8
unique_char=set()

str1 = '.'.join(words[:100])

for char in str1:
  unique_char.add(char)

stoi, itos = {}, {}

for i, character in enumerate(sorted(unique_char)):
  stoi[character] = i
  itos[i] = character

def build_dataset(words):
#  block_size = 8
  x,y = [],[]

  for w in words:
    context = [0] * block_size
    chs = w + '.'
    for char in chs:
      ix = stoi[char]
      x.append(context)
      y.append(ix)
      context = context[1:] + [ix]

  x = torch.tensor(x)
  y = torch.tensor(y)

  return x,y


n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])





class Linear:
  def __init__(self, input_f, out_f, bias=True):
    self.weight = torch.randn(input_f, out_f) * input_f**-0.5
    self.weight.requires_grad = True
    self.bias = torch.zeros(out_f) if bias else None
    if self.bias is not None:
      self.bias.requires_grad = True

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  def __init__(self, neurons, eps=1e-05, momentum=0.1):
    self.eps = eps
    self.momentum=momentum
    self.track_running_training=True

    self.bngain = torch.ones(neurons, requires_grad=True)
    self.bnbias = torch.zeros(neurons, requires_grad=True)
    self.bnmean_running = torch.zeros(neurons)
    self.bnvar_running = torch.ones(neurons)

  def __call__(self, x):

    if self.track_running_training:
      bnmean = x.mean(0, keepdim=True)
      bnvar = x.var(0, keepdim=True)
    else:
      bnmean = self.bnmean_running
      bnvar = self.bnvar_running

    xhat = (x - bnmean)/ torch.sqrt(bnvar + self.eps)
    self.out = self.bngain * xhat + self.bnbias

    if self.track_running_training:
      with torch.no_grad():
        self.bnmean_running = (1-self.momentum)*self.bnmean_running + self.momentum * bnmean
        self.bnvar_running = (1-self.momentum)*self.bnvar_running + self.momentum * bnvar

    return self.out

  def parameters(self):
    return [self.bngain, self.bnbias]


class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

class Embedding:
  def __init__(self, vocab_size, n_embd):
    self.weight = torch.randn(vocab_size, n_embd)
    self.weight.requires_grad = True

  def __call__(self, x):              #it will take data as input and not indices when called via model/sequential class
  #ix = torch.randint(1, Xtr.shape[0], (4,))
  #Xb, Yb = Xtr[ix], Ytr[ix]
  #logits = model(Xb)
    self.out = self.weight[x]
    return self.out

  def parameters(self):
    return [self.weight]

class Flatten:
  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out
  def parameters(self):
    return []

class Sequential:
  def __init__(self, layers):
    self.layers = layers
  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
      self.out = x
    return self.out
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

layers = [ Embedding(vocab_size, n_embd),
           Flatten(),
           Linear(n_embd*block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
           Linear(n_hidden, vocab_size),
          ]
with torch.no_grad():
  layers[-1].weight *= 0.1
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.67

  parameters = [p for layer in layers for p in layer.parameters()]
  for para in parameters:
    para.requires_grad=True

module = Sequential(layers) #created Sequential object

## improvement of lecture -6
layers = [ Embedding(vocab_size, n_embd),
           Flatten(),
           Linear(n_embd*block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
           Linear(n_hidden, vocab_size),
          ]
with torch.no_grad():
  layers[-1].weight *= 0.1
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.67

  parameters = [p for layer in layers for p in layer.parameters()]
  for para in parameters:
    para.requires_grad=True


#modulerize approach

model = Sequential([
           Embedding(vocab_size, n_embd),
           Flatten(),
           Linear(n_embd*block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
           Linear(n_hidden, vocab_size),
])

with torch.no_grad():
  model.layers[-1].weight *= 0.1
  for layer in model.layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.67

  parameters = [p for layer in model.layers for p in layer.parameters()]
  for para in parameters:
    para.requires_grad=True

#improvement
max_steps = 1000
lossi = []
ud = []
for i in range(max_steps):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  x = Xtr[ix]
  #data = C[Xtr[ix]]
  #x = data.view(data.shape[0], -1)


  for layer in layers:
    x = layer(x)

  loss = torch.nn.functional.cross_entropy(x, Ytr[ix])

  for p in parameters:
    p.grad = None

  loss.backward()
  lr = 0.1 if i<100000 else 0.01
  for p in parameters:
    p.data = p.data - lr*p.grad

  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())


  #improvement  - module - approach
max_steps = 1000
lossi = []
ud = []
for i in range(max_steps):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  x = Xtr[ix]
  #data = C[Xtr[ix]]
  #x = data.view(data.shape[0], -1)

  logits = model(x)
  loss = torch.nn.functional.cross_entropy(logits, Ytr[ix])

  for p in model.parameters():
    p.grad = None

  loss.backward()
  lr = 0.1 if i<100000 else 0.01
  for p in model.parameters():
    p.data = p.data - lr*p.grad

  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
