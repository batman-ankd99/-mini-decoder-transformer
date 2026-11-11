import torch
from dataclasses import dataclass #python generates __init__() and we dont need to write self.key=value for every data value

#Hyperparameters
@dataclass
class Config:
    # Training configuration
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4

    # Model configuration
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
