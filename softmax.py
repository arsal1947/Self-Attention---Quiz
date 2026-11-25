import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

np.random.seed(0)
B, T, C = 1, 4, 8   # --> taking the example data
head_size = 4

x  = np.random.randn(B, T, C)           
Wq = np.random.randn(C, head_size)     # --> initializing with random values  
Wk = np.random.randn(C, head_size)     # --> initializing with random values
Wv = np.random.randn(C, head_size)     # --> initializing with random values 

q = x @ Wq    # -> making of query (means what it want)                         
k = x @ Wk    # -> making of key   (mean what is have's)                          
v = x @ Wv    # -> making of value (mean what is looking for)                         

scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_size)   # -> multiplying the query and key value and the transpose in order to get(B,T,C)

mask = np.tril(np.ones((T, T)))
scores = np.where(mask == 0, -np.inf, scores)

wei = softmax(scores)    

out = wei @ v  


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd , head_size , dropout)
        self.value = nn.Linear(n_embd , head_size , bias=False)
        self.query = nn.Linear(n_embd , head_size , bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = softmax(wei)
        wei = self.dropout(wei)
        v = self.value(x)                                      
        return wei @ v                                    
        