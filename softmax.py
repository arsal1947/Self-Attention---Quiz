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

mask = np.tril(np.ones((T, T)))  # -> we are masking it so that we don't want to show the tokens the upcoming token or future tokens
scores = np.where(mask == 0, -np.inf, scores)  # -> here we are converting the 1 and -inf matrix 
 
wei = softmax(scores)     # -> applying softmax to the scores because we have to convert into probabilites

out = wei @ v  # ->  output of the wei with the value to get the output matrix


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd , head_size , bias=False)  #making of the linear layers and with the given no. of embeddings and head_size
        self.value = nn.Linear(n_embd , head_size , bias=False) # same making of the linear layers and with the given no. of embeddings and head_size
        self.query = nn.Linear(n_embd , head_size , bias=False) # same making of the linear layers and with the given no. of embeddings and head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # -> as it use for causal mask for self-attention and stores it
        self.dropout = nn.Dropout(dropout) # -> and dropout means that during training we have to drop out some layers or weights so it would not learn the data

    def forward(self , x):
        B,T,C = x.shape  # -> getting the batches , sequence , channel
        q = self.query(x)  #- > making of the query matrix 
        k = self.key(x)  # -> making of the key matrix
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  #-> multiplying the both ones
        wei = softmax(wei)
        wei = self.dropout(wei)
        v = self.value(x)                                      
        return wei @ v                                    
        
