import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self,x):
        return self.embedding(x)
    

class PostionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(max_seq_len, embed_dim)


    def forward(self, x):
        seq_len = x.size(0)
        position_ids = torch.arange(seq_len)

        return self.embedding(position_ids)
    


