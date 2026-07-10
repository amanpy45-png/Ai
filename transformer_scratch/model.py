import torch
import torch.nn as nn
import math

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
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
    
        self.Wq = nn.Linear(embed_dim, head_dim) 
        self.Wk = nn.Linear(embed_dim, head_dim)
        self.Wv = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x) 

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output
        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttention(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.W_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))

        multi_head_output = torch.cat(head_outputs, dim = -1)
        final_output = self.W_out(multi_head_output)
        return final_output
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor =4):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
    

    
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers):
        super().__init__()
        
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PostionalEmbedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(x)
    
        for layer in self.layers:
            x = layer(x)
            
        return self.fc_out(x)











