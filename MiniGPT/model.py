import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, embed_dim, head_size, context_length, dropout):
        super().__init__()

        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):
        B, T, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        out = attn @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, context_length, dropout):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        head_size = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [
                Head(embed_dim, head_size, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.proj(out)

        out = self.dropout(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, embed_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, context_length, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, context_length, dropout
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    context_length,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        token = self.token_embedding(x)

        pos = self.position_embedding(torch.arange(T, device=x.device))

        x = token + pos

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length :]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        if was_training:
            self.train()

        return idx