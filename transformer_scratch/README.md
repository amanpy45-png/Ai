# Transformer Language Model

A GPT-style decoder-only Transformer for autoregressive language modeling.

## Forward Pass Summary

| Step | Description | Output Shape |
|------|-------------|--------------|
| 1. Input | Token IDs | `(batch_size, seq_len)` |
| 2. Embedding | Token embeddings + Positional embeddings | `(batch_size, seq_len, embed_dim)` |
| 3. Transformer Blocks | N blocks, each with causal self-attention and a feed-forward network (FFN) | `(batch_size, seq_len, embed_dim)` |
| 4. Output | Projection to vocabulary logits | `(batch_size, seq_len, vocab_size)` |

## Architecture Details

```
Input (token IDs)
            │
            ▼
Token Embedding + Positional Embedding
            │
            ▼
┌─────────────────────────┐
│  Transformer Block × N  │
│ ┌─────────────────────┐ │
│ │Causal Self-Attention│ │
│ └─────────────────────┘ │
│ ┌─────────────────────┐ │
│ │Feed-Forward Network │ |
│ └─────────────────────┘ │
└─────────────────────────┘
           │
           ▼
Output Logits (vocab_size)
```

- **Causal self-attention** ensures each token can only attend to itself and previous tokens, preserving the autoregressive property.
- **Feed-forward network (FFN)** applies a position-wise non-linear transformation after attention in each block.
- **N** denotes the total number of stacked Transformer blocks (a configurable hyperparameter).

## Input / Output

- **Input:** `torch.LongTensor` of shape `(batch_size, seq_len)` containing token indices.
- **Output:** Logits of shape `(batch_size, seq_len, vocab_size)`, representing the unnormalized probability distribution over the vocabulary for each position.

## License

MIT LICENSE
