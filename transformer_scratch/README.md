GPT-Style Transformer
A custom, modular PyTorch implementation of a Decoder-only Transformer.

Architecture
Token & Positional Embeddings: Maps indices to vector space with sequence position information.

Multi-Head Attention: Implements causal (masked) self-attention to ensure autoregressive generation.

Transformer Blocks: Contains multi-head attention and feed-forward networks with residual connections and layer normalization.

Final Projection: Maps hidden states to vocabulary logits for next-token prediction.

File Map
model.py: Core architecture definitions (TokenEmbedding, PostionalEmbedding, SelfAttention, MultiHeadAttention, FeedForward, TransformerBlock, Transformer).

config.py: Centralized hyperparameter configuration (VOCAB_SIZE, EMBED_DIM, etc.).

test.py: Validation script to verify tensor shapes and ensure the causal mask is working.

Setup
Define your model architecture constants in config.py.


Run the validation script:

Bash
python test.py
Key Components
The Transformer class orchestrates the forward pass:

Input: (batch, seq_len)

Embedding: Sum of token and positional embeddings (batch, seq_len, embed_dim).

Processing: Passes through N TransformerBlock layers.

Output: Logits for vocabulary (batch, seq_len, vocab_size).