# Understanding Transformer Architecture

This document provides a concise overview of the key components and mechanisms that constitute the Transformer architecture, the foundation for modern Large Language Models (LLMs).

## 1. Why Transformers were invented
Transformers were introduced to overcome the limitations of Recurrent Neural Networks (RNNs) and LSTMs. RNNs process data sequentially, which prevents parallelization and makes it difficult to capture long-range dependencies in long sequences. Transformers enable parallel processing and utilize attention mechanisms to model dependencies regardless of their distance in the sequence.

## 2. Self-Attention
Self-attention allows the model to weigh the importance of different words in a sequence relative to a specific word. It helps the model understand the context and relationships between words, even if they are far apart in the sentence.

## 3. Query, Key, and Value (Q, K, V)
These are the three vectors derived from input embeddings that drive the attention mechanism:
- **Query (Q):** What I am looking for.
- **Key (K):** What I contain / What I can be matched with.
- **Value (V):** What information I actually contain.
The attention score is calculated by the compatibility of the Query with the Keys, used to weight the Values.

## 4. Softmax
The Softmax function is applied to the scaled dot-product of Query and Key vectors. It converts raw scores into probabilities (values between 0 and 1 that sum to 1), effectively determining how much "attention" should be paid to each part of the sequence.

## 5. Multi-Head Attention
Instead of performing self-attention once, Multi-Head Attention runs multiple attention mechanisms (heads) in parallel. This allows the model to simultaneously attend to information from different representation subspaces (e.g., one head might focus on grammatical relationships, while another focuses on semantic meaning).

## 6. Positional Encoding
Since Transformers process inputs in parallel, they have no inherent sense of order. Positional Encoding injects information about the relative or absolute position of tokens in the sequence into the input embeddings, ensuring the model respects the order of words.

## 7. Residual Connections (Add & Norm)
Residual (or skip) connections add the input of a layer to its output before applying normalization. This helps mitigate the vanishing gradient problem in deep networks, allowing for better gradient flow during training.

## 8. Layer Normalization
Layer Normalization stabilizes the hidden state dynamics by normalizing the inputs across the features for each token independently. This leads to faster convergence and more stable training.

## 9. Feed-Forward Network (FFN)
After the attention mechanism, each position's representation is passed through an identical, independent feed-forward neural network. This allows for additional non-linear transformations and feature processing.

## 10. Encoder vs. Decoder
- **Encoder:** Reads the entire input sequence and produces a rich, context-aware representation (e.g., BERT).
- **Decoder:** Generates an output sequence one token at a time, looking at both the encoded input and previously generated tokens (e.g., standard transformer).

## 11. Why GPT is Decoder-only
GPT (Generative Pre-trained Transformer) is a decoder-only model because its primary objective is **causal language modeling** (predicting the next token). Since it doesn't need to "encode" an input sequence in the traditional translation sense, it removes the encoder stack, allowing it to focus entirely on autoregressive generation.
