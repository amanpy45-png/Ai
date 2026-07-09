import torch
from model import TokenEmbedding, PostionalEmbedding

vocab_size = 30000
embed_dim = 512
max_seq_len =100

embedding = TokenEmbedding(vocab_size, embed_dim)

input_ids = torch.tensor([1045, 2293, 2634, 1012])

token_out = embedding(input_ids)

# print(input_ids)
# print()

# print(output)
# print()

# print(output.shape)

# print(embedding.embedding.weight[1045])


x = torch.tensor([1045, 2293, 2634, 1012])
seq_len = x.size(0)

position_ids = torch.arange(seq_len)

# print(position_ids)

pos_embedding = PostionalEmbedding(max_seq_len, embed_dim)
pos_out= pos_embedding(input_ids)
# print(output.shape)

final_embedding = token_out + pos_out
print(final_embedding.shape)