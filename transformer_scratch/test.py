import torch
from model import Transformer
from config import *

model = Transformer(VOCAB_SIZE, MAX_SEQ_LEN, EMBED_DIM, NUM_HEADS, NUM_LAYERS)

input_ids = torch.tensor([1045, 2293, 2634, 1012])

output = model(input_ids.unsqueeze(0))

print("Input Shape (with batch):", input_ids.unsqueeze(0).shape)
print("Output Shape:", output.shape)
