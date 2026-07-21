import torch

CONTEXT_LENGTH = 128
BATCH_SIZE = 32


EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.2


LEARNING_RATE = 3e-4
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"