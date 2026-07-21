import torch

from config import *
from model import MiniGPT
from tokenizer import Tokenizer

# read training text
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)

# model
model = MiniGPT(
    vocab_size=len(tokenizer),
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

# load trained weights
CHECKPOINT_PATH = f"checkpoints/model_epoch_{EPOCHS}.pth"

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 300

# encode prompt
context = torch.tensor(
    [tokenizer.encode(PROMPT)],
    dtype=torch.long,
    device=DEVICE,
)

# generate
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)

# decode
output = tokenizer.decode(generated[0].tolist())

print(output)