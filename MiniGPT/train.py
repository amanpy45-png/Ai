import os
import torch
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataset import GPTDataset
from model import MiniGPT
from config import *


with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)

tokens = tokenizer.encode(text)

dataset = GPTDataset(tokens, CONTEXT_LENGTH)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = MiniGPT(
    vocab_size=len(tokenizer),
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

model = model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)
os.makedirs("checkpoints", exist_ok=True)

torch.manual_seed(42)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(42)

for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0.0

    for x, y in dataloader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, loss = model(x, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(
        model.state_dict(),
        f"checkpoints/model_epoch_{epoch+1}.pth"
    )