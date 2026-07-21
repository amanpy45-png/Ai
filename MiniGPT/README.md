# 🧠 MiniGPT: Decoder-Only Transformer from Scratch

A lightweight implementation of a **GPT-style Decoder-Only Transformer** built completely from scratch using **PyTorch**. This project was developed to understand the internal architecture of modern Large Language Models (LLMs) by implementing every core component instead of relying on high-level libraries.

> **Note:** This project is intended for educational purposes. It demonstrates how GPT models work internally and is **not** meant to compete with pretrained models like GPT-2, LLaMA, or ChatGPT.

---

## 🚀 Features

- Character-level tokenizer
- Custom PyTorch Dataset & DataLoader
- Token & Positional Embeddings
- Masked Multi-Head Self-Attention
- Feed Forward Network (FFN)
- Residual Connections
- Layer Normalization (Pre-LN)
- Decoder-Only Transformer Architecture
- Autoregressive Text Generation
- Model Checkpoint Saving
- Training & Inference Scripts

---

## 📁 Project Structure

```
MiniGPT/
│
├── checkpoints/
│   └── model_epoch_x.pth
│
├── data/
│   └── input.txt
│
├── config.py
├── dataset.py
├── tokenizer.py
├── model.py
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

---

## 🏗️ Model Architecture

```
Input Text
      │
      ▼
Character Tokenizer
      │
      ▼
Token Embedding
      +
Position Embedding
      │
      ▼
N × Transformer Blocks
      │
      ├── LayerNorm
      ├── Masked Multi-Head Attention
      ├── Residual Connection
      ├── LayerNorm
      ├── Feed Forward Network
      └── Residual Connection
      │
      ▼
LayerNorm
      │
      ▼
Linear Language Modeling Head
      │
      ▼
Next Token Prediction
```

---

## 🧩 Components Implemented

### Tokenizer
- Character-level vocabulary
- String ↔ Token conversion

### Dataset
Creates training pairs:

```
Input : ABCDE
Target: BCDEF
```

---

### Attention

Implements **Scaled Dot-Product Attention** with a **causal mask**.

```
Q × Kᵀ
──────────
 √d
```

followed by

- Causal Mask
- Softmax
- Dropout
- Attention × Value

---

### Multi-Head Attention

Multiple attention heads operate in parallel.

```
Head1
Head2
Head3
Head4
   │
Concatenate
   │
Linear Projection
```

---

### Feed Forward Network

```
Linear
   ↓
GELU
   ↓
Linear
   ↓
Dropout
```

---

### Transformer Block

```
Input
 │
 ▼
LayerNorm
 │
 ▼
Masked Multi-Head Attention
 │
 ▼
Residual Add
 │
 ▼
LayerNorm
 │
 ▼
Feed Forward Network
 │
 ▼
Residual Add
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/MiniGPT.git
cd MiniGPT
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Training

Place your training corpus inside

```
data/input.txt
```

Run

```bash
python train.py
```

Training checkpoints are saved automatically inside

```
checkpoints/
```

---

## ✨ Text Generation

After training,

run

```bash
python generate.py
```

Example prompt

```python
PROMPT = "To be"
```

The model generates text autoregressively one token at a time.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy

---

## 📖 What I Learned

This project helped me understand the internal working of GPT-style language models by implementing every major building block from scratch, including:

- Embeddings
- Positional Encoding
- Self-Attention
- Multi-Head Attention
- Feed Forward Networks
- Layer Normalization
- Residual Connections
- Autoregressive Generation
- Cross-Entropy Training
- Transformer Architecture

---

## ⚠️ Limitations

- Character-level tokenizer
- Small model size
- Trained from scratch on a limited dataset
- Not intended to match the capabilities of pretrained LLMs such as GPT-2 or ChatGPT

---

## 🔮 Future Improvements

- Byte Pair Encoding (BPE) Tokenizer
- Top-k & Top-p Sampling
- Temperature Sampling
- Learning Rate Scheduler
- Mixed Precision Training
- Flash Attention
- KV Cache
- Larger Training Corpus
- GPT-2 Style Architecture

---

## 📜 License

This project is released under the MIT License.

---

## 👨‍💻 Author

**Aman Negi**

B.Tech CSE | AI & Data Science Enthusiast

LinkedIn: https://www.linkedin.com/in/amanpy54/

GitHub: https://github.com/<your-github-username>