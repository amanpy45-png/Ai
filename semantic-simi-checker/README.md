# Semantic Search Engine using BERT

A semantic search engine built using **BERT**, **PyTorch**, **Hugging Face Transformers**, and **Streamlit**. Instead of matching exact keywords, the application retrieves documents based on their **semantic meaning** using contextual embeddings and cosine similarity.

---

## Features

- Semantic search using BERT embeddings
- Context-aware sentence representations
- Masked Mean Pooling for sentence embedding generation
- Cosine Similarity-based ranking
- Load documents dynamically from `documents.txt`
- Precomputed document embeddings for faster search
- Interactive Streamlit interface
- Top-K ranked search results

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit

---

## Project Structure

```
Semantic-Search-Engine/
│
├── app.py                 # Streamlit UI
├── embedding.py           # BERT embedding generation
├── search.py              # Semantic search functions
├── utils.py               # Load documents
├── documents.txt          # Search corpus
├── requirements.txt
├── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/semantic-search-engine.git

cd semantic-search-engine
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## How It Works

### Step 1 — Load Documents

The application loads all documents from `documents.txt`.

↓

### Step 2 — Generate Embeddings

Each document is converted into a contextual embedding using **BERT**.

↓

### Step 3 — User Query

The user enters a search query.

↓

### Step 4 — Query Embedding

The query is converted into a BERT embedding.

↓

### Step 5 — Similarity Search

Cosine similarity is computed between the query embedding and every document embedding.

↓

### Step 6 — Ranking

Documents are sorted by similarity score and the **Top-K** results are displayed.

---

## Model Details

- **Model:** `bert-base-uncased`
- **Embedding Size:** 768
- **Pooling Strategy:** Masked Mean Pooling
- **Similarity Metric:** Cosine Similarity

---

## Example

### Query

```
I enjoy coding in Python.
```

### Output

```
Python is a programming language. (91.2%)

I love Machine Learning. (72.3%)

Artificial Intelligence is transforming the world. (58.1%)
```

---

##  Key Concepts Used

- Transformer Architecture
- BERT
- Tokenization
- Attention Masks
- Contextual Word Embeddings
- Masked Mean Pooling
- Cosine Similarity
- Semantic Search

---

## Future Improvements

- Replace BERT Mean Pooling with **Sentence-BERT (SBERT)** for better sentence embeddings.
- Search over PDFs and text files.
- Integrate FAISS for fast vector search.
- Support larger document collections.
- Build a Retrieval-Augmented Generation (RAG) pipeline.

---

##  Author

**Aman Negi**

- GitHub: https://github.com/amanpy45-png/
- LinkedIn: https://www.linkedin.com/in/amanpy54/

---

## ⭐ If you found this project useful, consider giving it a star!
