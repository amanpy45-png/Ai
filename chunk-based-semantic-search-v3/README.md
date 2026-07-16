# 🔍 Chunk-Based PDF Semantic Search using BERT

A semantic search engine built using **BERT**, **PyTorch**, **Hugging Face Transformers**, **PyMuPDF**, **NLTK**, and **Streamlit**. Unlike traditional keyword search, this application retrieves text based on semantic meaning and improves retrieval quality using **overlapping text chunking**.

---

## 📸 Demo

> Add a screenshot of the application here.

![Application Demo](screenshot.png)

---

## 🚀 Features

- 📄 Upload any PDF
- 🧠 Generate contextual embeddings using BERT (`bert-base-uncased`)
- ✂️ Intelligent text chunking with configurable chunk size and overlap
- 📊 Masked Mean Pooling for sentence embeddings
- 🔍 Semantic search using Cosine Similarity
- 🏆 Top-K ranked search results
- ⚡ Cached document embeddings for faster searches
- 🧩 Modular and reusable architecture

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- PyMuPDF
- NLTK

---

## 📂 Project Structure

```
Chunk-Based-PDF-Semantic-Search/
│
├── app.py                 # Streamlit application
├── embedding.py           # BERT embedding generation
├── search.py              # Semantic search logic
├── pdf_loader.py          # PDF text extraction
├── chunking.py            # Sentence extraction & overlapping chunking
├── requirements.txt
├── README.md
└── screenshot.png
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/chunk-based-pdf-semantic-search.git

cd chunk-based-pdf-semantic-search
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK tokenizer (first time only)

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

### Run the application

```bash
streamlit run app.py
```

---

## 📖 How It Works

### Step 1 — Upload PDF

The user uploads a PDF through the Streamlit interface.

↓

### Step 2 — Extract Text

Raw text is extracted using **PyMuPDF**.

↓

### Step 3 — Sentence Segmentation

The extracted text is divided into sentences using **NLTK's sentence tokenizer**.

↓

### Step 4 — Chunking

Sentences are grouped into **overlapping chunks** to preserve context across chunk boundaries.

↓

### Step 5 — Generate Embeddings

Each chunk is converted into a contextual embedding using **BERT**.

↓

### Step 6 — User Query

The query is embedded using the same BERT pipeline.

↓

### Step 7 — Similarity Search

Cosine similarity is computed between the query embedding and every chunk embedding.

↓

### Step 8 — Ranking

The Top-K most semantically relevant chunks are returned.

---

## 🧠 Chunking Strategy

Example with:

- Chunk Size = **3**
- Overlap = **1**

Input:

```
S1
S2
S3
S4
S5
S6
S7
```

Generated Chunks:

```
Chunk 1
S1 S2 S3

Chunk 2
S3 S4 S5

Chunk 3
S5 S6 S7
```

This overlapping strategy preserves context between neighboring chunks and improves retrieval quality.

---

## 🧠 Model Details

| Component | Value |
|-----------|-------|
| Model | bert-base-uncased |
| Embedding Size | 768 |
| Pooling Strategy | Masked Mean Pooling |
| Similarity Metric | Cosine Similarity |
| Chunking | Overlapping Fixed-Size Chunks |

---

## 💡 Key Concepts Implemented

- Transformer Architecture
- BERT
- Contextual Embeddings
- Attention Masks
- Masked Mean Pooling
- Cosine Similarity
- Semantic Search
- PDF Text Extraction
- Sentence Tokenization
- Overlapping Chunking
- Streamlit UI Development

---

## 📌 Current Limitations

- Fixed-size chunking may split related ideas.
- Large PDFs require embedding every chunk, increasing processing time.
- Uses vanilla BERT instead of Sentence-BERT (SBERT).
- No vector database (FAISS/ChromaDB) integration yet.

---

## 🚀 Future Improvements

- Semantic chunking
- Sentence-BERT (SBERT)
- FAISS vector search
- Multi-PDF search
- Metadata-aware retrieval
- Hybrid keyword + semantic search

---

## 👨‍💻 Author

**Aman Negi**

- GitHub: https://github.com/your-amanpy45-png
- LinkedIn: https://www.linkedin.com/in/amanpy54/

---

⭐ If you found this project useful, consider giving it a star!