# PDF Semantic Search Engine using BERT

A semantic search engine built using **BERT**, **PyTorch**, **Hugging Face Transformers**, **PyMuPDF**, and **Streamlit**. The application allows users to upload any PDF and retrieve the most semantically relevant content based on the meaning of their query rather than exact keyword matching.

---

## Features

- Upload any PDF through the Streamlit interface
- Generate contextual embeddings using BERT (`bert-base-uncased`)
- Masked Mean Pooling for sentence embedding generation
- Semantic search using Cosine Similarity
- Top-K ranked search results
- Cached document embeddings for faster searches
- Modular and reusable project architecture

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- PyMuPDF (fitz)

---

## Project Structure

```
PDF-Semantic-Search/
│
├── app.py                 # Streamlit Application
├── embedding.py           # BERT embedding generation
├── search.py              # Semantic search logic
├── pdf_loader.py          # PDF loading and text extraction
├── requirements.txt
├── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/pdf-semantic-search.git

cd pdf-semantic-search
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

### Step 1 — Upload a PDF

The user uploads any PDF document through the Streamlit interface.

↓

### Step 2 — Extract Text

The PDF is processed using **PyMuPDF**, and text is extracted from each page.

↓

### Step 3 — Generate Embeddings

Each extracted text segment is converted into a contextual embedding using **BERT**.

↓

### Step 4 — User Query

The user enters a search query.

↓

### Step 5 — Query Embedding

The query is converted into a BERT embedding using the same pipeline.

↓

### Step 6 — Similarity Search

Cosine similarity is computed between the query embedding and every document embedding.

↓

### Step 7 — Ranking

The application returns the **Top-K** most semantically similar results.

---

## Model Details

| Component         |  Value              |
|-------------------|---------------------|
| Model             | bert-base-uncased   |
| Embedding Size    | 768                 |
| Pooling Strategy  | Masked Mean Pooling |
| Similarity Metric | Cosine Similarity   |

---

##  Example

### Query

```
Machine Learning
```

### Output

```
🥇 Machine Learning is a branch of Artificial Intelligence.

🥈 Deep Learning is a subset of Machine Learning.

🥉 Neural Networks are inspired by the human brain.
```

---

##  Key Concepts Implemented

- Transformer Architecture
- BERT
- Contextual Embeddings
- Attention Masks
- Masked Mean Pooling
- Cosine Similarity
- Semantic Search
- PDF Text Extraction
- Streamlit UI Development

---

##  Current Limitations

- Basic sentence splitting is used after PDF text extraction.
- Performance may decrease for very large PDFs.
- Uses vanilla BERT with mean pooling instead of Sentence-BERT (SBERT).

---

##  Future Improvements

- Better sentence segmentation.
- Support multiple PDFs simultaneously.
- Add semantic chunking for long documents.
- Replace BERT with Sentence-BERT (SBERT).
- Integrate FAISS for faster similarity search.

---

##  Author

**Aman Negi**

- GitHub: https://github.com/amanpy45-png
- LinkedIn: https://www.linkedin.com/in/amanpy54/

---

⭐ If you found this project useful, consider giving it a star!
