# 🤖 AI Text Summarizer

Summarize long articles in seconds using **Hugging Face Transformers** (DistilBART) and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)

---

##  Features

- **Paste-and-summarize** — drop any article into the text box and get a summary instantly.
- **Live text stats** — character and word count update as you type.
- **Adjustable summary length** — Short / Medium / Long presets control the model's `min_length` / `max_length`.
- **Download summary** — export the generated summary as a `.txt` file.
- **Reduction metrics** — shows original word count, summary word count, and the percentage reduction.
- **Cached model loading** — uses `@st.cache_resource` so the model loads only once per session.

---

##  Model

- **Model:** [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- **Task:** Abstractive summarization
- **Framework:** 🤗 Transformers `pipeline("summarization")`

---

##  Project Structure

```
├── app.py            # Streamlit UI (main entry point)
├── summarizer.py      # Model loading + summarization logic
├── requirements.txt   # Python dependencies
└── README.md
```

---

##  Installation

```bash
git clone https://github.com/amanpy45-png/Ai.git
cd ai-text-summarizer

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
transformers
torch
```

---

##  Usage

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

1. Paste your article into the **Input Text** box.
2. Choose a summary length: **Short**, **Medium**, or **Long**.
3. Click **Generate Summary**.
4. Review the output, check the reduction stats, and optionally **Download Summary**.
5. Click **Clear** to reset and start over.

---

##  Configuration

Summary length presets map to the model's `min_length` / `max_length` parameters in `app.py`:

| Option | max_length | min_length |
|--------|-----------|-----------|
| Short  | 40        | 15        |
| Medium | 80        | 30        |
| Long   | 120       | 50        |

---

##  Author

**Aman Negi**

- GitHub: https://github.com/amanpy45-png/
- LinkedIn: https://www.linkedin.com/in/amanpy54/

---

⭐ If you found this project useful, consider giving it a star!
