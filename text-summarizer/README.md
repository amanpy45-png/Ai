# 🤖 AI Text Summarizer

Summarize long articles in seconds using **Hugging Face Transformers** (DistilBART) and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)

---

## ✨ Features

- **Paste-and-summarize** — drop any article into the text box and get a summary instantly.
- **Live text stats** — character and word count update as you type.
- **Adjustable summary length** — Short / Medium / Long presets control the model's `min_length` / `max_length`.
- **Download summary** — export the generated summary as a `.txt` file.
- **Reduction metrics** — shows original word count, summary word count, and the percentage reduction.
- **Cached model loading** — uses `@st.cache_resource` so the model loads only once per session.

---

## 🧠 Model

- **Model:** [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- **Task:** Abstractive summarization
- **Framework:** 🤗 Transformers `pipeline("summarization")`

---

## 📁 Project Structure

```
├── app.py            # Streamlit UI (main entry point)
├── summarizer.py      # Model loading + summarization logic
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
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

## ▶️ Usage

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

## 🔧 Configuration

Summary length presets map to the model's `min_length` / `max_length` parameters in `app.py`:

| Option | max_length | min_length |
|--------|-----------|-----------|
| Short  | 40        | 15        |
| Medium | 80        | 30        |
| Long   | 120       | 50        |

---

## 👀 Notes from the current build (based on the screenshot)

### ✅ Pros
- **Clean, modern UI** — dark theme, clear two-column layout separating input from output makes the tool easy to scan at a glance.
- **Good feedback loop** — the green "Summary Generated!" banner, live word/character counter, and reduction percentage give the user immediate confirmation the app worked.
- **Transparent about the model** — showing "Model: DistilBART (sshleifer/distilbart-cnn-12-6)" is a nice touch for trust/credibility, especially for a portfolio project.
- **Practical extras** — download button and a collapsible "How it works" section add polish beyond a bare-bones demo.
- **Sensible state handling** — using `st.session_state` for the summary means it persists across reruns (e.g. after pressing Download) instead of disappearing.

### ⚠️ Cons / things to improve
- **Weak input for this demo** — the sample article is only ~134 words. DistilBART-CNN was fine-tuned on much longer news articles, so feeding it a short paragraph doesn't showcase real summarization — the "summary" is close to a lightly-trimmed copy of the original rather than a genuine abstraction. Worth testing with a 400+ word article to see the model shine.
- **Punctuation spacing artifact** — the output shows stray spaces before periods (e.g. *"workflows ."*, *"processes ."*). This is a common HF summarization output quirk and should be cleaned up with a small post-processing step, e.g.:
  ```python
  import re
  summary = re.sub(r'\s+([.,!?])', r'\1', summary)
  ```
- **"Long" setting barely reduces the text** — a 48.5% reduction on an already-short 134-word input isn't a very compelling result; users may expect "Long" to mean a *longer, more detailed* summary rather than just "less aggressive trimming."
- **No handling for very long articles** — DistilBART has a token limit (~1024 tokens). Very long pasted text will silently get truncated (you do pass `truncation=True`, which avoids a crash, but the user isn't told any content was cut off).
- **No loading/latency indication beyond the spinner** — first-time model load can take a while (downloading weights); a message like "Loading model, this may take a minute on first run…" would set expectations.
- **No error handling for model/inference failures** — if the pipeline throws (e.g. OOM on CPU for a long doc), the app will likely crash rather than showing a friendly error.

---

## 🚀 Possible Next Steps
- Post-process output to fix punctuation spacing.
- Warn users when input exceeds the model's max token limit.
- Add a word-count-based dynamic `max_length`/`min_length` instead of fixed presets, so short inputs aren't summarized to near-identical length.
- Add support for larger/better models (e.g. `facebook/bart-large-cnn`) as an optional toggle.
- Wrap `summarize()` in a `try/except` with a user-facing error message.

---

## 🙌 Credits

Built by **Aman** using Hugging Face Transformers & Streamlit.