import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

def summarize(text, max_len, min_len):
    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True
    )
    return result[0]["summary_text"]