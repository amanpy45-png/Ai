import streamlit as st
from summarizer import summarize

st.set_page_config(page_title="AI Text Summarizer", page_icon="🤖", layout="wide")

st.title("🤖 AI Text Summarizer")
st.markdown("Summarize long articles in seconds using **Hugging Face Transformers**.")

# Initialize session state for persistence
if "summary" not in st.session_state:
    st.session_state.summary = None

with st.expander("How it works"):
    st.write("This app uses **DistilBART**, a transformer model fine-tuned on the CNN/DailyMail dataset to generate concise summaries.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Text")
    text = st.text_area("Paste your article here", height=350, placeholder="Paste your article here...")
    
    # Calculate stats live
    words = len(text.split())
    st.caption(f"**Characters:** {len(text)} | **Words:** {words}")

    option = st.radio("Summary Length", ["Short", "Medium", "Long"], horizontal=True)
    
    # Logic mapping
    config = {"Short": (40, 15), "Medium": (80, 30), "Long": (120, 50)}
    max_len, min_len = config[option]

    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("Generate Summary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating..."):
                st.session_state.summary = summarize(text, max_len, min_len)
    
    if c_btn2.button("Clear", use_container_width=True):
        st.session_state.summary = None
        st.rerun()

with col2:
    st.subheader("Generated Summary")
    
    if st.session_state.summary:
        st.success("Summary Generated!")
        with st.container(border=True):
            st.write(st.session_state.summary)
        
        st.download_button("Download Summary", st.session_state.summary, file_name="summary.txt", use_container_width=True)
        
        st.divider()
        
        # Calculate reduction stats
        sum_words = len(st.session_state.summary.split())
        red = ((words - sum_words) / words * 100) if words > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Original", words)
        m2.metric("Summary", sum_words)
        m3.metric("Reduction", f"{red:.1f}%")
        
        st.info("Model: DistilBART (sshleifer/distilbart-cnn-12-6)")
    else:
        st.info("Your summary will appear here after you click 'Generate'.")

st.divider()
st.caption("<center>Built by Aman using Hugging Face Transformers & Streamlit</center>", unsafe_allow_html=True)