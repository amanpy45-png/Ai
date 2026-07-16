import streamlit as st

from embedding import get_embedding
from search import semantic_search
from pdf_loader import load_pdf
from chunking import chunk_text

# Page Configuration

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="centered"
)

st.title("Semantic Search Engine")
st.markdown(
    "Upload a PDF and search it using **BERT semantic embeddings**."
)

# Build Search Index
@st.cache_resource
def build_search_index(uploaded_file):

    # Extract raw text
    text = load_pdf(uploaded_file)

    # Convert text into overlapping chunks
    documents = chunk_text(
        text,
        chunk_size=3,
        overlap=1
    )

    # Generate embeddings
    document_embeddings = [
        get_embedding(doc)
        for doc in documents
    ]

    return documents, document_embeddings

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# Main Application
if uploaded_file is not None:

    documents, document_embeddings = build_search_index(uploaded_file)

    query = st.text_input("Enter your query")

    k = st.slider(
        "Top Results",
        min_value=1,
        max_value=len(documents),
        value=min(3, len(documents))
    )

    if st.button("Search"):

        if not query.strip():
            st.warning("Please enter a query.")

        else:

            results = semantic_search(
                query,
                documents,
                document_embeddings,
                k
            )

            st.subheader("📋 Top Matches")

            medals = ["🥇", "🥈", "🥉"]

            for i, (score, doc) in enumerate(results):

                emoji = medals[i] if i < 3 else "📄"

                with st.container():

                    st.markdown(f"### {emoji} Match #{i + 1}")

                    st.write(doc)

                    st.progress(int(score * 100))

                    st.caption(
                        f"Similarity: {score * 100:.2f}%"
                    )

                    st.divider()

else:

    st.info("Upload a PDF to begin searching.")
