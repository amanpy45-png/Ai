import streamlit as st

from embedding import get_embedding
from search import semantic_search
from pdf_loader import load_pdf


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

# Cache Search Engine
@st.cache_resource
def prepare_search_engine(uploaded_file):
    documents = load_pdf(uploaded_file)

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


# Main App
if uploaded_file is not None:

    documents, document_embeddings = prepare_search_engine(uploaded_file)

    query = st.text_input(
        "Enter your query"
    )

    k = st.slider(
        "Top Results",
        min_value=1,
        max_value=len(documents),
        value=min(3, len(documents))
    )

    if st.button("Search"):

        if query.strip():

            results = semantic_search(
                query,
                documents,
                document_embeddings,
                k
            )

            st.subheader("Top Matches")

            medals = ["🥇", "🥈", "🥉"]

            for i, (score, doc) in enumerate(results):

                emoji = medals[i] if i < 3 else "📄"

                with st.container():

                    st.markdown(f"### {emoji} Match #{i+1}")

                    st.write(doc)

                    st.progress(int(score * 100))

                    st.caption(
                        f"Similarity: {score * 100:.2f}%"
                    )

                    st.divider()

        else:
            st.warning("Please enter a query.")

else:
    st.info(" Upload a PDF to begin searching.")
