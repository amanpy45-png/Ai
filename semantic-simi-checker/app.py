import streamlit as st
from embedding import get_embedding
from search import semantic_search
from utils import load_documents

@st.cache_resource
def load_embeddings():
    documents = load_documents("documents.txt")
    embeddings = [get_embedding(doc) for doc in documents]
    return documents, embeddings

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="centered"
)

st.title("Semantic Search Engine")
st.markdown("Compare your query with stored documents using **BERT embeddings**.")


documents, document_embeddings = load_embeddings()

query = st.text_input("Enter your query")

k = st.slider("Top Results", 1, len(documents), 3)

if st.button("🚀 Search"):

    if not query.strip():
        st.warning("Please enter a query.")
    else:
        results = semantic_search(query, documents, document_embeddings, k)

        st.subheader("Top Matches")

        medals = ["🥇", "🥈", "🥉"]

        for i, (score, doc) in enumerate(results):
            emoji = medals[i] if i < 3 else "📄"

            st.markdown(f"### {emoji} {doc}")
            st.progress(min(score, 1.0))
            st.write(f"**Similarity:** {score*100:.2f}%")
            st.divider()