from embedding import get_embedding
import torch.nn.functional as F

def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2).item()

def semantic_search(query, documents, document_embeddings, k=3):
    query_embedding = get_embedding(query)

    results = []

    for doc, emb in zip(documents, document_embeddings):
        score = cosine_similarity(query_embedding, emb)
        results.append((score, doc))

    results.sort(reverse=True)

    return results[:k]