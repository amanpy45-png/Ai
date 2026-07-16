from nltk.tokenize import sent_tokenize

def extract_sentences(text):
    """
    Split raw text into sentences using NLTK.
    """

    return sent_tokenize(text)

def chunk_sentences(
    sentences,
    chunk_size=3,
    overlap=1
):

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    step = chunk_size - overlap

    chunks = []

    for i in range(0, len(sentences), step):

        chunk = sentences[i : i + chunk_size]

        if len(chunk) < chunk_size:
            break

        chunks.append(" ".join(chunk))

    return chunks


def chunk_text(
    text,
    chunk_size=3,
    overlap=1
):

    sentences = extract_sentences(text)

    chunks = chunk_sentences(
        sentences,
        chunk_size,
        overlap
    )

    return chunks