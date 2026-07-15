import fitz


def load_pdf(pdf_source):
    """
    Loads a PDF from either:
    1. A file path (str)
    2. A Streamlit uploaded file

    Returns:
        List[str] -> list of extracted sentences
    """

    # Local PDF
    if isinstance(pdf_source, str):
        document = fitz.open(pdf_source)

    # Uploaded PDF
    else:
        pdf_bytes = pdf_source.read()
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    # Very basic sentence splitting (we'll improve later)
    documents = [
        sentence.strip()
        for sentence in text.split("\n")
        if sentence.strip()
    ]

    return documents