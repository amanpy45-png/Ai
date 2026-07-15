import fitz


def load_pdf(pdf_source):

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

    documents = [
        sentence.strip()
        for sentence in text.split("\n")
        if sentence.strip()
    ]

    return documents
