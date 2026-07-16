import fitz


def load_pdf(pdf_source):

    if isinstance(pdf_source, str):
        document = fitz.open(pdf_source)

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

    return text