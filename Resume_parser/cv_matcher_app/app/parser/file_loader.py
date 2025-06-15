# File type handling (PDF/DOCX)
# file_loader.py

from pdfminer.high_level import extract_text

def load_file(path):
    """
    Load text from a PDF file. Extend this for DOCX etc.
    """
    if path.lower().endswith(".pdf"):
        return extract_text(path)
    else:
        raise ValueError("Unsupported file format. Only PDF is supported.")
