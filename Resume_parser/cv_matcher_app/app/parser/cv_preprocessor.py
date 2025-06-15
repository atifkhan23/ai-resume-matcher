# Handles file reading & cleaning
# cv_matcher_app/app/parser/cv_preprocessor.py

from app.parser.file_loader import load_file
from app.parser.utils import clean_text, split_sentences

def preprocess_cv(filepath):
    """
    Loads, cleans, and splits CV text.
    Returns a dict with cleaned text and sentences.
    """
    raw_text = load_file(filepath)
    cleaned = clean_text(raw_text)
    sentences = split_sentences(cleaned)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences
    }
