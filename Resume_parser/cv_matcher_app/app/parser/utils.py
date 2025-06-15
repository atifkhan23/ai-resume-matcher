# Shared helpers (text cleaning, regex rules)
# utils.py

import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """
    Remove extra whitespace and fix encoding issues.
    """
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces
    text = text.replace('\x0c', '')           # remove form feed char
    return text.strip()

def split_sentences(text):
    """
    Split text into sentences.
    """
    return sent_tokenize(text)
