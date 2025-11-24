# src/preprocess.py

import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Convert to lowercase
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str):
    """
    Very simple sentence splitter based on punctuation.
    In a more advanced version, you could use nltk or spaCy.
    """
    # Split on period, question mark, or exclamation mark
    sentences = re.split(r"[.!?]+", text)
    # Remove empty strings and trim whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_words(text: str):
    """
    Split text into words by whitespace and basic punctuation.
    """
    # Remove punctuation (keep letters and numbers and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    return words
