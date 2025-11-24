from typing import Dict
import numpy as np

from .preprocess import clean_text, split_sentences, split_words


def extract_features_from_text(text: str) -> Dict[str, float]:
    """
    Extract simple stylometric features from a single text string.
    Returns a dictionary mapping feature names to numeric values.
    """
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    words = split_words(cleaned)

    num_sentences = len(sentences) if sentences else 1
    num_words = len(words) if words else 1

    # Average sentence length (in words)
    sentence_lengths = [len(split_words(s)) for s in sentences] if sentences else [0]
    avg_sentence_length = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0

    # Average word length (characters)
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0

    # Vocabulary richness (type-token ratio)
    unique_words = len(set(words))
    type_token_ratio = unique_words / num_words if num_words > 0 else 0.0

    # Simple punctuation features (in original text, before stripping)
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    exclamation_count = text.count("!")
    question_count = text.count("?")

    features = {
        "num_sentences": float(num_sentences),
        "num_words": float(num_words),
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "type_token_ratio": type_token_ratio,
        "comma_count": float(comma_count),
        "semicolon_count": float(semicolon_count),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
    }

    return features


def extract_feature_matrix(texts):
    """
    Apply extract_features_from_text to a list/Series of texts and return:
    - X: 2D numpy array of shape (n_samples, n_features)
    - feature_names: list of feature names in order
    """
    all_feature_dicts = [extract_features_from_text(t) for t in texts]

    # Assuming all have same keys
    feature_names = list(all_feature_dicts[0].keys())
    X = np.array([[fd[name] for name in feature_names] for fd in all_feature_dicts])

    return X, feature_names
