# src/predict_ai_classifier_emb.py

import os
import joblib
import numpy as np

from .features import extract_features_from_text
from .embeddings import get_embedding


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "ai_model_embedded.pkl")
    data = joblib.load(model_path)
    return data["model"], data["style_features"], data["embedding_dim"]


def predict(text):
    model, style_features, emb_dim = load_model()

    # Stylometric features
    style_dict = extract_features_from_text(text)
    style_vec = np.array([style_dict[name] for name in style_features])

    # HuggingFace embedding
    emb_vec = get_embedding(text)

    # Combine both into one vector
    X = np.hstack([style_vec, emb_vec]).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return pred, prob


def main():
    sample = input("Enter text to classify (AI or human):\n> ")
    pred, prob = predict(sample)
    print("\nPrediction:", pred)
    print("Probabilities:", prob)


if __name__ == "__main__":
    main()
