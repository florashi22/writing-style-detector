# src/train_ai_classifier.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .features import extract_feature_matrix
from .embeddings import get_embedding


def main():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ai_vs_human.csv")
    df = pd.read_csv(data_path)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print("Extracting stylometric features...")
    X_style, style_features = extract_feature_matrix(texts)

    print("Extracting embeddings (this may take a moment)...")
    embedding_vectors = []
    for t in texts:
        emb = get_embedding(t)
        embedding_vectors.append(emb)

    X_emb = np.array(embedding_vectors)

    # Combine style + embeddings into one vector
    X = np.hstack([X_style, X_emb])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Train classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Save model + metadata
    model_path = os.path.join(os.path.dirname(__file__), "..", "ai_model_embedded.pkl")
    joblib.dump({
        "model": clf,
        "style_features": style_features,
        "embedding_dim": X_emb.shape[1]
    }, model_path)

    print(f"\nSaved Embedding-Powered AI-vs-Human model to {model_path}")


if __name__ == "__main__":
    main()
