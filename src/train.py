import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .features import extract_feature_matrix


def main():
    # 1. Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.csv")
    df = pd.read_csv(data_path)

    # Expect columns: "text", "author"
    texts = df["text"].tolist()
    labels = df["author"].tolist()

    # 2. Extract features
    X, feature_names = extract_feature_matrix(texts)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # 4. Train a simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    print("Feature names:", feature_names)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model + feature names
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    joblib.dump({"model": clf, "feature_names": feature_names}, model_path)
    print(f"\nSaved trained model to {model_path}")


if __name__ == "__main__":
    main()
