import os
import joblib

from .features import extract_features_from_text


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    data = joblib.load(model_path)
    return data["model"], data["feature_names"]


def predict_author(text: str):
    model, feature_names = load_model()
    feature_dict = extract_features_from_text(text)

    # Make sure we pass features in the same order as during training
    X = [[feature_dict[name] for name in feature_names]]

    predicted_author = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return predicted_author, probabilities


def main():
    sample_text = input("Enter a text to analyze:\n> ")

    predicted_author, probs = predict_author(sample_text)
    print(f"\nPredicted author: {predicted_author}")
    print("Class probabilities:", probs)


if __name__ == "__main__":
    main()
