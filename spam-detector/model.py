"""Train a tiny spam classifier with scikit-learn."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_FILE = Path(__file__).parent / "spam_data.csv"


def train_model():
    """Read the dataset and return a trained model.

    The model is a pipeline of two steps:
      1. TfidfVectorizer: turns text into numbers (word importance scores).
      2. MultinomialNB: Naive Bayes classifier, a classic choice for text.
    """
    data = pd.read_csv(DATA_FILE)
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english"),
        MultinomialNB(),
    )
    model.fit(data["text"], data["label"])
    return model


def predict(model, message):
    """Return ("SPAM" or "NOT SPAM", spam probability between 0 and 1)."""
    spam_probability = model.predict_proba([message])[0][list(model.classes_).index("spam")]
    label = "SPAM" if spam_probability >= 0.5 else "NOT SPAM"
    return label, spam_probability
