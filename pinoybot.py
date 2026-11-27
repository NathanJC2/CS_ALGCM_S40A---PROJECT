"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from functools import lru_cache
from typing import Iterable, Sequence

from features import extract_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def load_artifacts(
    classifier_path: str | os.PathLike[str] = CLASSIFIER_PATH,
    vectorizer_path: str | os.PathLike[str] = VECTORIZER_PATH,
):
    if not os.path.isfile(classifier_path) or not os.path.isfile(vectorizer_path):
        raise FileNotFoundError(
            "PinoyBot artifacts are missing. Run: python train_model.py"
        )
    with open(classifier_path, "rb") as stream:
        classifier = pickle.load(stream)
    with open(vectorizer_path, "rb") as stream:
        vectorizer = pickle.load(stream)
    return classifier, vectorizer


@lru_cache(maxsize=1)
def _get_artifacts():
    return load_artifacts()


def predict_language(tokens: Sequence[str]) -> tuple[list[str], list[list[float]], list[str]]:
    """Return labels, class probabilities, and class names in batch."""
    if not tokens:
        return [], [], []
    classifier, vectorizer = _get_artifacts()
    matrix = vectorizer.transform([extract_features(str(token)) for token in tokens])
    labels = classifier.predict(matrix)
    probabilities = classifier.predict_proba(matrix)
    classes = [str(label) for label in classifier.classes_]
    return labels.tolist(), probabilities.tolist(), classes


def tag_language(tokens: Iterable[str]) -> list[str]:
    """
    Predict the language tag of every token.

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]
    """
    labels, _, _ = predict_language(list(tokens))
    return labels