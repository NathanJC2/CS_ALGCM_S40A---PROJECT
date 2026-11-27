"""
train_model.py

Trains PinoyBot's language identification model

Input:
    67.csv

Output:
    classifier.pkl
    vectorizer.pkl
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

from features import extract_features


def train(input_csv: str | Path = "67.csv", output_dir: str | Path = "."):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["word", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.upper().replace({
        "FII": "FIL", "FILL": "FIL", "FIL.": "FIL", "O": "OTH",
        "OTJ": "OTH", "EOTH": "OTH",
    })
    valid = ["ENG", "FIL", "CS", "OTH"]
    df = df[df["label"].isin(valid)].copy()
    print("label_counts=" + df["label"].value_counts().to_dict().__str__())

    x = DictVectorizer().fit_transform([extract_features(word) for word in df["word"]])
    y = df["label"]
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    classifier.fit(x_train, y_train)

    for name, features, truth in (
        ("validation", x_valid, y_valid),
        ("test", x_test, y_test),
    ):
        prediction = classifier.predict(features)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, prediction, average="weighted", zero_division=0
        )
        print(f"{name}: accuracy={accuracy_score(truth, prediction):.4f} "
              f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        print(classification_report(truth, prediction, zero_division=0))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "classifier.pkl").open("wb") as stream:
        pickle.dump(classifier, stream)
    vectorizer = DictVectorizer()
    vectorizer.fit([extract_features(word) for word in df["word"]])
    with (target / "vectorizer.pkl").open("wb") as stream:
        pickle.dump(vectorizer, stream)
    print(f"Saved {target / 'classifier.pkl'} and {target / 'vectorizer.pkl'}")
    return classifier, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", default="67.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    train(args.input_csv, args.output_dir)
