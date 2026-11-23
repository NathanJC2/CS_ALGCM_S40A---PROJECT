"""Train lightweight word-context vectors with a scikit-learn SVD fallback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ContextModel:
    """A small word-vector lookup independent of the selected training backend."""

    vectors: dict[str, np.ndarray]
    backend: str

    def similarity(self, word1: str, word2: str) -> float:
        """Return cosine similarity, or zero when either token is unknown."""
        vector1 = self.vectors.get(word1)
        vector2 = self.vectors.get(word2)
        if vector1 is None or vector2 is None:
            return 0.0
        denominator = float(np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(vector1, vector2) / denominator)


def _train_svd(
    sentences: Sequence[Sequence[str]], components: int, window_size: int
) -> ContextModel:
    """Build a sparse symmetric co-occurrence matrix and reduce it with SVD."""
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD

    vocabulary = sorted({word for sentence in sentences for word in sentence})
    if not vocabulary:
        return ContextModel({}, "sklearn-svd")
    word_index = {word: index for index, word in enumerate(vocabulary)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for sentence in sentences:
        indices = [word_index[word] for word in sentence]
        for center, center_index in enumerate(indices):
            start = max(0, center - window_size)
            end = min(len(indices), center + window_size + 1)
            for context in range(start, end):
                if context != center:
                    row_indices.append(center_index)
                    column_indices.append(indices[context])
                    values.append(1.0 / abs(center - context))

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(vocabulary), len(vocabulary)),
        dtype=np.float64,
    ).tocsr()
    if len(vocabulary) == 1:
        reduced = np.ones((1, 1), dtype=np.float64)
    else:
        dimensions = min(max(1, components), len(vocabulary) - 1)
        reduced = TruncatedSVD(n_components=dimensions, random_state=0).fit_transform(matrix)
    vectors = {word: reduced[index] for word, index in word_index.items()}
    return ContextModel(vectors, "sklearn-svd")


def train_context_model(
    sentences: Sequence[Sequence[str]],
    backend: str = "auto",
    vector_size: int = 50,
    window_size: int = 5,
    min_count: int = 1,
    components: int = 50,
    epochs: int = 10,
) -> ContextModel:
    """Train Word2Vec or fall back to sparse co-occurrence TruncatedSVD."""
    if backend not in {"auto", "gensim", "sklearn"}:
        raise ValueError("backend must be 'auto', 'gensim', or 'sklearn'.")
    if vector_size < 1 or window_size < 1 or min_count < 1 or components < 1 or epochs < 1:
        raise ValueError("Model dimensions, windows, counts, and epochs must be positive.")
    if backend != "sklearn":
        try:
            from gensim.models import Word2Vec
        except ImportError:
            if backend == "gensim":
                raise
        else:
            model = Word2Vec(
                sentences=[list(sentence) for sentence in sentences if sentence],
                vector_size=vector_size,
                window=window_size,
                min_count=min_count,
                workers=1,
                seed=0,
                epochs=epochs,
            )
            vectors = {
                word: np.asarray(model.wv[word], dtype=np.float64)
                for word in model.wv.index_to_key
            }
            return ContextModel(vectors, "gensim-word2vec")
    return _train_svd(sentences, components, window_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "gensim", "sklearn"), default="auto")
    parser.add_argument("--vector-size", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    sample = [["words", "share", "contexts"], ["words", "appear", "nearby"]]
    model = train_context_model(
        sample, args.backend, args.vector_size, args.window_size,
        args.min_count, args.components, args.epochs
    )
    print(f"backend={model.backend} similarity={model.similarity('words', 'contexts'):.3f}")
    return 0


if __name__ == "__main__":
    assert ContextModel({"a": np.array([1.0]), "b": np.array([1.0])}, "test").similarity(
        "a", "b"
    ) == 1.0
    raise SystemExit(main())
