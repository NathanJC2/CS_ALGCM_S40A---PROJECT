"""Generate, validate, train from, and expand human spelling-pair annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


ROOT = Path(__file__).resolve().parent
ANNOTATION_COLUMNS = (
    "word1",
    "word2",
    "word1_count",
    "word2_count",
    "character_difference",
    "edit_distance",
    "context_similarity",
    "word1_language",
    "word2_language",
    "word1_context_examples",
    "word2_context_examples",
    "is_variant",
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\r?\n+")
VOWELS = set("aeiou")
REQUIRED_ACCURACY = 0.70
FINAL_EVALUATION_SIZE = 100


def _load_module(filename: str, module_name: str):
    module_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_pair(word1: str, word2: str) -> tuple[str, str]:
    """Canonicalize orientation and reject self-pairs."""
    left, right = str(word1).strip().casefold(), str(word2).strip().casefold()
    if not left or not right or left == right:
        raise ValueError("Pairs must contain two distinct non-empty words")
    return tuple(sorted((left, right)))


def _normalize_label(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    text = str(value).strip()
    if text in {"1", "1.0"}:
        return "1"
    if text in {"0", "0.0"}:
        return "0"
    raise ValueError(f"Invalid is_variant value {value!r}; use 1, 0, or blank")


def validate_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize pairs, validate human labels, and remove reversed duplicates."""
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        original = (str(row["word1"]).strip().casefold(),
                    str(row["word2"]).strip().casefold())
        pair = canonical_pair(*original)
        label = _normalize_label(row["is_variant"])
        if pair != original:
            for left_key, right_key in (
                ("word1_count", "word2_count"),
                ("word1_language", "word2_language"),
                ("word1_context_examples", "word2_context_examples"),
            ):
                row[left_key], row[right_key] = row[right_key], row[left_key]
        existing = rows.get(pair)
        if existing and existing["is_variant"] and label and existing["is_variant"] != label:
            raise ValueError(f"Conflicting labels for duplicate pair {pair}")
        normalized = dict(row)
        normalized["word1"], normalized["word2"] = pair
        normalized["is_variant"] = label
        if existing:
            if not existing["is_variant"] and label:
                normalized.update(
                    word1_count=existing["word1_count"],
                    word2_count=existing["word2_count"],
                )
            else:
                continue
        rows[pair] = normalized
    return pd.DataFrame(rows.values(), columns=frame.columns)


def merge_annotations(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Merge CSV files by canonical pair, rejecting contradictory human labels."""
    if not paths:
        raise ValueError("Provide at least one annotation CSV")
    loaded = [pd.read_csv(path, keep_default_na=False) for path in paths]
    combined = pd.concat(loaded, ignore_index=True)
    return validate_annotations(combined)


def _tokenize(preprocess, text: str) -> list[str]:
    return preprocess.tokenize_text(text)


def _iter_sentences(path: Path, encoding: str = "utf-8"):
    preprocess = _load_module("01_preprocess_vocab.py", "annotation_preprocess")
    with path.open("r", encoding=encoding, errors="replace") as stream:
        for line in stream:
            for sentence in SENTENCE_BOUNDARY.split(line):
                tokens = _tokenize(preprocess, sentence)
                if tokens:
                    yield " ".join(sentence.split()), tokens


def _file_metadata(path: str | Path) -> tuple[str, int, int]:
    source = Path(path)
    stats = source.stat()
    return str(source.resolve()), stats.st_size, stats.st_mtime_ns


def _predict_languages(
    words: Sequence[str],
    corpus_path: str | Path,
    batch_size: int = 1000,
) -> dict[str, str]:
    """Cache unique-word PinoyBot predictions against corpus and model versions."""
    cache_path = Path(corpus_path).with_name(".language_predictions.pkl")
    model_metadata = tuple(
        _file_metadata(ROOT / name) if (ROOT / name).is_file() else (name, 0, 0)
        for name in ("classifier.pkl", "vectorizer.pkl")
    )
    metadata = (_file_metadata(corpus_path), model_metadata)
    saved: dict[str, object] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as stream:
            saved = pickle.load(stream)
    cache = saved.get("predictions", {}) if saved.get("metadata") == metadata else {}
    missing = [word for word in sorted(set(words)) if word not in cache]
    if missing:
        pinoy = _load_module("pinoybot.py", "annotation_pinoybot")
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            labels, _, _ = pinoy.predict_language(batch)
            cache.update(zip(batch, labels))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as stream:
            pickle.dump({"metadata": metadata, "predictions": cache}, stream)
    return {word: str(cache[word]) for word in words if word in cache}


def corpus_counts(
    corpus_path: str | Path,
    cache_path: str | Path | None = None,
    encoding: str = "utf-8",
) -> Counter[str]:
    """Stream corpus counts and reuse them when the source file is unchanged."""
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = _file_metadata(path)
    cache = Path(cache_path) if cache_path else path.with_name(".annotation_cache.pkl")
    if cache.is_file():
        with cache.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("metadata") == metadata:
            return Counter(saved["counts"])

    preprocess = _load_module("01_preprocess_vocab.py", "annotation_preprocess_counts")
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    with path.open("r", encoding=encoding, errors="replace") as stream:
        for line in stream:
            counts.update(_tokenize(preprocess, line))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as stream:
        pickle.dump({"metadata": metadata, "counts": dict(counts)}, stream)
    print(f"[corpus counts] {sum(counts.values())} tokens, {len(counts)} types "
          f"in {time.perf_counter() - started:.1f}s")
    return counts


def _canonical_word(word: str) -> str:
    result = word.casefold()
    for old, new in (("ph", "f"), ("qu", "k"), ("f", "p"),
                     ("c", "k"), ("o", "u"), ("e", "i"), ("d", "r")):
        result = result.replace(old, new)
    return result.replace("-", "").replace("'", "").replace("\u2019", "")


def _candidate_signatures(word: str) -> dict[str, str]:
    """Index canonical transforms, deletion forms, and character n-grams."""
    value = word.casefold()
    result = {f"canonical:{_canonical_word(value)}": "canonical"}
    result[f"deletion:{value}"] = "deletion"
    for position in range(len(value)):
        result[f"deletion:{value[:position]}{value[position + 1:]}"] = "deletion"
    for width in (2, 3):
        for position in range(max(0, len(value) - width + 1)):
            result[f"ngram{width}:{value[position:position + width]}"] = "ngram"
    return result


def generate_candidate_pool(
    frequencies: Mapping[str, int],
    pool_limit: int = 12000,
    random_state: int = 42,
    max_length_difference: int = 2,
) -> dict[tuple[str, str], set[str]]:
    """Bounded signature-neighbor sampling; never forms a vocabulary-wide product."""
    if pool_limit < 1 or max_length_difference < 0:
        raise ValueError("pool_limit must be positive and length difference non-negative")
    words = sorted(word for word in frequencies if word)
    postings: dict[str, list[str]] = defaultdict(list)
    word_signatures: dict[str, dict[str, str]] = {}
    for word in words:
        signatures = _candidate_signatures(word)
        word_signatures[word] = signatures
        for signature in signatures:
            postings[signature].append(word)

    rng = random.Random(random_state)
    source_limit = min(len(words), max(12000, pool_limit * 2))
    high_frequency = sorted(words, key=lambda word: (-frequencies[word], word))
    sources = high_frequency[:min(source_limit // 3, 8000)]
    source_set = set(sources)
    remaining = [word for word in words if word not in source_set]
    sources.extend(rng.sample(remaining, min(len(remaining), source_limit - len(sources))))
    rng.shuffle(sources)

    pairs: dict[tuple[str, str], set[str]] = {}
    for word in sources:
        typed_groups: dict[str, list[tuple[int, str, list[str]]]] = {
            "canonical": [],
            "deletion": [],
            "ngram": [],
        }
        for signature, kind in word_signatures[word].items():
            group = postings[signature]
            if len(group) > 1:
                typed_groups[kind].append((len(group), signature, group))
        category_limits = {
            "canonical": (1, 2),
            "deletion": (1, 2),
            "ngram": (2, 1),
        }
        for kind, (group_limit, peer_limit) in category_limits.items():
            groups = sorted(typed_groups[kind], key=lambda item: (item[0], item[1]))
            for _, _, group in groups[:group_limit]:
                sample_size = min(
                    len(group),
                    32 if kind == "canonical" else 8,
                )
                sampled = (
                    group if len(group) <= sample_size
                    else rng.sample(group, sample_size)
                )
                nearest = sorted(
                    (
                        (_char_ngram_cosine(word, other), other)
                        for other in sampled
                        if other != word
                        and abs(len(word) - len(other)) <= max_length_difference
                    ),
                    key=lambda item: (-item[0], item[1]),
                )[:peer_limit]
                for _, other in nearest:
                    pair = canonical_pair(word, other)
                    pairs.setdefault(pair, set()).add(kind)

    if len(pairs) <= pool_limit:
        return pairs

    # Keep a substantial canonical-transformation slice, then sample the rest.
    canonical_pairs = [
        pair for pair, methods in pairs.items() if "canonical" in methods
    ]
    canonical_set = set(canonical_pairs)
    other_pairs = [pair for pair in pairs if pair not in canonical_set]
    canonical_limit = min(len(canonical_pairs), pool_limit // 3)
    canonical_ranked = sorted(
        canonical_pairs,
        key=lambda pair: (
            -_char_ngram_cosine(*pair),
            pair,
        ),
    )
    nearest_quota = min(canonical_limit, max(1, canonical_limit // 2))
    selected = canonical_ranked[:nearest_quota]
    canonical_remaining = canonical_ranked[nearest_quota:]
    selected.extend(rng.sample(
        canonical_remaining,
        min(canonical_limit - len(selected), len(canonical_remaining)),
    ))
    selected_set = set(selected)
    other_limit = pool_limit - len(selected)
    selected.extend(rng.sample(other_pairs, min(other_limit, len(other_pairs))))
    selected_set = set(selected)
    if len(selected) < pool_limit:
        spare = [pair for pair in canonical_pairs if pair not in selected_set]
        selected.extend(rng.sample(spare, min(pool_limit - len(selected), len(spare))))
    return {pair: pairs[pair] for pair in selected}


def _char_ngram_cosine(word1: str, word2: str, minimum: int = 2, maximum: int = 4) -> float:
    def grams(word: str) -> set[str]:
        padded = f"^{word}$"
        return {
            padded[index:index + width]
            for width in range(minimum, maximum + 1)
            for index in range(max(0, len(padded) - width + 1))
        }

    left, right = grams(word1), grams(word2)
    return len(left & right) / math.sqrt(max(1, len(left)) * max(1, len(right)))


def _context_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    dot = sum(left[word] * right[word] for word in common)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _collect_pair_contexts(
    corpus_path: str | Path,
    pairs: Iterable[tuple[str, str]],
    encoding: str = "utf-8",
    max_context_types: int = 64,
    max_examples: int = 2,
) -> tuple[dict[str, Counter[str]], dict[str, list[str]]]:
    targets = {word for pair in pairs for word in pair}
    cache_path = Path(corpus_path).with_name(".context_profiles.pkl")
    metadata = _file_metadata(corpus_path)
    cached_profiles: dict[str, tuple[Counter[str], list[str]]] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("metadata") == metadata:
            cached_profiles = saved.get("profiles", {})
    contexts: dict[str, Counter[str]] = {}
    examples: dict[str, list[str]] = {}
    missing = set(targets)
    for word in targets:
        profile = cached_profiles.get(word)
        if profile is not None and (max_examples == 0 or profile[1]):
            contexts[word] = profile[0]
            examples[word] = profile[1]
            missing.discard(word)
    contexts.update({word: Counter() for word in missing})
    examples.update({word: [] for word in missing})
    if not missing:
        return contexts, examples
    for text, sentence in _iter_sentences(Path(corpus_path), encoding):
        present = missing.intersection(sentence)
        if not present:
            continue
        for index, word in enumerate(sentence):
            if word not in present:
                continue
            if len(examples[word]) < max_examples:
                examples[word].append(text[:360])
            left = max(0, index - 4)
            right = min(len(sentence), index + 5)
            counter = contexts[word]
            for neighbor_index in range(left, right):
                if neighbor_index == index:
                    continue
                neighbor = sentence[neighbor_index]
                if neighbor in counter or len(counter) < max_context_types:
                    counter[neighbor] += 1
    cached_profiles.update({
        word: (contexts[word], examples[word])
        for word in missing
    })
    with cache_path.open("wb") as stream:
        pickle.dump({"metadata": metadata, "profiles": cached_profiles}, stream)
    return contexts, examples


def _pair_info(pair, methods, frequencies, edit_module):
    word1, word2 = pair
    edit = edit_module.analyze_edit_distance(word1, word2)
    return {
        "word1": word1,
        "word2": word2,
        "word1_count": int(frequencies.get(word1, 0)),
        "word2_count": int(frequencies.get(word2, 0)),
        "character_difference": edit.character_difference,
        "edit_distance": int(edit.distance),
        "_ngram_similarity": _char_ngram_cosine(word1, word2),
        "_canonical": float(_canonical_word(word1) == _canonical_word(word2)),
        "_method": "+".join(sorted(methods)),
    }


def _build_records(
    pool: Mapping[tuple[str, str], set[str]],
    frequencies: Mapping[str, int],
) -> list[dict[str, object]]:
    """Score only bounded blocked pairs and discard implausibly distant pairs."""
    edit_module = _load_module("03_dp_edit_distance.py", "annotation_pair_edits")
    records: list[dict[str, object]] = []
    for pair, methods in pool.items():
        row = _pair_info(pair, methods, frequencies, edit_module)
        distance = int(row["edit_distance"])
        denominator = max(1, len(pair[0]), len(pair[1]))
        if distance <= 4 and distance / denominator <= 0.5:
            records.append(row)
    return records


def _diverse_indices(records: list[dict[str, object]], sample_size: int, seed: int) -> list[int]:
    """Round-robin across spelling, context, rarity, and candidate-source strata."""
    if sample_size >= len(records):
        return list(range(len(records)))
    context_order = sorted(
        range(len(records)), key=lambda index: float(records[index]["context_similarity"])
    )
    context_bins: dict[int, int] = {}
    for rank, index in enumerate(context_order):
        context_bins[index] = min(3, rank * 4 // max(1, len(context_order)))
    best_canonical: dict[str, int] = {}
    for index, row in enumerate(records):
        if not row["_canonical"]:
            continue
        key = _canonical_word(str(row["word1"]))
        current = best_canonical.get(key)
        if current is None or float(row["_ngram_similarity"]) > float(
            records[current]["_ngram_similarity"]
        ):
            best_canonical[key] = index
    reserved = sorted(
        best_canonical.values(),
        key=lambda index: (
            -float(records[index]["_ngram_similarity"]),
            int(records[index]["edit_distance"]),
            str(records[index]["word1"]),
            str(records[index]["word2"]),
        ),
    )[:min(sample_size // 100, len(best_canonical))]
    strata: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        distance = int(row["edit_distance"])
        rare = min(3, int(math.log10(max(1, min(
            int(row["word1_count"]), int(row["word2_count"])
        )))))
        key = (
            row["_method"],
            row["_canonical"],
            min(3, distance),
            context_bins[index],
            rare,
        )
        strata[key].append(index)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata, key=str)
    selected: list[int] = list(reserved)
    reserved_set = set(reserved)
    while len(selected) < sample_size and keys:
        remaining = []
        for key in keys:
            while strata[key] and strata[key][-1] in reserved_set:
                strata[key].pop()
            if strata[key]:
                selected.append(strata[key].pop())
                if len(selected) == sample_size:
                    break
            if strata[key]:
                remaining.append(key)
        keys = remaining
    return selected


def generate_annotation_candidates(
    corpus_path: str | Path,
    output_path: str | Path = "annotation_candidates.csv",
    sample_size: int = 1000,
    pool_limit: int | None = None,
    random_state: int = 42,
    cache_path: str | Path | None = None,
    encoding: str = "utf-8",
) -> int:
    """Build a balanced, real-corpus annotation sheet with authentic contexts."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    frequencies = corpus_counts(corpus_path, cache_path, encoding)
    pool_limit = pool_limit or max(12000, sample_size * 12)
    pool = generate_candidate_pool(frequencies, pool_limit, random_state)
    if not pool:
        raise ValueError("No candidate pairs found; inspect corpus tokenization and blocking")
    records = _build_records(pool, frequencies)
    if not records:
        raise ValueError("No candidates remained after edit-distance sanity filtering")
    contexts, examples = _collect_pair_contexts(
        corpus_path, pool.keys(), encoding=encoding
    )
    for row in records:
        word1, word2 = str(row["word1"]), str(row["word2"])
        row["context_similarity"] = _context_cosine(
            contexts.get(word1, {}), contexts.get(word2, {})
        )
        row["word1_context_examples"] = " || ".join(examples.get(word1, []))
        row["word2_context_examples"] = " || ".join(examples.get(word2, []))

    endpoints = sorted({str(row[key]) for row in records for key in ("word1", "word2")})
    language = _predict_languages(endpoints, corpus_path)
    for row in records:
        row["word1_language"] = language.get(str(row["word1"]), "OTH")
        row["word2_language"] = language.get(str(row["word2"]), "OTH")

    selected = _diverse_indices(records, min(sample_size, len(records)), random_state)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for index in selected:
            row = records[index]
            writer.writerow({
                key: row.get(key, "") for key in ANNOTATION_COLUMNS
            })
    print(f"[annotation candidates] pool={len(pool)} endpoints={len(endpoints)} "
          f"written={len(selected)} output={output}")
    return len(selected)


def annotation_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build consistent numeric features from the annotation CSV columns."""
    edit_module = _load_module("03_dp_edit_distance.py", "annotation_feature_edits")
    feature_module = _load_module("05_feature_extractor.py", "annotation_features_core")
    rows: list[dict[str, float]] = []
    for row in frame.to_dict(orient="records"):
        word1, word2 = canonical_pair(row["word1"], row["word2"])
        edit = edit_module.analyze_edit_distance(word1, word2)
        denominator = max(1, len(word1), len(word2))
        positions = edit.normalized_positions
        transformations = [part.split(" vs. ") for part in edit.transformations]
        left_count = int(row["word1_count"] or 0)
        right_count = int(row["word2_count"] or 0)
        left_language = str(row["word1_language"])
        right_language = str(row["word2_language"])
        rows.append({
            "edit_distance": float(edit.distance),
            "normalized_edit_distance": edit.distance / denominator,
            "edit_position_start": float(any(pos <= 1 / 3 for pos in positions)),
            "edit_position_middle": float(any(1 / 3 < pos < 2 / 3 for pos in positions)),
            "edit_position_end": float(any(pos >= 2 / 3 for pos in positions)),
            "transform_vowel_to_vowel": float(any(
                len(a) == len(b) == 1 and a in VOWELS and b in VOWELS
                for a, b in transformations
            )),
            "transform_consonant_to_consonant": float(any(
                len(a) == len(b) == 1 and a not in VOWELS and b not in VOWELS
                and a != "-" and b != "-" for a, b in transformations
            )),
            "transform_insertion_or_deletion": float(any(
                "-" in part for pair in transformations for part in pair
            )),
            "context_cosine": float(row["context_similarity"] or 0),
            "local_context_jaccard": 0.0,
            "log_frequency_ratio": math.log(
                (max(left_count, right_count) + 1) / (min(left_count, right_count) + 1)
            ),
            "minimum_frequency_met": float(min(left_count, right_count) >= 2),
            "character_ngram_tfidf_cosine": _char_ngram_cosine(word1, word2),
            "word_length_difference": float(abs(len(word1) - len(word2))),
            "common_prefix_length": float(len(os_commonprefix(word1, word2))),
            "common_suffix_length": float(common_suffix_length(word1, word2)),
            "pinoy_language_same": float(left_language == right_language),
            "pinoy_filipino_probability": float(
                (left_language == "FIL") + (right_language == "FIL")
            ) / 2.0,
            "pinoy_language_confidence": 0.0,
        })
    return pd.DataFrame(rows, columns=feature_module.FEATURE_COLUMNS)


def os_commonprefix(left: str, right: str) -> str:
    index = 0
    for first, second in zip(left, right):
        if first != second:
            break
        index += 1
    return left[:index]


def common_suffix_length(left: str, right: str) -> int:
    length = 0
    for first, second in zip(reversed(left), reversed(right)):
        if first != second:
            break
        length += 1
    return length


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_metadata_path(test_csv: str | Path) -> Path:
    path = Path(test_csv)
    return path.with_suffix(path.suffix + ".metadata.json")


def _training_pair_set(artifact: Mapping[str, object]) -> set[tuple[str, str]]:
    stored = artifact.get("training_pairs", [])
    return {
        canonical_pair(str(pair[0]), str(pair[1]))
        for pair in stored
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }


def _known_independent_test_pairs(
    exclude_test_data_path: str | Path | None = None,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    excluded_path = (
        Path(exclude_test_data_path).resolve()
        if exclude_test_data_path is not None
        else None
    )
    for metadata_path in ROOT.rglob("*.csv.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("independent_test_dataset"):
            continue
        registered_path = metadata.get("test_dataset_path")
        if registered_path and excluded_path == Path(registered_path).resolve():
            continue
        pairs.update(
            canonical_pair(str(pair[0]), str(pair[1]))
            for pair in metadata.get("test_pairs", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        )
    return pairs


def _annotation_source_paths(
    artifact: Mapping[str, object],
    extra_paths: Sequence[str | Path] = (),
) -> list[Path]:
    sources = list(artifact.get("training_source", [])) + list(extra_paths)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        path = Path(str(source))
        if not path.is_file() and not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            final = path.resolve()
            if final not in seen:
                seen.add(final)
                resolved.append(final)
    return resolved


def _load_exclusions(
    artifact: Mapping[str, object],
    annotation_paths: Sequence[str | Path] = (),
    exclude_test_data_path: str | Path | None = None,
) -> tuple[set[tuple[str, str]], set[str], list[Path]]:
    missing_paths = [str(path) for path in annotation_paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Annotation CSV file(s) not found: {', '.join(missing_paths)}"
        )
    paths = _annotation_source_paths(artifact, annotation_paths)
    _reject_independent_test_training(paths)
    pairs = _training_pair_set(artifact)
    if paths:
        frame = merge_annotations(paths)
        pairs.update(
            canonical_pair(str(row.word1), str(row.word2))
            for row in frame.itertuples(index=False)
        )
    elif not pairs:
        raise ValueError(
            "Cannot prove test independence: provide the training/annotation CSVs "
            "or retrain to save training pair identities in the model."
        )
    pairs.update(_known_independent_test_pairs(exclude_test_data_path))
    lexical_groups = {
        _canonical_word(word)
        for pair in pairs
        for word in pair
    }
    return pairs, lexical_groups, paths


def _reject_independent_test_training(paths: Sequence[str | Path]) -> None:
    for source in paths:
        metadata_path = _test_metadata_path(source)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("independent_test_dataset"):
                raise ValueError(
                    f"{source} is an independent test set and must never be used for training"
                )
    annotated_pairs: set[tuple[str, str]] = set()
    for source in paths:
        if not Path(source).is_file():
            continue
        frame = pd.read_csv(source, keep_default_na=False)
        if {"word1", "word2"}.issubset(frame.columns):
            annotated_pairs.update(
                canonical_pair(str(left), str(right))
                for left, right in zip(frame["word1"], frame["word2"])
            )
    overlap = annotated_pairs & _known_independent_test_pairs()
    if overlap:
        raise ValueError(
            f"Training data contains {len(overlap)} pair(s) from a registered "
            "independent test dataset; do not train on test labels."
        )


def train_from_annotations(
    annotation_paths: Sequence[str | Path],
    model_path: str | Path = "variant_classifier.pkl",
    n_estimators: int = 300,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Train only on verified 0/1 labels; tune threshold from out-of-fold scores."""
    _reject_independent_test_training(annotation_paths)
    annotations = merge_annotations(annotation_paths)
    reviewed = annotations[annotations["is_variant"].isin(["0", "1"])].copy()
    if reviewed.empty:
        raise ValueError("No reviewed 0/1 labels; blank annotations are not training data")
    labels = reviewed["is_variant"].astype(int).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("Verified training data must include both positive and negative labels")
    features = annotation_features(reviewed)
    groups = np.asarray([
        "|".join(sorted((_canonical_word(str(row["word1"])),
                         _canonical_word(str(row["word2"])))))
        for row in reviewed.to_dict(orient="records")
    ])
    class_counts = np.bincount(labels)
    class_group_counts = [
        len(set(groups[labels == label]))
        for label in (0, 1)
    ]
    folds = min(n_splits, int(class_counts.min()), min(class_group_counts))
    if folds < 2:
        raise ValueError("At least two reviewed examples per class are needed for OOF evaluation")
    estimator = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state,
        class_weight="balanced", n_jobs=-1,
    )
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    oof_probability = cross_val_predict(
        estimator, features, labels, cv=splitter,
        groups=groups, method="predict_proba"
    )[:, 1]
    thresholds = np.arange(0.10, 0.951, 0.025)
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for threshold in thresholds:
        predicted = (oof_probability >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        key = (precision, f1)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    estimator.fit(features, labels)
    predicted = (oof_probability >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    accuracy = accuracy_score(labels, predicted)
    balanced_accuracy = balanced_accuracy_score(labels, predicted)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    training_pairs = sorted({
        canonical_pair(str(row.word1), str(row.word2))
        for row in reviewed.itertuples(index=False)
    })
    fingerprint_rows = sorted(
        (
            *canonical_pair(str(row.word1), str(row.word2)),
            str(row.is_variant),
        )
        for row in reviewed.itertuples(index=False)
    )
    training_data_fingerprint = hashlib.sha256(
        "\n".join("\t".join(row) for row in fingerprint_rows).encode("utf-8")
    ).hexdigest()
    import sklearn

    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump({
            "classifier": estimator,
            "feature_columns": tuple(features.columns),
            "threshold": best_threshold,
            "training_source": [str(Path(path).resolve()) for path in annotation_paths],
            "training_pairs": training_pairs,
            "training_data_fingerprint": training_data_fingerprint,
            "sklearn_version": sklearn.__version__,
            "verified_rows": len(reviewed),
            "evaluation": {
                "method": f"{folds}-fold out-of-fold",
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "confusion_matrix": matrix,
            },
        }, stream)
    result = {
        "verified_rows": len(reviewed),
        "threshold": best_threshold,
        "oof_precision": float(precision),
        "oof_recall": float(recall),
        "oof_f1": float(f1),
        "oof_accuracy": float(accuracy),
        "oof_balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix": matrix,
        "model": str(target),
    }
    print(f"[training] {result}")
    return result


def _load_variant_artifact(model_path: str | Path) -> dict[str, object]:
    with Path(model_path).open("rb") as stream:
        artifact = pickle.load(stream)
    if not isinstance(artifact, dict) or "classifier" not in artifact:
        raise ValueError(f"{model_path} is not a spelling-variant model artifact")
    import sklearn

    saved_version = artifact.get("sklearn_version")
    if saved_version and saved_version != sklearn.__version__:
        raise RuntimeError(
            f"Model uses scikit-learn {saved_version}, current version is "
            f"{sklearn.__version__}; retrain from the verified annotation CSV."
        )
    return artifact


def _candidate_frame_for_model(
    corpus_path: str | Path,
    artifact: Mapping[str, object],
    pool_limit: int,
    random_state: int,
    excluded_pairs: set[tuple[str, str]] | None = None,
    excluded_groups: set[str] | None = None,
    include_context_examples: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, Counter[str], int]:
    frequencies = corpus_counts(corpus_path)
    pool = generate_candidate_pool(frequencies, pool_limit, random_state)
    records = _build_records(pool, frequencies)
    excluded_pairs = excluded_pairs or set()
    excluded_groups = excluded_groups or set()
    records = [
        row for row in records
        if canonical_pair(str(row["word1"]), str(row["word2"])) not in excluded_pairs
        and _canonical_word(str(row["word1"])) not in excluded_groups
        and _canonical_word(str(row["word2"])) not in excluded_groups
    ]
    if not records:
        raise ValueError("No unseen candidate pairs remain after exclusions")

    pairs = [(str(row["word1"]), str(row["word2"])) for row in records]
    contexts, examples = _collect_pair_contexts(
        corpus_path,
        pairs,
        max_examples=2 if include_context_examples else 0,
    )
    endpoints = sorted({word for pair in pairs for word in pair})
    languages = _predict_languages(endpoints, corpus_path)
    for row in records:
        word1, word2 = str(row["word1"]), str(row["word2"])
        row["context_similarity"] = _context_cosine(
            contexts.get(word1, {}), contexts.get(word2, {})
        )
        row["word1_language"] = languages.get(word1, "OTH")
        row["word2_language"] = languages.get(word2, "OTH")
        row["word1_context_examples"] = " || ".join(examples.get(word1, []))
        row["word2_context_examples"] = " || ".join(examples.get(word2, []))
        row["is_variant"] = ""
    frame = pd.DataFrame(records)
    features = annotation_features(frame)
    feature_columns = list(artifact.get("feature_columns", features.columns))
    missing_features = set(feature_columns) - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Current feature extractor is missing model columns: {sorted(missing_features)}"
        )
    classifier = artifact["classifier"]
    positive_index = list(classifier.classes_).index(1)
    probabilities = classifier.predict_proba(features[feature_columns])[:, positive_index]
    return frame, probabilities, Counter(frequencies), len(pool)


def _write_annotation_sheet(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(ANNOTATION_COLUMNS)].to_csv(
        output, index=False, encoding="utf-8-sig"
    )


def create_independent_test_sample(
    corpus_path: str | Path,
    model_path: str | Path,
    annotation_paths: Sequence[str | Path],
    output_path: str | Path = "Test_Dataset.csv",
    sample_size: int = 1000,
    candidate_limit: int = 100000,
    random_state: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a proportionate, prediction-hidden sample excluded from training groups."""
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2 to include both prediction classes")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    output = Path(output_path)
    metadata_path = _test_metadata_path(output)
    if (output.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"{output} or its metadata already exists; choose a new path or pass --overwrite"
        )

    artifact = _load_variant_artifact(model_path)
    excluded_pairs, excluded_groups, source_paths = _load_exclusions(
        artifact, annotation_paths
    )
    frame, probabilities, frequencies, pool_count = _candidate_frame_for_model(
        corpus_path,
        artifact,
        candidate_limit,
        random_state,
        excluded_pairs,
        excluded_groups,
    )
    threshold = float(artifact["threshold"])
    predictions = (probabilities >= threshold).astype(int)
    positive_indices = np.flatnonzero(predictions == 1).tolist()
    negative_indices = np.flatnonzero(predictions == 0).tolist()
    if not positive_indices or not negative_indices:
        raise ValueError(
            "The eligible candidate pool must contain both predicted classes for a test sample"
        )
    if sample_size > len(frame):
        raise ValueError(
            f"Requested {sample_size} pairs, but only {len(frame)} unseen eligible pairs exist"
        )

    positive_quota = round(
        sample_size * len(positive_indices) / len(frame)
    )
    positive_quota = max(1, min(sample_size - 1, positive_quota))
    positive_quota = min(positive_quota, len(positive_indices))
    negative_quota = sample_size - positive_quota
    if negative_quota > len(negative_indices):
        negative_quota = len(negative_indices)
        positive_quota = sample_size - negative_quota
    if positive_quota > len(positive_indices):
        raise ValueError("Not enough unseen candidates from both predicted classes")

    rng = random.Random(random_state)
    selected = rng.sample(positive_indices, positive_quota)
    selected.extend(rng.sample(negative_indices, negative_quota))
    rng.shuffle(selected)
    test_frame = frame.iloc[selected].loc[:, list(ANNOTATION_COLUMNS)].copy()
    test_frame["is_variant"] = ""
    pairs = {
        canonical_pair(str(row.word1), str(row.word2))
        for row in test_frame.itertuples(index=False)
    }
    overlap = pairs & excluded_pairs
    if overlap:
        raise RuntimeError(f"Internal error: {len(overlap)} training pairs entered the test set")
    _write_annotation_sheet(test_frame, output)

    source_digest = hashlib.sha256(
        "\n".join(sorted(str(path) for path in source_paths)).encode("utf-8")
    ).hexdigest()
    metadata = {
        "independent_test_dataset": True,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset_path": str(output.resolve()),
        "corpus": _file_metadata(corpus_path),
        "model_sha256": _sha256_file(model_path),
        "model_sklearn_version": artifact.get("sklearn_version"),
        "training_data_fingerprint": artifact.get("training_data_fingerprint"),
        "training_sources_fingerprint": source_digest,
        "threshold": threshold,
        "sample_size": len(test_frame),
        "random_state": random_state,
        "candidate_limit": candidate_limit,
        "candidate_pool_pairs": pool_count,
        "eligible_unseen_candidates": len(frame),
        "excluded_annotated_pairs": len(excluded_pairs),
        "excluded_lexical_groups": len(excluded_groups),
        "eligible_predicted_positive": len(positive_indices),
        "eligible_predicted_negative": len(negative_indices),
        "sampled_predicted_positive": positive_quota,
        "sampled_predicted_negative": negative_quota,
        "test_pairs": [list(pair) for pair in sorted(pairs)],
        "training_pair_overlap": len(overlap),
        "prediction_values_are_hidden": True,
        "corpus_vocabulary_size": len(frequencies),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[independent test sample] exported={len(test_frame)} "
        f"positive={positive_quota} negative={negative_quota} "
        f"pool={pool_count} eligible={len(frame)} output={output}"
    )
    return metadata


def _parse_binary_value(value: object, field_name: str) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().casefold()
    positive = {"1", "1.0", "true", "yes", "correct", "variant"}
    negative = {"0", "0.0", "false", "no", "incorrect", "not variant"}
    if text in positive:
        return 1
    if text in negative:
        return 0
    raise ValueError(
        f"Invalid {field_name} value {value!r}; use correct/incorrect or 1/0"
    )


def _training_cv_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    evaluation = artifact.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    matrix = evaluation.get("confusion_matrix")
    if matrix and len(matrix) == 2 and len(matrix[0]) == 2:
        tn, fp = int(matrix[0][0]), int(matrix[0][1])
        fn, tp = int(matrix[1][0]), int(matrix[1][1])
        total = tn + fp + fn + tp
        accuracy = float(evaluation.get("accuracy", (tn + tp) / total if total else 0))
        balanced = float(evaluation.get(
            "balanced_accuracy",
            ((tn / (tn + fp)) + (tp / (tp + fn))) / 2
            if (tn + fp) and (tp + fn) else 0,
        ))
    else:
        tn = fp = fn = tp = 0
        total = int(artifact.get("verified_rows", 0))
        accuracy = evaluation.get("accuracy")
        balanced = evaluation.get("balanced_accuracy")
    return {
        "status": "AVAILABLE" if evaluation else "NOT_AVAILABLE",
        "method": evaluation.get("method"),
        "training_samples": int(artifact.get("verified_rows", total)),
        "threshold": artifact.get("threshold"),
        "accuracy": accuracy,
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "balanced_accuracy": balanced,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sklearn_version": artifact.get("sklearn_version"),
    }


def _load_final_report(path: str | Path, artifact: Mapping[str, object]) -> dict[str, object]:
    report_path = Path(path)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report["training_cross_validation"] = _training_cv_summary(artifact)
    report.setdefault("final_output_precision", {"status": "NOT_EVALUATED"})
    report.setdefault("independent_raw_test", {"status": "NOT_EVALUATED"})
    return report


def _format_percent(value: object) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def _write_final_reports(
    report: Mapping[str, object],
    json_path: str | Path = "Final_Evaluation.json",
    text_path: str | Path = "Final_Evaluation.txt",
) -> None:
    json_target, text_target = Path(json_path), Path(text_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    text_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cv = report["training_cross_validation"]
    output = report["final_output_precision"]
    test = report["independent_raw_test"]
    lines = [
        "FINAL EVALUATION REPORT",
        "",
        "1. TRAINING CROSS-VALIDATION",
        f"Status: {cv.get('status')}",
        f"Method: {cv.get('method') or 'N/A'}",
        f"Training samples: {cv.get('training_samples', 0)}",
        f"Threshold: {cv.get('threshold')}",
        f"Accuracy: {_format_percent(cv.get('accuracy'))}",
        f"Precision: {_format_percent(cv.get('precision'))}",
        f"Recall: {_format_percent(cv.get('recall'))}",
        f"F1-score: {_format_percent(cv.get('f1'))}",
        f"Balanced accuracy: {_format_percent(cv.get('balanced_accuracy'))}",
        "",
        "2. FINAL RAW.TXT OUTPUT PRECISION",
        f"Status: {output.get('status')}",
        f"Evaluated pairs: {output.get('evaluated_pairs', 0)}",
        f"Correct variants: {output.get('correct_variants', 0)}",
        f"Incorrect variants: {output.get('incorrect_variants', 0)}",
        f"Unreviewed pairs: {output.get('unreviewed_pairs', 0)}",
        f"Output precision: {_format_percent(output.get('output_precision'))}",
        "Required threshold: >70%",
        "",
        "3. INDEPENDENT RAW.TXT CANDIDATE TEST",
        f"Status: {test.get('status')}",
        f"Test samples: {test.get('test_samples', 0)}",
        f"Evaluated samples: {test.get('evaluated_samples', 0)}",
        f"Unreviewed samples: {test.get('unreviewed_samples', 0)}",
        f"Training pair overlap: {test.get('training_pair_overlap', 'N/A')}",
        f"Accuracy: {_format_percent(test.get('accuracy'))}",
        f"Majority-class baseline accuracy: {_format_percent(test.get('majority_class_baseline_accuracy'))}",
        f"Precision: {_format_percent(test.get('precision'))}",
        f"Recall: {_format_percent(test.get('recall'))}",
        f"F1-score: {_format_percent(test.get('f1'))}",
        f"Balanced accuracy: {_format_percent(test.get('balanced_accuracy'))}",
        f"Threshold: {test.get('threshold')}",
        f"True positives: {test.get('true_positives', 0)}",
        f"True negatives: {test.get('true_negatives', 0)}",
        f"False positives: {test.get('false_positives', 0)}",
        f"False negatives: {test.get('false_negatives', 0)}",
        "Required accuracy: >70%",
    ]
    if test.get("confusion_matrix") is not None:
        lines.extend(["Confusion matrix:", str(test["confusion_matrix"])])
    text_target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_final_output(
    evaluation_xlsx: str | Path,
    corpus_path: str | Path = "Raw.txt",
    model_path: str | Path = "variant_classifier.pkl",
    json_path: str | Path = "Final_Evaluation.json",
    text_path: str | Path = "Final_Evaluation.txt",
) -> dict[str, object]:
    """Calculate final output precision from human-marked workbook cells."""
    from openpyxl import load_workbook

    workbook = load_workbook(evaluation_xlsx, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        try:
            header_row = next(
                sheet.iter_rows(min_row=1, max_row=1, values_only=True)
            )
        except StopIteration as error:
            raise ValueError("Evaluation workbook is empty") from error
        header = [
            str(value).strip().casefold() if value is not None else ""
            for value in header_row
        ]
        missing_columns = {"word1", "word2", "is_correct"} - set(header)
        if missing_columns:
            raise ValueError(
                f"Evaluation workbook is missing columns: {sorted(missing_columns)}"
            )
        label_column = header.index("is_correct")
        word1_column = header.index("word1")
        word2_column = header.index("word2")
        values: list[int | None] = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if not row[word1_column] and not row[word2_column]:
                continue
            values.append(_parse_binary_value(row[label_column], "is_correct"))
    finally:
        workbook.close()

    correct = sum(value == 1 for value in values)
    incorrect = sum(value == 0 for value in values)
    evaluated = correct + incorrect
    unreviewed = len(values) - evaluated
    precision = correct / evaluated if evaluated else None
    complete = len(values) == FINAL_EVALUATION_SIZE and unreviewed == 0
    status = (
        "INCOMPLETE" if not complete
        else "PASS" if precision is not None and precision > REQUIRED_ACCURACY
        else "FAIL"
    )
    metrics: dict[str, object] = {
        "status": status,
        "input_corpus": str(corpus_path),
        "evaluation_workbook": str(evaluation_xlsx),
        "sample_pairs": len(values),
        "evaluated_pairs": evaluated,
        "correct_variants": correct,
        "incorrect_variants": incorrect,
        "unreviewed_pairs": unreviewed,
        "output_precision": precision,
        "required_precision_strictly_greater_than": REQUIRED_ACCURACY,
        "requires_exactly_100_reviewed_pairs_for_pass": True,
    }
    artifact = _load_variant_artifact(model_path)
    report = _load_final_report(json_path, artifact)
    report["final_output_precision"] = metrics
    _write_final_reports(report, json_path, text_path)

    print("========== FINAL CORPUS EVALUATION ==========")
    print(f"Input corpus: {corpus_path}")
    print(f"Evaluated pairs: {evaluated}")
    print(f"Correct variants: {correct}")
    print(f"Incorrect variants: {incorrect}")
    print(f"Unreviewed pairs: {unreviewed}")
    print(f"Output precision: {_format_percent(precision)}")
    print("Required threshold: >70%")
    print(f"Result: {status}")
    print("=============================================")
    return metrics


def evaluate_independent_test(
    corpus_path: str | Path,
    model_path: str | Path,
    test_data_path: str | Path,
    annotation_paths: Sequence[str | Path] = (),
    json_path: str | Path = "Final_Evaluation.json",
    text_path: str | Path = "Final_Evaluation.txt",
) -> dict[str, object]:
    """Evaluate frozen model predictions against manually labeled independent pairs."""
    metadata_path = _test_metadata_path(test_data_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing independence metadata {metadata_path}; generate the test sample "
            "with the test-sample command."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("independent_test_dataset"):
        raise ValueError("Test dataset metadata does not identify an independent test set")
    if metadata.get("model_sha256") != _sha256_file(model_path):
        raise RuntimeError(
            "The model changed after test-sample generation; create a fresh independent test set."
        )
    if tuple(metadata.get("corpus", ())) != _file_metadata(corpus_path):
        raise RuntimeError(
            "The corpus changed after test-sample generation; create a fresh independent test set."
        )
    artifact = _load_variant_artifact(model_path)
    if metadata.get("threshold") != artifact.get("threshold"):
        raise RuntimeError("The classifier threshold changed; generate a fresh independent test set.")

    excluded_pairs, excluded_groups, _ = _load_exclusions(
        artifact, annotation_paths, test_data_path
    )
    raw_frame = pd.read_csv(test_data_path, keep_default_na=False)
    frame = validate_annotations(raw_frame)
    test_pairs = {
        canonical_pair(str(row.word1), str(row.word2))
        for row in frame.itertuples(index=False)
    }
    registered_pairs = {
        canonical_pair(str(pair[0]), str(pair[1]))
        for pair in metadata.get("test_pairs", [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }
    if registered_pairs != test_pairs:
        raise RuntimeError(
            "The pair identities in the labeled test CSV differ from its generation manifest."
        )
    exact_overlap = test_pairs & excluded_pairs
    lexical_overlap_words = {
        _canonical_word(word)
        for pair in test_pairs
        for word in pair
    } & excluded_groups
    if exact_overlap or lexical_overlap_words:
        raise ValueError(
            f"Independent test leakage detected: {len(exact_overlap)} exact pairs, "
            f"{len(lexical_overlap_words)} lexical groups overlap training/annotations"
        )

    reviewed = frame[frame["is_variant"].isin(["0", "1"])].copy()
    unreviewed = len(frame) - len(reviewed)
    labels = reviewed["is_variant"].astype(int).to_numpy()
    sample_is_complete = (
        len(frame) == int(metadata.get("sample_size", -1))
        and unreviewed == 0
        and len(frame) > 0
    )
    threshold = float(artifact["threshold"])
    if len(reviewed):
        features = annotation_features(reviewed)
        feature_columns = list(artifact.get("feature_columns", features.columns))
        probabilities = artifact["classifier"].predict_proba(
            features[feature_columns]
        )[:, list(artifact["classifier"].classes_).index(1)]
        predicted = (probabilities >= threshold).astype(int)
        accuracy = float(accuracy_score(labels, predicted))
        precision = float(precision_score(labels, predicted, zero_division=0))
        recall = float(recall_score(labels, predicted, zero_division=0))
        f1 = float(f1_score(labels, predicted, zero_division=0))
        balanced_accuracy = float(balanced_accuracy_score(labels, predicted))
        matrix = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        majority_baseline = float(max(np.bincount(labels)) / len(labels))
    else:
        accuracy = precision = recall = f1 = balanced_accuracy = None
        matrix = [[0, 0], [0, 0]]
        tn = fp = fn = tp = 0
        majority_baseline = None
    status = (
        "INCOMPLETE" if not sample_is_complete
        else "PASS" if accuracy is not None and accuracy > REQUIRED_ACCURACY
        else "FAIL"
    )
    metrics: dict[str, object] = {
        "status": status,
        "input_corpus": str(corpus_path),
        "test_dataset": str(test_data_path),
        "test_samples": len(frame),
        "evaluated_samples": len(reviewed),
        "unreviewed_samples": unreviewed,
        "training_pair_overlap": len(exact_overlap),
        "training_lexical_group_overlap": len(lexical_overlap_words),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "majority_class_baseline_accuracy": majority_baseline,
        "confusion_matrix": matrix,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "required_accuracy_strictly_greater_than": REQUIRED_ACCURACY,
        "metrics_use_reviewed_rows_only": True,
        "test_scope": "blocked candidate pairs sampled from Raw.txt, not all vocabulary pairs",
    }
    report = _load_final_report(json_path, artifact)
    report["independent_raw_test"] = metrics
    _write_final_reports(report, json_path, text_path)

    print("========== INDEPENDENT RAW.TXT TEST ==========")
    print(f"Input corpus: {corpus_path}")
    print(f"Test samples: {len(frame)}")
    print(f"Evaluated samples: {len(reviewed)}")
    print(f"Unreviewed samples: {unreviewed}")
    print(f"Training pair overlap: {len(exact_overlap)}")
    print(f"Training lexical-group overlap: {len(lexical_overlap_words)}")
    print(f"Classification threshold: {threshold:.4f}")
    print(f"Accuracy: {_format_percent(accuracy)}")
    print(f"Precision: {_format_percent(precision)}")
    print(f"Recall: {_format_percent(recall)}")
    print(f"F1-score: {_format_percent(f1)}")
    print(f"Balanced accuracy: {_format_percent(balanced_accuracy)}")
    print(f"True positives: {tp}")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print("Confusion matrix:")
    print(f"[[{tn}, {fp}],")
    print(f" [{fn}, {tp}]]")
    print(f"Majority-class baseline accuracy: {_format_percent(majority_baseline)}")
    print("Required accuracy: >70%")
    print(f"Result: {status}")
    print("==============================================")
    return metrics


def create_active_learning_sample(
    corpus_path: str | Path,
    model_path: str | Path,
    output_path: str | Path = "active_learning_candidates.csv",
    sample_size: int = 1000,
    pool_limit: int | None = None,
    random_state: int = 42,
) -> int:
    """Export near-threshold, stratified unreviewed examples for human review."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    artifact = _load_variant_artifact(model_path)
    classifier = artifact["classifier"]
    threshold = float(artifact["threshold"])
    frequencies = corpus_counts(corpus_path)
    pool = generate_candidate_pool(
        frequencies, pool_limit or max(sample_size * 15, 15000), random_state
    )
    records = _build_records(pool, frequencies)
    if not records:
        raise ValueError("No candidates remained after edit-distance sanity filtering")
    contexts, examples = _collect_pair_contexts(
        corpus_path, pool.keys(), max_examples=0
    )
    endpoints = sorted({word for pair in pool for word in pair})
    language = _predict_languages(endpoints, corpus_path)
    for row in records:
        left, right = str(row["word1"]), str(row["word2"])
        row["context_similarity"] = _context_cosine(
            contexts.get(left, {}), contexts.get(right, {})
        )
        row["word1_context_examples"] = " || ".join(examples.get(left, []))
        row["word2_context_examples"] = " || ".join(examples.get(right, []))
        row["word1_language"] = language.get(left, "OTH")
        row["word2_language"] = language.get(right, "OTH")
        row["is_variant"] = ""
    frame = pd.DataFrame(records)
    features = annotation_features(frame)
    probabilities = classifier.predict_proba(features)[:, 1]
    frame["_probability"] = probabilities
    frame["_distance_to_threshold"] = np.abs(probabilities - threshold)
    # Include candidates on both sides of the threshold to avoid one-sided review.
    ranked = frame.sort_values(
        ["_distance_to_threshold", "_ngram_similarity"],
        ascending=[True, False],
    )
    seen_keys: set[str] = set()
    selected: list[int] = []
    for index, row in ranked.iterrows():
        key = _canonical_word(str(row["word1"]))
        other_key = _canonical_word(str(row["word2"]))
        if key in seen_keys or other_key in seen_keys:
            continue
        selected.append(index)
        seen_keys.update((key, other_key))
        if len(selected) >= sample_size:
            break
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[selected, list(ANNOTATION_COLUMNS)].to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    print(f"[active learning] pool={len(frame)} exported={len(selected)} "
          f"threshold={threshold:.3f} output={output}")
    return len(selected)


def run_full_pipeline(
    corpus_path: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    evaluation_xlsx: str | Path | None = None,
    candidate_limit: int = 100000,
    random_state: int = 42,
) -> dict[str, object]:
    """Predict from a bounded candidate pool, resolve, export, and optionally sample."""
    artifact = _load_variant_artifact(model_path)
    frequencies = corpus_counts(corpus_path)
    pool = generate_candidate_pool(frequencies, candidate_limit, random_state)
    records = _build_records(pool, frequencies)
    if not records:
        exporter = _load_module("08_export_variants_csv.py", "fullrun_empty_export")
        exporter.export_variants([], dict(frequencies), output_csv)
        evaluation_count = 0
        if evaluation_xlsx:
            sampler = _load_module(
                "09_sampler_for_evaluation.py", "fullrun_empty_sampler"
            )
            evaluation_count = sampler.create_evaluation_sample(
                output_csv, evaluation_xlsx, 100, random_state
            )
        result = {
            "vocabulary": len(frequencies),
            "candidate_pairs": len(pool),
            "predicted_pairs": 0,
            "exported_pairs": 0,
            "evaluation_sample": evaluation_count,
            "threshold": float(artifact["threshold"]),
        }
        print(f"[full pipeline] {result}")
        return result
    contexts, _ = _collect_pair_contexts(
        corpus_path, pool.keys(), max_examples=0
    )
    endpoints = sorted({word for pair in pool for word in pair})
    language = _predict_languages(endpoints, corpus_path)
    for row in records:
        left, right = str(row["word1"]), str(row["word2"])
        row["context_similarity"] = _context_cosine(
            contexts.get(left, {}), contexts.get(right, {})
        )
        row["word1_context_examples"] = ""
        row["word2_context_examples"] = ""
        row["word1_language"] = language.get(left, "OTH")
        row["word2_language"] = language.get(right, "OTH")
        row["is_variant"] = ""
    frame = pd.DataFrame(records)
    features = annotation_features(frame)
    classifier = artifact["classifier"]
    probabilities = classifier.predict_proba(features)[:, 1]
    threshold = float(artifact["threshold"])
    resolver = _load_module("07_greedy_cluster_resolver.py", "fullrun_resolver")
    selected = [
        (str(row.word1), str(row.word2), float(probability))
        for row, probability in zip(frame.itertuples(index=False), probabilities)
        if probability >= threshold
    ]
    resolved = resolver.resolve_variant_pairs(
        selected, minimum_confidence=threshold, cohesion_threshold=threshold
    )
    exporter = _load_module("08_export_variants_csv.py", "fullrun_export")
    count = exporter.export_variants(resolved, dict(frequencies), output_csv)
    evaluation_count = 0
    if evaluation_xlsx:
        sampler = _load_module("09_sampler_for_evaluation.py", "fullrun_sampler")
        evaluation_count = sampler.create_evaluation_sample(
            output_csv, evaluation_xlsx, 100, random_state
        )
    result = {
        "vocabulary": len(frequencies),
        "candidate_pairs": len(pool),
        "predicted_pairs": len(resolved),
        "exported_pairs": count,
        "evaluation_sample": evaluation_count,
        "threshold": threshold,
    }
    print(f"[full pipeline] {result}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate blank annotation candidates")
    generate.add_argument("--corpus", default="Raw.txt")
    generate.add_argument("--output", default="annotation_candidates.csv")
    generate.add_argument("--sample-size", type=int, default=1000)
    generate.add_argument("--pool-limit", type=int)
    generate.add_argument("--random-state", type=int, default=42)
    generate.add_argument("--cache")

    validate = commands.add_parser("validate", help="Validate and canonicalize annotations")
    validate.add_argument("--annotations", nargs="+", required=True)
    validate.add_argument("--output", default="annotations_merged.csv")

    train = commands.add_parser("train", help="Train from verified 0/1 labels")
    train.add_argument("--annotations", nargs="+", required=True)
    train.add_argument("--model", default="variant_classifier.pkl")
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--n-splits", type=int, default=5)
    train.add_argument("--random-state", type=int, default=42)

    active = commands.add_parser("active", help="Generate active-learning candidates")
    active.add_argument("--corpus", default="Raw.txt")
    active.add_argument("--model", default="variant_classifier.pkl")
    active.add_argument("--output", default="active_learning_candidates.csv")
    active.add_argument("--sample-size", type=int, default=1000)
    active.add_argument("--pool-limit", type=int)
    active.add_argument("--random-state", type=int, default=42)

    run = commands.add_parser("run", help="Run bounded full-corpus prediction and export")
    run.add_argument("--corpus", default="Raw.txt")
    run.add_argument("--model", default="variant_classifier.pkl")
    run.add_argument("--output-csv", default="Variants.csv")
    run.add_argument("--evaluation-xlsx", default="Evaluation.xlsx")
    run.add_argument("--candidate-limit", type=int, default=100000)
    run.add_argument("--random-state", type=int, default=42)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate manually reviewed final output pairs"
    )
    evaluate.add_argument("--evaluation-xlsx", default="Evaluation.xlsx")
    evaluate.add_argument("--corpus", default="Raw.txt")
    evaluate.add_argument("--model", default="variant_classifier.pkl")
    evaluate.add_argument("--json-output", default="Final_Evaluation.json")
    evaluate.add_argument("--text-output", default="Final_Evaluation.txt")

    test_sample = commands.add_parser(
        "test-sample", help="Create a hidden-prediction independent Raw.txt test sample"
    )
    test_sample.add_argument("--corpus", default="Raw.txt")
    test_sample.add_argument("--model", default="variant_classifier.pkl")
    test_sample.add_argument("--annotations", nargs="+", required=True)
    test_sample.add_argument("--output", default="Test_Dataset.csv")
    test_sample.add_argument("--sample-size", type=int, default=1000)
    test_sample.add_argument("--candidate-limit", type=int, default=100000)
    test_sample.add_argument("--random-state", type=int, default=42)
    test_sample.add_argument("--overwrite", action="store_true")

    test = commands.add_parser(
        "test", help="Evaluate manually labeled independent test candidates"
    )
    test.add_argument("--corpus", default="Raw.txt")
    test.add_argument("--model", default="variant_classifier.pkl")
    test.add_argument("--test-data", required=True)
    test.add_argument("--annotations", nargs="*", default=[])
    test.add_argument("--json-output", default="Final_Evaluation.json")
    test.add_argument("--text-output", default="Final_Evaluation.txt")

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_annotation_candidates(
            args.corpus, args.output, args.sample_size, args.pool_limit,
            args.random_state, args.cache,
        )
    elif args.command == "validate":
        merged = merge_annotations(args.annotations)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"validated_rows={len(merged)} reviewed={(merged['is_variant'] != '').sum()}")
    elif args.command == "train":
        train_from_annotations(
            args.annotations, args.model, args.n_estimators,
            args.n_splits, args.random_state,
        )
    elif args.command == "active":
        create_active_learning_sample(
            args.corpus, args.model, args.output, args.sample_size,
            args.pool_limit, args.random_state,
        )
    elif args.command == "run":
        run_full_pipeline(
            args.corpus, args.model, args.output_csv, args.evaluation_xlsx,
            args.candidate_limit, args.random_state,
        )
    elif args.command == "evaluate":
        evaluate_final_output(
            args.evaluation_xlsx, args.corpus, args.model,
            args.json_output, args.text_output,
        )
    elif args.command == "test-sample":
        create_independent_test_sample(
            args.corpus, args.model, args.annotations, args.output,
            args.sample_size, args.candidate_limit, args.random_state,
            args.overwrite,
        )
    elif args.command == "test":
        evaluate_independent_test(
            args.corpus, args.model, args.test_data, args.annotations,
            args.json_output, args.text_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
