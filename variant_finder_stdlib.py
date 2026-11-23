#!/usr/bin/env python3
"""
variant_finder.py - find spelling variants of Filipino words in a CSV file.

Pipeline
--------
1. Read the CSV (chosen columns), strip markup such as <token>...</token>,
   normalize, and tokenize.
2. Build a vocabulary with word counts.
3. Generate candidate pairs:
     - "key"   : words that become identical after collapsing known Filipino
                 spelling alternations (o/u, e/i, d/r, c/k, f/p, ...).
     - "edit1" : (optional) words differing by exactly one substituted letter.
4. Score every pair by context similarity (do the two words appear next to
   the same neighbouring words?). This separates true variants such as
   politiko/pulitiko from distinct words such as boto/buto.
5. Write variants.csv (one row per pair) and clusters.csv (one row per group).

Standard library only. Python 3.8+.

(This is the dependency-free version. variant_finder.py is the RapidFuzz version.)

Example
-------
    python variant_finder.py input.csv --column text
    python variant_finder.py input.csv --column 0 --no-header --edit1
    python variant_finder.py input.csv --distinct-pairs distinct_pairs.txt \
        --wordlist tagalog_words.txt
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION - edit these to suit your data
# ---------------------------------------------------------------------------

# Spelling alternations collapsed when building a word's "key".
# Rules are applied in order, so multi-character rules go first.
# Remove a rule if it produces too many false positives; add your own as needed.
KEY_RULES = [
    ("ph", "f"),
    ("qu", "k"),
    ("f", "p"),   # Filipinas / Pilipinas
    ("c", "k"),   # politica / politika
    ("o", "u"),   # politiko / pulitiko
    ("e", "i"),   # lalake / lalaki
    ("d", "r"),   # marami / madami
]

# Patterns removed before tokenizing. Add more for other markup styles,
# e.g. re.compile(r"\[/?token\]") for [token]...[/token].
STRIP_PATTERNS = [
    re.compile(r"</?[A-Za-z][^>]*>"),         # <token>, </token>, <token id="3">
    re.compile(r"https?://\S+|www\.\S+"),     # URLs
    re.compile(r"\S+@\S+\.\S+"),              # e-mail addresses
]

# A word = letters, optionally joined by hyphens or apostrophes (mag-aral).
WORD_RE = re.compile(r"[^\W\d_]+(?:[-'\u2019][^\W\d_]+)*")

STATUS_ORDER = {"likely_variant": 0, "unverified": 1, "review": 2}


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Unescape HTML entities, strip markup, normalize Unicode, casefold."""
    text = html.unescape(str(raw))
    for pattern in STRIP_PATTERNS:
        text = pattern.sub(" ", text)
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


@lru_cache(maxsize=None)
def make_key(word: str) -> str:
    """Collapse known alternations so variants share one key."""
    w = re.sub(r"[-'\u2019]", "", word)
    for old, new in KEY_RULES:
        w = w.replace(old, new)
    return w


def char_diff(a: str, b: str) -> str:
    """Describe the differences between two words, e.g. 'o vs. u'.

    Uses a standard Levenshtein alignment (pure Python, no dependencies) and
    lists each differing character separately ('c vs. k, a vs. o'). A missing
    character is shown as the empty-set sign.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,
                           dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((a[i - 1], ""))
            i -= 1
        else:
            ops.append(("", b[j - 1]))
            j -= 1
    ops.reverse()

    diffs = [f"{x or '∅'} vs. {y or '∅'}" for x, y in ops if x != y]
    return ", ".join(diffs)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def resolve_columns(spec: str, header: list[str] | None) -> list[int] | None:
    """Turn '--column' into a list of indices (None means all columns)."""
    if spec.strip().lower() == "all":
        return None
    indices = []
    for item in (s.strip() for s in spec.split(",")):
        if header and item in header:
            indices.append(header.index(item))
        elif item.isdigit():
            indices.append(int(item))
        else:
            sys.exit(f"error: column '{item}' not found in header {header}")
    return indices


def read_documents(path: str, column_spec: str, has_header: bool, encoding: str):
    """Return a list of (tokens, display_text), one entry per non-empty cell."""
    csv.field_size_limit(2 ** 31 - 1)
    docs = []
    with open(path, newline="", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None) if has_header else None
        columns = resolve_columns(column_spec, header)
        for row in reader:
            cells = row if columns is None else [row[i] for i in columns if i < len(row)]
            for cell in cells:
                if not cell or not cell.strip():
                    continue
                text = clean_text(cell)
                tokens = tokenize(text)
                if tokens:
                    docs.append((tokens, " ".join(text.split())))
    return docs


def load_pairs_file(path: str | None) -> set[frozenset]:
    """Read 'word1,word2' lines (comma, tab or space separated; # = comment)."""
    pairs: set[frozenset] = set()
    if not path:
        return pairs
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        parts = [p for p in re.split(r"[,\t ]+", line) if p]
        if len(parts) == 2:
            pairs.add(frozenset(clean_text(p) for p in parts))
    return pairs


def load_wordlist(path: str | None) -> set[str] | None:
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return {clean_text(w).strip() for w in lines if w.strip()}


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def order_pair(a: str, b: str, vocab: Counter) -> tuple[str, str]:
    """Less frequent word first (ties broken alphabetically)."""
    return tuple(sorted((a, b), key=lambda w: (vocab[w], w)))  # type: ignore


def key_candidates(vocab: Counter) -> dict[tuple[str, str], str]:
    groups = defaultdict(list)
    for w in vocab:
        groups[make_key(w)].append(w)
    pairs = {}
    for members in groups.values():
        for a, b in combinations(members, 2):
            pairs[order_pair(a, b, vocab)] = "key"
    return pairs


def edit1_candidates(vocab: Counter, min_len: int, existing: set) -> dict[tuple[str, str], str]:
    """Words that differ by exactly one substituted letter.

    Uses wildcard buckets ('bato' -> '*ato', 'b*to', ...) so no all-pairs
    comparison is needed.
    """
    buckets = defaultdict(list)
    for w in vocab:
        if len(w) < min_len:
            continue
        for i in range(len(w)):
            buckets[w[:i] + "*" + w[i + 1:]].append(w)
    pairs = {}
    for members in buckets.values():
        for a, b in combinations(members, 2):
            p = order_pair(a, b, vocab)
            if p not in existing:
                pairs[p] = "edit1"
    return pairs


# ---------------------------------------------------------------------------
# Context similarity
# ---------------------------------------------------------------------------

def build_context(docs, targets: set[str], window: int, stop_keys: set[str]):
    """For each target word, count the (keyed) neighbours within +/- window."""
    ctx: dict[str, Counter] = defaultdict(Counter)
    example: dict[str, str] = {}
    for tokens, text in docs:
        for i, tok in enumerate(tokens):
            if tok not in targets:
                continue
            example.setdefault(tok, text)
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                k = make_key(tokens[j])
                if k not in stop_keys:
                    ctx[tok][k] += 1
    return ctx, example


def weighted_cosine(a: Counter, b: Counter, weight: dict[str, float]) -> float | None:
    if not a or not b:
        return None
    dot = sum(a[k] * b[k] * weight[k] ** 2 for k in a.keys() & b.keys())
    na = math.sqrt(sum((v * weight[k]) ** 2 for k, v in a.items()))
    nb = math.sqrt(sum((v * weight[k]) ** 2 for k, v in b.items()))
    return dot / (na * nb) if na and nb else None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Find spelling variants of Filipino words in a CSV.")
    p.add_argument("input", help="input CSV file")
    p.add_argument("--column", default="0",
                   help="column name or 0-based index; comma-separate several, or 'all' (default: 0)")
    p.add_argument("--no-header", action="store_true", help="CSV has no header row")
    p.add_argument("--encoding", default="utf-8-sig", help="input encoding (default: utf-8-sig)")
    p.add_argument("--out-dir", default="output", help="output folder (default: output)")
    p.add_argument("--min-count", type=int, default=1, help="ignore words seen fewer times (default: 1)")
    p.add_argument("--min-length", type=int, default=3, help="ignore words shorter than this (default: 3)")
    p.add_argument("--edit1", action="store_true",
                   help="also look for one-letter-substitution pairs the key rules miss "
                        "(kept only when context similarity supports them)")
    p.add_argument("--edit1-min-length", type=int, default=4,
                   help="minimum word length for --edit1 candidates (default: 4)")
    p.add_argument("--window", type=int, default=2, help="context window size on each side (default: 2)")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="context similarity needed for 'likely_variant' (default: 0.25)")
    p.add_argument("--min-context", type=int, default=3,
                   help="min context words per word before similarity is trusted (default: 3)")
    p.add_argument("--stopwords-top", type=int, default=25,
                   help="ignore this many of the most frequent words as context; capped at 1%% "
                        "of the vocabulary (default: 25)")
    p.add_argument("--distinct-pairs", help="text file of known-distinct pairs to exclude (e.g. 'boto,buto')")
    p.add_argument("--wordlist", help="dictionary file (one word per line); adds a both_in_wordlist column")
    return p.parse_args()


def main():
    args = parse_args()

    # 1-2. Read, clean, tokenize, count
    docs = read_documents(args.input, args.column, not args.no_header, args.encoding)
    if not docs:
        sys.exit("error: no text found. Check --column / --no-header.")
    raw_vocab = Counter(t for tokens, _ in docs for t in tokens)
    vocab = Counter({w: c for w, c in raw_vocab.items()
                     if c >= args.min_count and len(w) >= args.min_length})
    print(f"Read {len(docs):,} text cells, {sum(raw_vocab.values()):,} tokens, "
          f"{len(vocab):,} unique words after filtering.")

    # 3. Candidate pairs
    candidates = key_candidates(vocab)
    if args.edit1:
        candidates.update(edit1_candidates(vocab, args.edit1_min_length, set(candidates)))

    distinct = load_pairs_file(args.distinct_pairs)
    candidates = {p: r for p, r in candidates.items() if frozenset(p) not in distinct}
    print(f"{len(candidates):,} candidate pairs "
          f"({sum(r == 'key' for r in candidates.values()):,} from key rules).")

    # 4. Context similarity
    key_freq = Counter(make_key(t) for tokens, _ in docs for t in tokens)
    total = sum(key_freq.values())
    n_stop = min(args.stopwords_top, len(key_freq) // 100)
    stop_keys = {k for k, _ in key_freq.most_common(n_stop)}
    weight = {k: math.log(total / c) + 1e-6 for k, c in key_freq.items()}  # idf-style

    targets = {w for pair in candidates for w in pair}
    ctx, example = build_context(docs, targets, args.window, stop_keys)
    wordlist = load_wordlist(args.wordlist)

    rows = []
    for (w1, w2), rule in candidates.items():
        c1, c2 = ctx.get(w1, Counter()), ctx.get(w2, Counter())
        sim = weighted_cosine(c1, c2, weight)
        enough = min(sum(c1.values()), sum(c2.values())) >= args.min_context
        if sim is None or not enough:
            status = "unverified"
        elif sim >= args.threshold:
            status = "likely_variant"
        else:
            status = "review"
        if rule == "edit1" and status != "likely_variant":
            continue  # edit1 pairs need context evidence to be reported
        both = ""
        if wordlist is not None:
            both = "yes" if (w1 in wordlist and w2 in wordlist) else "no"
        rows.append({
            "word1": w1,
            "word2": w2,
            "word1_count": vocab[w1],
            "word2_count": vocab[w2],
            "character_difference": char_diff(w1, w2),
            "rule": rule,
            "context_similarity": "" if sim is None else round(sim, 3),
            "both_in_wordlist": both,
            "status": status,
            "example_word1": example.get(w1, "")[:120],
            "example_word2": example.get(w2, "")[:120],
        })

    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], -(r["word1_count"] + r["word2_count"]), r["word1"]))

    # 5. Write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_fields = ["word1", "word2", "word1_count", "word2_count", "character_difference",
                   "rule", "context_similarity", "both_in_wordlist", "status",
                   "example_word1", "example_word2"]
    with open(out_dir / "variants.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(rows)

    # Clusters use every pair except 'review' ones, so lookalikes such as
    # boto/buto are not merged into a group.
    uf = UnionFind()
    for r in rows:
        if r["status"] != "review":
            uf.union(r["word1"], r["word2"])
    members = defaultdict(list)
    for w in uf.parent:
        members[uf.find(w)].append(w)
    clusters = []
    for ws in members.values():
        ws.sort(key=lambda w: (-vocab[w], w))
        clusters.append({
            "canonical": ws[0],
            "canonical_count": vocab[ws[0]],
            "variants": "; ".join(f"{w} ({vocab[w]})" for w in ws[1:]),
            "cluster_size": len(ws),
            "total_count": sum(vocab[w] for w in ws),
        })
    clusters.sort(key=lambda c: -c["total_count"])
    with open(out_dir / "clusters.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical", "canonical_count", "variants",
                                               "cluster_size", "total_count"])
        writer.writeheader()
        writer.writerows(clusters)

    counts = Counter(r["status"] for r in rows)
    print(f"Wrote {len(rows):,} pairs to {out_dir / 'variants.csv'} "
          f"(likely_variant: {counts['likely_variant']}, unverified: {counts['unverified']}, "
          f"review: {counts['review']})")
    print(f"Wrote {len(clusters):,} clusters to {out_dir / 'clusters.csv'}")


if __name__ == "__main__":
    main()
