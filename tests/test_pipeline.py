from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


def load(name: str):
    path = Path(__file__).parents[1] / name
    spec = spec_from_file_location(name.replace(".", "_"), path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def test_preprocessing_counts_unicode_and_markup(self):
        module = load("01_preprocess_vocab.py")
        counts, sentences = module.read_corpus([
            tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
        ]) if False else (Counter(), [])
        text = "<token>pulitiko</token>, Pulitiko! política"
        tokens = module.tokenize_text(text)
        self.assertIn("pulitiko", tokens)
        self.assertIn("política", tokens)
        self.assertEqual(module.tokenize_sentences("One word. Two words!"),
                         [["one", "word"], ["two", "words"]])

    def test_edit_distance_and_character_difference(self):
        module = load("03_dp_edit_distance.py")
        result = module.analyze_edit_distance("politiko", "pulitiko")
        self.assertEqual(result.distance, 1)
        self.assertIn("o vs. u", result.character_difference)
        self.assertEqual(module.analyze_edit_distance("cat", "cut").distance, 1)

    def test_candidate_generation_includes_known_variant(self):
        module = load("02_candidate_blocking_dnc.py")
        candidates = module.generate_candidates(
            ["politiko", "pulitiko", "opo", "upo", "lalaki", "lalake"]
        )
        self.assertIn(("politiko", "pulitiko"), candidates)
        self.assertIn(("lalake", "lalaki"), candidates)

    def test_batched_candidate_generation_matches_full_blocker(self):
        module = load("02_candidate_blocking_dnc.py")
        words = [
            "politiko", "pulitiko", "opo", "upo", "lalaki", "lalake",
            "bato", "boto", "pamilya", "pamiliya", "salamat",
        ]
        expected = module.generate_candidates(
            words, max_length_diff=2, leaf_size=3, max_pairs_per_signature=20
        )
        batches = list(module.iter_candidate_batches(
            words,
            batch_size=3,
            max_length_diff=2,
            leaf_size=3,
            max_pairs_per_signature=20,
        ))
        actual = [pair for batch in batches for pair in batch]
        self.assertEqual(actual, expected)
        self.assertTrue(all(len(batch) <= 3 for batch in batches))

    def test_batched_blocker_has_no_global_one_hundred_thousand_pair_cap(self):
        module = load("02_candidate_blocking_dnc.py")
        words = [f"abc{index:03d}" for index in range(500)]
        total = 0
        largest_batch = 0
        for batch in module.iter_candidate_batches(
            words,
            batch_size=4096,
            max_length_diff=0,
            leaf_size=32,
            max_pairs_per_signature=5000,
        ):
            total += len(batch)
            largest_batch = max(largest_batch, len(batch))
        self.assertGreater(total, 100000)
        self.assertLessEqual(largest_batch, 4096)

    def test_export_deduplicates_and_formats_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            module = load("08_export_variants_csv.py")
            output = Path(directory) / "Variants.csv"
            count = module.export_variants(
                [("pulitiko", "politiko", 0.9),
                 ("politiko", "pulitiko", 0.8),
                 ("opo", "opo", 1.0)],
                {"politiko": 435, "pulitiko": 1093},
                output,
            )
            self.assertEqual(count, 1)
            with output.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]),
                             ["word1", "word2", "word1_count",
                              "word2_count", "character_difference"])
            self.assertEqual(rows[0]["character_difference"], "o vs. u")

    def test_greedy_resolver_requires_direct_support(self):
        module = load("07_greedy_cluster_resolver.py")
        resolved = module.resolve_variant_pairs([
            ("alpha", "alfa", 0.95),
            ("alfa", "omega", 0.88),
            ("alpha", "omega", 0.20),
        ])
        self.assertEqual(len(resolved), 1)

    def test_evaluation_sample_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "variants.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["word1", "word2", "word1_count",
                                 "word2_count", "character_difference"])
                for index in range(120):
                    writer.writerow([f"a{index}", f"b{index}", 1, 1, "a vs. b"])
            module = load("09_sampler_for_evaluation.py")
            first = module.create_evaluation_sample(
                csv_path, directory / "first.xlsx", 100, 42
            )
            second = module.create_evaluation_sample(
                csv_path, directory / "second.xlsx", 100, 42
            )
            self.assertEqual(first, 100)
            self.assertEqual(second, 100)


if __name__ == "__main__":
    unittest.main()
