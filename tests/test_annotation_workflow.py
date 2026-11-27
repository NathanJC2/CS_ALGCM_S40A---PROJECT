from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "annotation_workflow_test", ROOT / "annotation_workflow.py"
)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def annotation_row(word1: str, word2: str, label: str = "") -> dict[str, object]:
    return {
        "word1": word1,
        "word2": word2,
        "word1_count": 8,
        "word2_count": 3,
        "character_difference": "o vs. u",
        "edit_distance": 1,
        "context_similarity": 0.6,
        "word1_language": "FIL",
        "word2_language": "FIL",
        "word1_context_examples": f"{word1} was in this context.",
        "word2_context_examples": f"{word2} was in this context.",
        "is_variant": label,
    }


class AnnotationWorkflowTests(unittest.TestCase):
    def test_candidate_pool_includes_canonical_variants_without_self_pairs(self):
        pairs = workflow.generate_candidate_pool(
            {"politiko": 5, "pulitiko": 3, "opo": 2, "upo": 4},
            pool_limit=100,
            random_state=4,
        )
        self.assertIn(("politiko", "pulitiko"), pairs)
        self.assertIn(("opo", "upo"), pairs)
        self.assertTrue(all(left != right for left, right in pairs))

    def test_labels_canonicalize_reversed_pairs_and_detect_conflicts(self):
        columns = list(workflow.ANNOTATION_COLUMNS)
        frame = pd.DataFrame([
            annotation_row("pulitiko", "politiko", "1"),
            annotation_row("politiko", "pulitiko", ""),
        ], columns=columns)
        result = workflow.validate_annotations(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["is_variant"], "1")
        self.assertEqual(result.iloc[0]["word1"], "politiko")

        conflict = pd.DataFrame([
            annotation_row("one", "two", "1"),
            annotation_row("two", "one", "0"),
        ], columns=columns)
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            workflow.validate_annotations(conflict)

    def test_invalid_labels_are_rejected_and_blanks_excluded_from_training(self):
        columns = list(workflow.ANNOTATION_COLUMNS)
        invalid = pd.DataFrame([annotation_row("a", "b", "yes")], columns=columns)
        with self.assertRaisesRegex(ValueError, "Invalid is_variant"):
            workflow.validate_annotations(invalid)
        frame = pd.DataFrame([
            annotation_row("politiko", "pulitiko", "1"),
            annotation_row("opo", "upo", "0"),
            annotation_row("lalaki", "lalake", ""),
        ], columns=columns)
        checked = workflow.validate_annotations(frame)
        self.assertEqual((checked["is_variant"] != "").sum(), 2)

    def test_annotation_features_are_numeric_and_context_sensitive(self):
        frame = pd.DataFrame([
            annotation_row("politiko", "pulitiko", ""),
            {**annotation_row("opo", "upo", ""),
             "context_similarity": 0.05, "word1_language": "FIL",
             "word2_language": "FIL"},
        ])
        features = workflow.annotation_features(frame)
        self.assertEqual(list(features.columns), list(workflow._load_module(
            "05_feature_extractor.py", "test_annotation_feature_columns"
        ).FEATURE_COLUMNS))
        self.assertEqual(features.iloc[0]["context_cosine"], 0.6)
        self.assertEqual(features.iloc[1]["context_cosine"], 0.05)

    def test_training_uses_only_verified_rows_and_saves_oof_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            rows = [
                annotation_row("politiko", "pulitiko", "1"),
                annotation_row("lalaki", "lalake", "1"),
                annotation_row("marami", "madami", "1"),
                annotation_row("opo", "upo", "0"),
                annotation_row("bata", "bota", "0"),
                annotation_row("sako", "sakoa", "0"),
                annotation_row("ito", "iyon", ""),
            ]
            pd.DataFrame(rows, columns=workflow.ANNOTATION_COLUMNS).to_csv(
                path, index=False
            )
            model_path = Path(directory) / "model.pkl"
            result = workflow.train_from_annotations(
                [path], model_path, n_estimators=10, n_splits=2
            )
            self.assertEqual(result["verified_rows"], 6)
            self.assertTrue(model_path.is_file())
            self.assertIn("oof_precision", result)

    def test_candidate_csv_and_active_sample_use_real_contexts(self):
        original_loader = workflow._load_module

        def fake_pinoy_loader(filename: str, module_name: str):
            if filename == "pinoybot.py":
                return types.SimpleNamespace(
                    predict_language=lambda words: (
                        ["FIL"] * len(words),
                        [[1.0, 0.0] for _ in words],
                        ["FIL", "ENG"],
                    )
                )
            return original_loader(filename, module_name)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            corpus.write_text(
                ("Ang politiko ay dumating sa bayan. Ang pulitiko ay nagsalita. "
                 "Ang politika at pulitika ay mahalaga. Marami madami ang dumalo. "
                 "Lalaki at lalake ang dumating. Opo ang sagot, upo sa upuan. "
                 "Bato at boto ang halimbawa. " * 8),
                encoding="utf-8",
            )
            output = root / "annotation_candidates.csv"
            with unittest.mock.patch.object(
                workflow, "_load_module", side_effect=fake_pinoy_loader
            ):
                count = workflow.generate_annotation_candidates(
                    corpus, output, sample_size=8, pool_limit=80, random_state=2,
                    cache_path=root / "counts.pkl",
                )
            self.assertGreater(count, 0)
            generated = pd.read_csv(output, keep_default_na=False)
            self.assertEqual(list(generated.columns), list(workflow.ANNOTATION_COLUMNS))
            self.assertTrue((generated["is_variant"] == "").all())
            text = corpus.read_text(encoding="utf-8")
            examples = (
                generated["word1_context_examples"].tolist()
                + generated["word2_context_examples"].tolist()
            )
            normalized_text = " ".join(text.split())
            self.assertTrue(any(
                snippet and " ".join(snippet.split()) in normalized_text
                for example in examples
                for snippet in example.split(" || ")
            ))

            generated["is_variant"] = [
                str(index % 2) for index in range(len(generated))
            ]
            labels_path = root / "labeled.csv"
            generated.to_csv(labels_path, index=False)
            model_path = root / "variant_model.pkl"
            workflow.train_from_annotations(
                [labels_path], model_path, n_estimators=10, n_splits=2
            )
            active_path = root / "active.csv"
            with mock.patch.object(
                workflow, "_load_module", side_effect=fake_pinoy_loader
            ):
                active_count = workflow.create_active_learning_sample(
                    corpus, model_path, active_path, sample_size=3,
                    pool_limit=80, random_state=2,
                )
            self.assertGreater(active_count, 0)
            active = pd.read_csv(active_path, keep_default_na=False)
            self.assertTrue((active["is_variant"] == "").all())


if __name__ == "__main__":
    unittest.main()
