from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


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


class FixedProbabilityClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, predicted: list[int]) -> None:
        self.predicted = predicted

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        selected = self.predicted[:len(features)]
        positive = np.array([0.9 if value else 0.1 for value in selected])
        return np.column_stack((1 - positive, positive))


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
            with model_path.open("rb") as stream:
                artifact = workflow.pickle.load(stream)
            self.assertEqual(artifact["sklearn_version"], sklearn.__version__)
            self.assertEqual(len(artifact["training_pairs"]), 6)
            self.assertTrue(artifact["training_data_fingerprint"])

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

    def test_final_precision_requires_all_100_and_strictly_exceeds_70_percent(self):
        artifact = {
            "threshold": 0.9,
            "verified_rows": 10,
            "evaluation": {
                "method": "test OOF",
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.84,
                "confusion_matrix": [[5, 0], [1, 4]],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "Evaluation.xlsx"
            rows = [
                {**annotation_row(f"w{index}", f"x{index}"),
                 "is_correct": index < 70}
                for index in range(100)
            ]
            pd.DataFrame(rows).to_excel(workbook_path, index=False)
            with mock.patch.object(
                workflow, "_load_variant_artifact", return_value=artifact
            ):
                exactly_70 = workflow.evaluate_final_output(
                    workbook_path,
                    model_path=root / "model.pkl",
                    json_path=root / "report.json",
                    text_path=root / "report.txt",
                )
                self.assertEqual(exactly_70["output_precision"], 0.7)
                self.assertEqual(exactly_70["status"], "FAIL")

                rows[70]["is_correct"] = True
                pd.DataFrame(rows).to_excel(workbook_path, index=False)
                above_70 = workflow.evaluate_final_output(
                    workbook_path,
                    model_path=root / "model.pkl",
                    json_path=root / "report.json",
                    text_path=root / "report.txt",
                )
                self.assertEqual(above_70["output_precision"], 0.71)
                self.assertEqual(above_70["status"], "PASS")

    def test_final_precision_counts_blanks_as_unreviewed_and_rejects_bad_labels(self):
        artifact = {"threshold": 0.5}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "Evaluation.xlsx"
            rows = [
                {**annotation_row(f"a{index}", f"b{index}"),
                 "is_correct": True if index < 99 else ""}
                for index in range(100)
            ]
            pd.DataFrame(rows).to_excel(workbook_path, index=False)
            with mock.patch.object(
                workflow, "_load_variant_artifact", return_value=artifact
            ):
                result = workflow.evaluate_final_output(
                    workbook_path,
                    model_path=root / "model.pkl",
                    json_path=root / "report.json",
                    text_path=root / "report.txt",
                )
            self.assertEqual(result["correct_variants"], 99)
            self.assertEqual(result["unreviewed_pairs"], 1)
            self.assertEqual(result["status"], "INCOMPLETE")
            with self.assertRaisesRegex(ValueError, "Invalid is_correct"):
                workflow._parse_binary_value("maybe", "is_correct")

    def test_independent_test_sample_is_reproducible_hidden_and_proportionate(self):
        artifact = {
            "threshold": 0.5,
            "training_pairs": [("known", "knownword")],
        }
        rows = [
            annotation_row(f"left{index}", f"right{index}")
            for index in range(12)
        ]
        frame = pd.DataFrame(rows, columns=workflow.ANNOTATION_COLUMNS)
        probabilities = np.array([0.9] * 6 + [0.1] * 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            corpus.write_text("A small corpus.", encoding="utf-8")
            model_path = root / "model.pkl"
            model_path.write_bytes(b"fixed model bytes")
            with (
                mock.patch.object(
                    workflow, "_load_variant_artifact", return_value=artifact
                ),
                mock.patch.object(
                    workflow,
                    "_candidate_frame_for_model",
                    return_value=(frame, probabilities, workflow.Counter(), 12),
                ) as candidate_builder,
            ):
                first_path = root / "test-one.csv"
                second_path = root / "test-two.csv"
                first_meta = workflow.create_independent_test_sample(
                    corpus, model_path, [], first_path, 4, 12, 17
                )
                workflow.create_independent_test_sample(
                    corpus, model_path, [], second_path, 4, 12, 17
                )
            sample_call = candidate_builder.call_args
            self.assertIn(("known", "knownword"), sample_call.args[4])
            self.assertIn(workflow._canonical_word("known"), sample_call.args[5])
            self.assertIn(workflow._canonical_word("knownword"), sample_call.args[5])
            first = pd.read_csv(first_path, keep_default_na=False)
            second = pd.read_csv(second_path, keep_default_na=False)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(list(first.columns), list(workflow.ANNOTATION_COLUMNS))
            self.assertTrue((first["is_variant"] == "").all())
            self.assertEqual(first_meta["sampled_predicted_positive"], 2)
            self.assertEqual(first_meta["sampled_predicted_negative"], 2)
            self.assertEqual(first_meta["training_pair_overlap"], 0)
            self.assertFalse(any("predict" in column for column in first.columns))

    def _write_independent_test_fixture(
        self,
        root: Path,
        labels: list[str],
        predicted: list[int],
    ) -> tuple[Path, Path, Path, dict[str, object]]:
        corpus = root / "corpus.txt"
        corpus.write_text("A minimal test corpus.", encoding="utf-8")
        model = root / "model.pkl"
        model.write_bytes(b"frozen model bytes")
        test_data = root / "Test_Dataset.csv"
        frame = pd.DataFrame(
            [
                annotation_row(f"testleft{index}", f"testright{index}", label)
                for index, label in enumerate(labels)
            ],
            columns=workflow.ANNOTATION_COLUMNS,
        )
        frame.to_csv(test_data, index=False)
        artifact: dict[str, object] = {
            "classifier": FixedProbabilityClassifier(predicted),
            "feature_columns": [],
            "threshold": 0.5,
            "training_pairs": [("trainingword", "trainingform")],
            "verified_rows": 4,
            "evaluation": {
                "method": "test OOF",
                "accuracy": 0.75,
                "precision": 0.75,
                "recall": 0.75,
                "f1": 0.75,
                "confusion_matrix": [[2, 1], [0, 1]],
            },
        }
        metadata = {
            "independent_test_dataset": True,
            "sample_size": len(frame),
            "model_sha256": workflow._sha256_file(model),
            "corpus": list(workflow._file_metadata(corpus)),
            "threshold": 0.5,
            "test_dataset_path": str(test_data.resolve()),
            "test_pairs": [
                list(workflow.canonical_pair(row.word1, row.word2))
                for row in frame.itertuples(index=False)
            ],
        }
        workflow._test_metadata_path(test_data).write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return corpus, model, test_data, artifact

    def test_independent_metrics_confusion_baseline_and_strict_accuracy(self):
        predicted = [1] * 7 + [0] * 3
        exact_70_labels = ["1", "1", "1", "1", "1", "0", "0", "1", "0", "0"]
        above_70_labels = ["1", "1", "1", "1", "1", "1", "0", "0", "0", "0"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for labels, expected_status, expected_accuracy, expected_matrix in (
                (exact_70_labels, "FAIL", 0.7, [[2, 2], [1, 5]]),
                (above_70_labels, "PASS", 0.9, [[3, 1], [0, 6]]),
            ):
                case = root / expected_status
                case.mkdir()
                corpus, model, test_data, artifact = self._write_independent_test_fixture(
                    case, labels, predicted
                )
                with (
                    mock.patch.object(
                        workflow, "_load_variant_artifact", return_value=artifact
                    ),
                    mock.patch.object(
                        workflow,
                        "annotation_features",
                        side_effect=lambda frame: pd.DataFrame(index=range(len(frame))),
                    ),
                ):
                    metrics = workflow.evaluate_independent_test(
                        corpus,
                        model,
                        test_data,
                        json_path=case / "report.json",
                        text_path=case / "report.txt",
                    )
                self.assertEqual(metrics["status"], expected_status)
                self.assertEqual(metrics["accuracy"], expected_accuracy)
                self.assertEqual(metrics["training_pair_overlap"], 0)
                self.assertEqual(metrics["confusion_matrix"], expected_matrix)
                self.assertEqual(metrics["majority_class_baseline_accuracy"], 0.6)

    def test_independent_test_incomplete_and_training_overlap_guard(self):
        labels = ["1", "0", "1", "0", "1", "0", "1", "0", "1", ""]
        predicted = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, model, test_data, artifact = self._write_independent_test_fixture(
                root, labels, predicted
            )
            with (
                mock.patch.object(
                    workflow, "_load_variant_artifact", return_value=artifact
                ),
                mock.patch.object(
                    workflow,
                    "annotation_features",
                    side_effect=lambda frame: pd.DataFrame(index=range(len(frame))),
                ),
            ):
                metrics = workflow.evaluate_independent_test(
                    corpus,
                    model,
                    test_data,
                    json_path=root / "report.json",
                    text_path=root / "report.txt",
                )
            self.assertEqual(metrics["evaluated_samples"], 9)
            self.assertEqual(metrics["unreviewed_samples"], 1)
            self.assertEqual(metrics["status"], "INCOMPLETE")

            training_csv = root / "training.csv"
            pd.DataFrame(
                [annotation_row("pairb", "paira", "1")],
                columns=workflow.ANNOTATION_COLUMNS,
            ).to_csv(training_csv, index=False)
            with mock.patch.object(
                workflow,
                "_known_independent_test_pairs",
                return_value={("paira", "pairb")},
            ):
                with self.assertRaisesRegex(ValueError, "independent test dataset"):
                    workflow._reject_independent_test_training([training_csv])

    def test_independent_test_rejects_a_changed_frozen_model(self):
        labels = ["1", "0"]
        predicted = [1, 0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, model, test_data, artifact = self._write_independent_test_fixture(
                root, labels, predicted
            )
            model.write_bytes(b"changed model bytes")
            with mock.patch.object(
                workflow, "_load_variant_artifact", return_value=artifact
            ):
                with self.assertRaisesRegex(RuntimeError, "model changed"):
                    workflow.evaluate_independent_test(
                        corpus,
                        model,
                        test_data,
                        json_path=root / "report.json",
                        text_path=root / "report.txt",
                    )

    def test_full_corpus_test_batches_predictions_and_scores_external_truth(self):
        candidate_pairs = [
            ("politiko", "pulitiko"),
            ("bato", "boto"),
            ("trained", "training"),
            ("trained", "trainish"),
        ]
        artifact = {
            "classifier": FixedProbabilityClassifier([1, 0, 1, 0]),
            "feature_columns": ["test_feature"],
            "threshold": 0.5,
            "training_pairs": [("trained", "training")],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "Raw.txt"
            corpus.write_text("A small spelling test corpus.", encoding="utf-8")
            model = root / "variant_classifier.pkl"
            model.write_bytes(b"frozen model bytes")
            training_csv = root / "training.csv"
            pd.DataFrame(
                [annotation_row("trained", "training", "1")],
                columns=workflow.ANNOTATION_COLUMNS,
            ).to_csv(training_csv, index=False)
            ground_truth = root / "ground_truth.csv"
            pd.DataFrame([
                ("politiko", "pulitiko", "1"),
                ("bato", "boto", "0"),
                ("missed", "missform", "1"),
                ("trained", "training", "0"),
                ("trained", "trainish", "1"),
            ], columns=["word1", "word2", "is_variant"]).to_csv(
                ground_truth, index=False
            )
            frequencies = workflow.Counter({
                "politiko": 5, "pulitiko": 4,
                "bato": 3, "boto": 2,
                "trained": 2, "training": 1, "trainish": 1,
                "missed": 1, "missform": 1,
            })
            original_loader = workflow._load_module

            def fake_loader(filename: str, module_name: str):
                if filename == "02_candidate_blocking_dnc.py":
                    return types.SimpleNamespace(
                        iter_candidate_batches=lambda *args, **kwargs: iter(
                            [candidate_pairs[:2], candidate_pairs[2:]]
                        )
                    )
                return original_loader(filename, module_name)

            with (
                mock.patch.object(
                    workflow, "_load_variant_artifact", return_value=artifact
                ),
                mock.patch.object(
                    workflow, "corpus_counts", return_value=frequencies
                ),
                mock.patch.object(
                    workflow,
                    "_predict_languages",
                    return_value={word: "FIL" for word in frequencies},
                ),
                mock.patch.object(
                    workflow,
                    "_collect_pair_contexts",
                    return_value=({}, {}),
                ),
                mock.patch.object(
                    workflow,
                    "annotation_features",
                    side_effect=lambda frame: pd.DataFrame({
                        "test_feature": np.zeros(len(frame))
                    }),
                ),
                mock.patch.object(workflow, "_load_module", side_effect=fake_loader),
            ):
                report = workflow.run_full_corpus_test(
                    corpus,
                    model,
                    [training_csv],
                    ground_truth,
                    root / "Variants.csv",
                    root / "full-report.json",
                    batch_size=2,
                )
                no_truth = workflow.run_full_corpus_test(
                    corpus,
                    model,
                    [training_csv],
                    None,
                    root / "Variants-no-truth.csv",
                    root / "no-truth-report.json",
                    batch_size=2,
                )

            self.assertEqual(model.read_bytes(), b"frozen model bytes")
            self.assertEqual(report["total_corpus_tokens"], sum(frequencies.values()))
            self.assertEqual(report["total_unique_words"], len(frequencies))
            self.assertEqual(report["total_generated_candidate_pairs"], 4)
            self.assertEqual(report["total_classified_candidate_pairs"], 4)
            self.assertEqual(report["total_predicted_spelling_variants"], 2)
            self.assertEqual(report["total_predicted_non_variants"], 2)
            self.assertEqual(report["total_greedy_resolved_variants"], 2)
            self.assertEqual(report["training_pairs_excluded_from_independent_metrics"], 2)
            self.assertEqual(report["ground_truth_pairs_excluded_for_training_overlap"], 2)
            self.assertEqual(report["ground_truth_pairs_in_generated_candidates"], 2)
            self.assertEqual(report["ground_truth_coverage"], 2 / 3)
            self.assertEqual(report["confusion_matrix"], [[1, 0], [1, 1]])
            self.assertAlmostEqual(report["accuracy"], 2 / 3)
            self.assertEqual(report["status"], "FAIL")
            self.assertIsNone(no_truth["accuracy"])
            self.assertIsNone(no_truth["confusion_matrix"])
            self.assertEqual(no_truth["status"], "INCOMPLETE")
            exported = pd.read_csv(root / "Variants.csv")
            self.assertEqual(
                list(exported.columns),
                [
                    "word1", "word2", "word1_count", "word2_count",
                    "character_difference",
                ],
            )


if __name__ == "__main__":
    unittest.main()
