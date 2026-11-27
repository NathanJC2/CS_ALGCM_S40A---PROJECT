# CSALGCM Spelling Variant Detection

This project implements a hybrid Filipino spelling-variant pipeline using
PinoyBot language identification, divide-and-conquer candidate blocking,
dynamic-programming edit distance, contextual features, supervised
classification, greedy pair resolution, and deterministic CSV/XLSX export.

## Installation

```powershell
py -m pip install -r requirements.txt
py train_model.py 67.csv --output-dir .
```

## Train and run

Verified word-pair labels are required for supervised training. Use a CSV:

```csv
word1,word2,is_variant
politiko,pulitiko,1
opo,upo,0
polittico,pulitiko,0
```

```powershell
py 10_pipeline_runner.py --corpus Raw.txt --labels-csv labels.csv --output-csv Variants.csv --evaluation-xlsx Evaluation.xlsx --embedding-backend sklearn
```

A small test run is available with a synthetic corpus:

```powershell
py 10_pipeline_runner.py --demo
```

After training, inference can reuse `variant_classifier.pkl`:

```powershell
py 10_pipeline_runner.py --corpus Raw.txt --variant-model variant_classifier.pkl --output-csv Variants.csv
```

PinoyBot language artifacts are `classifier.pkl` and `vectorizer.pkl`. They are
trained separately from the spelling-variant classifier with `train_model.py`.

## Human annotation and active learning

Start with a diverse, reproducible, blank-label sample generated from the
actual corpus. The candidate generator uses canonical Filipino spelling
transforms, deletion signatures, character n-grams, edit distance, corpus
frequency, PinoyBot predictions, and contexts observed in `Raw.txt`. It samples
bounded posting-list neighbors instead of enumerating all vocabulary pairs.
Context examples are copied from the corpus. None of the candidate-generation
signals is treated as a verified label.

```powershell
py annotation_workflow.py generate --corpus Raw.txt --output annotation_candidates.csv --sample-size 1000 --random-state 42
```

In `annotation_candidates.csv`, enter `1` for the same lexical item, `0` for
different lexical items, or leave `is_variant` blank when uncertain/unreviewed.
Keep the other columns unchanged. Validate and canonicalize labels, optionally
combining multiple annotator files, then train:

```powershell
py annotation_workflow.py validate --annotations annotation_candidates.csv annotator2.csv --output annotations_merged.csv
py annotation_workflow.py train --annotations annotations_merged.csv --model variant_classifier.pkl
```

Only verified `0`/`1` rows are used for training. Conflicting duplicate or
reversed-pair labels stop with an error. Training reports out-of-fold precision,
recall, F1, confusion matrix, and chooses a threshold from out-of-fold
probabilities. The folds group pairs by their canonical spelling keys to reduce
lexical-group leakage. These metrics describe the available annotation sample,
not the final corpus output.

Generate another human review batch near the selected threshold, while avoiding
repeated canonical lexical forms:

```powershell
py annotation_workflow.py active --corpus Raw.txt --model variant_classifier.pkl --output active_learning_candidates.csv --sample-size 1000
py annotation_workflow.py validate --annotations annotations_merged.csv active_learning_candidates.csv --output annotations_merged.csv
py annotation_workflow.py train --annotations annotations_merged.csv --model variant_classifier.pkl
```

The `active` workflow leaves all labels blank. Add only human-verified values,
then merge and retrain. It does not create pseudo-labels.

## Final-output and independent testing

The final-output precision evaluation reads the existing 100-row `Evaluation.xlsx`.
Manually enter `TRUE`/`FALSE` or `1`/`0` in `is_correct` for each pair, then run:

```powershell
py annotation_workflow.py evaluate --evaluation-xlsx Evaluation.xlsx --corpus Raw.txt --model variant_classifier.pkl
```

Blank labels are unreviewed, not incorrect. A final PASS requires all 100 rows
to be reviewed and precision strictly greater than 70%.

Create a separate independent candidate test set after freezing the trained
model and threshold:

```powershell
py annotation_workflow.py test-sample --corpus Raw.txt --model variant_classifier.pkl --annotations annotations_final.csv --output Test_Dataset.csv --sample-size 1000
```

The generator uses the same blocking and feature pipeline as prediction,
excludes annotated word pairs and lexical groups, draws a reproducible sample
proportional to the model's predicted positive/negative populations, and does
not include model predictions in the CSV. Manually enter `1`, `0`, or blank in
`is_variant`. Keep the generated `Test_Dataset.csv.metadata.json` beside the
CSV; it records the model, threshold, corpus, and pair identities needed to
check independence. Do not copy test labels into any training or active-learning
annotation file.

Evaluate the labeled sample without using its labels for training or threshold
tuning:

```powershell
py annotation_workflow.py test --corpus Raw.txt --model variant_classifier.pkl --test-data Test_Dataset.csv --annotations annotations_final.csv
```

The command reports standard classification metrics, a confusion matrix,
majority-class baseline, and unreviewed count. Accuracy exactly 70% is FAIL;
only accuracy above 70% passes, and any unreviewed test rows produce
INCOMPLETE. The independent result measures the blocked candidate pairs sampled
from `Raw.txt`, not every possible word pair in its vocabulary. If the model,
threshold, corpus, or feature pipeline changes after sample generation, create
a fresh independent dataset before making a final assessment. `Final_Evaluation.json`
and `Final_Evaluation.txt` keep training cross-validation, final-output
precision, and independent-test performance in separate sections.

## Full-corpus test (no sample)

Run the `test-full` command for full-corpus inference. It reads all of `Raw.txt`,
uses the Divide-and-Conquer blocker without a global candidate-pair limit, and
spools/deduplicates candidate pairs to temporary SQLite storage before scoring
them in feature batches. It does not randomly sample candidates and does not
modify the trained model, `Evaluation.xlsx`, or training annotation files.

```powershell
py annotation_workflow.py test-full --corpus Raw.txt --model variant_classifier.pkl --annotations annotations_final.csv
```

By default this writes the normal five-column `Variants.csv` plus
`Full_Corpus_Test.json`. The classifier's per-signature blocking safeguard
defaults to 5,000 pairs; adjust it with `--max-pairs-per-signature` if needed.
`--batch-size` controls the pair batch size (default 2,048).

`--annotations` identifies verified training pairs and lexical groups; those
are still included in full-corpus inference/output, but excluded from
independent ground-truth metrics. To calculate independent metrics, provide a
separate CSV with `word1,word2,is_variant` and verified `0`/`1` labels:

```powershell
py annotation_workflow.py test-full --corpus Raw.txt --model variant_classifier.pkl --annotations annotations_final.csv --ground-truth independent_verified.csv
```

Ground-truth pairs that overlap training pairs or training lexical groups are
excluded. Metrics are computed against the final Greedy-resolved pair set;
verified reference variants missed by blocking or edit-distance filtering
count as false negatives. Ground-truth coverage reports what fraction of the
eligible verified reference pairs were generated by candidate blocking. Without
a ground-truth file, the command still reports full-corpus counts and
predictions, but accuracy, precision, recall, F1, balanced accuracy, and the
confusion matrix are `N/A`; the result is `INCOMPLETE`, never an invented score.
`test-full` does not use or fall back to `Test_Dataset.csv`.

## Full-corpus prediction

After you have enough reviewed positive and negative examples:

```powershell
py annotation_workflow.py run --corpus Raw.txt --model variant_classifier.pkl --output-csv Variants.csv --evaluation-xlsx Evaluation.xlsx --candidate-limit 100000 --random-state 42
```

The candidate limit bounds the number of indexed candidate pairs scored; it is
not the vocabulary size. Increase it to trade runtime and memory for candidate
recall. The generator does not compare every pair. The runner resolves
high-confidence pairs and writes the exact five required CSV columns. The
evaluation workbook independently samples up to 100 output pairs, with blank
human assessment cells; precision remains unavailable until those cells are
manually completed.

Counts, bounded sparse word-context profiles, and unique-word PinoyBot
predictions are cached against corpus metadata (and, for language labels, model
metadata) in `.annotation_cache.pkl`, `.context_profiles.pkl`, and
`.language_predictions.pkl`. Remove the relevant cache explicitly to force a
rebuild. Context profiles are capped at 64 context types per word. Their cosine
is used as a sparse co-occurrence embedding feature; this annotation workflow
does not claim to train a separate dense Word2Vec model.
There is no guarantee that blocking retrieves every true variant, so inspect
recall against manually annotated examples before describing output coverage.

## Output

`Variants.csv` contains exactly:

```text
word1,word2,word1_count,word2_count,character_difference
```

`Evaluation.xlsx` samples up to 100 predicted pairs reproducibly. The
`is_correct` column is blank until a human manually verifies each pair.
No automatic precision is claimed without human labels.

The installed scikit-learn version must match the pinned version in
`requirements.txt`. Existing `variant_classifier.pkl` artifacts created with a
different version should be regenerated after installing dependencies:

```powershell
py -m pip install -r requirements.txt
py annotation_workflow.py train --annotations annotations_final.csv --model variant_classifier.pkl
```

The spelling-variant classifier stores scikit-learn decision-tree estimators;
retraining serializes them with the pinned version. PinoyBot's separate
`classifier.pkl` and `vectorizer.pkl` are not spelling-variant model artifacts.
The checked-in PinoyBot `classifier.pkl` and `vectorizer.pkl` were serialized
with scikit-learn 1.9.1; the current dependency pin matches them. The existing
spelling model should be retrained under that same environment before use.

## Algorithmic complexity

- Preprocessing: O(T) time and O(V + S) state for T tokens, V vocabulary, and S
  retained sentence tokens.
- Candidate blocking: average O(B + postings); bounded per signature. The
  implementation is a heuristic and may miss a pair; downstream stages do not
  perform exhaustive comparisons.
- Edit distance: O(|a||b|) time and space in the current matrix implementation.
- Context embeddings: sparse co-occurrence plus TruncatedSVD; cost depends on
  matrix density and requested dimensions.
- Features: O(P*(|w| + context lookups)) for P candidates.
- Random forest: training and inference scale with estimators and features.
- Greedy resolver: O(E log E + E*C) in its complete-linkage checks, with
  O(V + E) state.

## Testing

```powershell
py -m unittest discover -s tests -v
```