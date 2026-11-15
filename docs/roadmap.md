# Roadmap

Where **LangBridge AI** goes after **Igbo 1.0** (internal: v3). The order follows one rule: **get one direction right, prove it on clean data, then scale.**

## Step 0: fix the evaluation (next)

Before anything else, make the Igbo 1.0 result trustworthy:

1. **Clean test set.** Rebuild the splits so no test pair appears in training. Today 1,506 of 1,693 test rows (88.95%) do ([evaluation.md](evaluation.md#issue-4-test-split-overlaps-the-training-split)). Make [`scripts/check_split_overlap.py`](../scripts/check_split_overlap.py) part of the dataset build.
2. **Re-score now** on what is already clean: the 187 test rows that are not in train, and the 597-row validation split. Both models, both directions.
3. **Significance test.** Use paired bootstrap resampling (Koehn, 2004): resample the test sentences many times and count how often the fine-tuned model beats the baseline. Report a confidence interval for the AfriCOMET difference.
4. **Second metric.** Add chrF++ and BLEU next to AfriCOMET, so the result does not depend on one learned metric.
5. **Human check.** Ask native Igbo speakers to rate a blind sample of both models' outputs.
6. **Save the evidence.** Keep `trainer_state.json`, generated outputs and per-sentence scores for every run.

## Step 1: English → Igbo 1.1

- Retrain on the rebuilt splits, if Step 0 shows the gain does not hold.
- Remove the 248 duplicate pairs in the training split.
- Add more data from the candidate list below.

## Step 2: from one direction to six

Same recipe, one direction at a time, each with its own clean test set:

| Order | Direction | Notes |
|---|---|---|
| 1 | English → Igbo | Igbo 1.0 (current) |
| 2 | Igbo → English | Separate adapter. Igbo 1.0 is worse than the base model here (−10.86%) |
| 3 | English → Hausa | MAFAND-MT has `en-hau`; FLORES-200 has `hau_Latn` |
| 4 | Hausa → English | v1 lost the most here (−46.13%). Check the `src_lang` setup first |
| 5 | English → Yoruba | MAFAND-MT has `en-yor`; FLORES-200 has `yor_Latn` |
| 6 | Yoruba → English | Compute the missing v1 baseline first |

## Per-direction model routing

Serve each direction with whichever model is best for it: a LangBridge adapter where it beats the base model, otherwise the base NLLB model. The current table is in [deployment.md](deployment.md#routing-plan). As new adapters are added, each one replaces the base model only for its own direction.

## Evaluation-gated promotion

A new model goes live only if it beats the current model on the same clean test set. The rule:

1. Same test set, same generation settings, same metric version.
2. Better AfriCOMET score, with a paired bootstrap confidence interval that excludes zero.
3. No large drop on a second metric (chrF++).
4. A human spot check of 50 sentences shows no new failure modes.

If any step fails, the current model stays.

## Datasets to consider

From [`archive/notes/v3_data/datasets_to_keep_in_mind.txt`](../archive/notes/v3_data/datasets_to_keep_in_mind.txt):

- [`ccibeekeoc42/english_to_igbo`](https://huggingface.co/datasets/ccibeekeoc42/english_to_igbo) on Hugging Face. Check its license and quality before use.

Already used, and available for the other languages:

- MAFAND-MT: `en-hau`, `en-yor` configs in [`scripts/data_v3/mafand.py`](../scripts/data_v3/mafand.py).
- FLORES-200: Hausa and Yoruba dev/devtest. Keep at least one part for testing only.

## Deployment

See the checklist in [deployment.md](deployment.md#checklist).
