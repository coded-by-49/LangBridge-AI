# Data

This page covers the data behind **LangBridge v1** (archived) and **LangBridge AI Igbo 1.0** (internal: v3): where it came from, how it was cleaned, how it was split, and its licenses.

The Igbo 1.0 data lives on Hugging Face: **[coded-by-49/langbridge_ai_v3_igbo_datasets](https://huggingface.co/datasets/coded-by-49/langbridge_ai_v3_igbo_datasets)**. This repository does not store the dataset. A local working copy (`langbridge_ai_v3_data/data/`) is git-ignored.

A **parallel corpus** is a set of sentence pairs where each sentence has a translation in another language. Every row in these datasets is one English sentence and its Igbo translation:

```json
{"id": 1, "translation": {"eng_Latn": "We are confusing regulation with policy.", "ibo_Latn": "Anyị na-ewehie ndozi dịka iwu"}}
```

(Row copied from `langbridge_ai_v3_data/data/igbo_v3/prod_test_json_files/jsonfile2.jsonl`.)

---

## 1. LangBridge v1 data (deleted)

The v1 dataset, "LangBridge Wazobia" (`coded-by-49/Langbridge_wazobia_data` on Hugging Face and `henryalohfabian/langbridge-wazobia-dataset` on Kaggle), **has been deleted and no longer exists**. Only its description and code remain.

**Languages:** Hausa, Igbo and Yoruba, each paired with English.

**Sources** (from the [old README](../archive/v1/README_v1.md) and [`archive/v1/data_pipeline/Data_cleaning.py`](../archive/v1/data_pipeline/Data_cleaning.py)):

- JW300 (religious texts)
- OPUS corpora, including CCMatrix, WikiTitles and XLEnt
- FLORES-200
- Tatoeba
- Hypa_Fleurs
- MAFAND-MT (a `masakane()` helper in the cleaning script)
- Custom web scraping

**Size:** The old README and project notes say "about 4 million" pairs. The v1 pipeline notebook ([`lang_bridge_data_pipeline.ipynb`](../archive/v1/data_pipeline/lang_bridge_data_pipeline.ipynb)) merged 15 tokenized shards into **3,000,000 training examples** (saved output: "Total Training Examples: 3000000"). The igbo validation file it tokenized had 93,319 rows.

**v1 cleaning steps** (from `Data_cleaning.py`):

1. Drop empty lines.
2. Drop lines where 10% or more of the characters fall outside the Latin Unicode blocks (Basic Latin, Latin-1, Latin Extended-A/B, Latin Extended Additional).
3. Drop lines where 5% or more of the characters are control characters or the Unicode replacement character (`�`).
4. Drop lines where a single character makes up 20% or more of the line.
5. Remove exact duplicate lines.
6. Convert everything to a JSONL format with NLLB language codes.

A git commit from 2025-07-28 reports "reduction of noisy data in ccmatrix file by over 95%". Possible issue **(guess)**: rule 4 also counts the space character. Spaces often make up 15–20% of a normal sentence, so the rule may have dropped some clean, short-word sentences.

**Direction:** As saved, the v1 tokenization code builds only *into-English* pairs (`ibo/hau/yor → eng_Latn`), with sequences truncated to 128 tokens. Whether other runs added English → X pairs is not recorded.

---

## 2. LangBridge AI Igbo 1.0 data

### Why new data

v1 mixed large, noisy, partly religious-domain corpora across six translation directions, and it scored below its base model everywhere (see [evaluation.md](evaluation.md)). For Igbo 1.0 the plan was the opposite: a small set of human-translated, research-grade sources, one language, one direction.

### Sources

From the [dataset card](https://huggingface.co/datasets/coded-by-49/langbridge_ai_v3_igbo_datasets) and [`archive/notes/v3_data/citations_for_langbridge_ai_v3.txt`](../archive/notes/v3_data/citations_for_langbridge_ai_v3.txt):

| Source | What it is | License of the original (from its Hugging Face card) |
|---|---|---|
| FLORES-101 / FLORES-200 (Goyal et al., 2022; NLLB Team, 2022) | Professional translations of English Wikipedia sentences, a standard evaluation benchmark | CC-BY-SA-4.0 (`facebook/flores`) |
| Igbo–English MT benchmark (Ezeani et al., 2020) | Human-translated news-domain sentences built for Igbo evaluation | Not listed in the citations file (TBD) |
| MAFAND-MT (Adelani et al., 2022) | African news translation corpus; includes English–Igbo | CC-BY-NC-4.0 (`masakhane/mafand`) |

The combined dataset is listed as **MIT** on Hugging Face. MAFAND-MT is non-commercial, so the MIT label does not cover every row. Check each source's terms before you reuse the data.

The MAFAND-MT loading script used is kept at [`scripts/data_v3/mafand.py`](../scripts/data_v3/mafand.py). It is the Hugging Face Datasets script for MAFAND-MT (Apache-2.0), with configs for `en-ibo`, `en-hau`, `en-yor` and others.

### Cleaning and building steps

From [`scripts/data_v3/text_dataset_processing.py`](../scripts/data_v3/text_dataset_processing.py) and [`scripts/data_v3/v3_igbo_datasets_under_256.py`](../scripts/data_v3/v3_igbo_datasets_under_256.py):

1. **Align plain-text files into pairs** (`create_parallel_corpus`): read an English file and an Igbo file line by line, strip whitespace, and write `{"id", "translation": {"eng_Latn", "ibo_Latn"}}` rows.
2. **De-duplicate across sources by the English side** (`remove_duplicate_jsonlines`): drop rows from file B whose English sentence already appears in file A. The last recorded use removed Ezeani validation rows that were already in MAFAND-MT dev.
3. **Concatenate files** (`extend_one_json_file_with_another`): append file A to file B and renumber the ids.
4. **Length filter for validation** (`v3_igbo_datasets_under_256.py`): drop pairs where either side is longer than 256 NLLB tokens (tokenizer of `facebook/nllb-200-distilled-1.3B`). Training separately truncates everything to 256 tokens.

The dataset card also says the data was "normalized into a unified CSV format". The published files are JSONL, not CSV.

### Counts before and after cleaning

| Step | Before | After | Source |
|---|---|---|---|
| Validation length filter (≤ 256 tokens) | 597 | 589 (8 removed) | line counts of the two validation files |
| Ezeani validation de-duplication against MAFAND dev | TBD | 193 | line count of `ezeani_val_igbo_eng.jsonl`; original count not recorded |
| Cross-source de-duplication of the training split | TBD | 12,009 | original count not recorded |

The training split still has **248 exact duplicate pairs** (12,009 rows, 11,761 unique pairs) and 501 repeated English sentences (11,508 unique). These numbers were computed during the repository overhaul.

### Splits

Sizes are line counts of the published files. They are the same in the local copy (the files are byte-identical; checked during the overhaul).

| Split | File on Hugging Face | Rows | Built from |
|---|---|---|---|
| Train | `igbo_eng_lv1_train.jsonl` | 12,009 | Includes all of FLORES dev (997) and devtest (1,012), MAFAND-MT test (1,500) and MAFAND-MT dev (1,500). The rest (~7,000 rows) is probably MAFAND-MT train **(guess; not recorded)** |
| Validation (full) | `igbo_eng_lv1_validation.jsonl` | 597 | Ezeani et al. test set (local note: original file name `ezeani_test_igbo_eng.jsonl`) |
| Validation (used in training) | `igbo_eng_lv1_validation_256_filtered.jsonl` | 589 | Validation after the 256-token filter |
| Test | `igbo_eng_lv1_test_.jsonl` | 1,693 | MAFAND-MT dev (1,500) + Ezeani et al. validation (193) (local note: "made up [of] ezeani val and mafand dev") |
| **Total** (train + full validation + test) | | **14,299** | |

The "includes all of" claims in the Train row come from exact pair matching against the local source files in `concatenated_data/`.

> **Overlap warning.** Because MAFAND-MT dev is in both train and test, **1,506 of the 1,693 test rows (88.95%) also appear in the training split.** Validation has no overlap with training. Reproduce this with [`scripts/check_split_overlap.py`](../scripts/check_split_overlap.py); results are in [`results/v3_split_overlap.csv`](../results/v3_split_overlap.csv). See [evaluation.md, Issue 4](evaluation.md#issue-4-test-split-overlaps-the-training-split).

> **Card mismatch.** The Hugging Face dataset card lists Validation = 1,693 and Test = 597. The files say the opposite. The two files were renamed on 2026-09-30 (dataset commit history).

### Domain and limits

All three sources are news or encyclopedic text. Informal, conversational and dialectal Igbo is barely covered. Diacritics and tone marks are used inconsistently across sources (dataset card).

---

## 3. Other files

- [`data/samples/`](../data/samples): two tiny JSONL files (14 rows) that look like test output of the de-duplication/concatenation functions. Note the repeated sentence with id 6997.
- `langbridge_ai_v3_data/data/igbo_v3/prod_test_json_files/` (local only): 13 short news sentences kept for manual production checks.

## 4. Citations

```bibtex
@article{nllb2022,
  author = {NLLB Team and Costa-jussà, Marta R. and Cross, James and others},
  title  = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year   = {2022},
  eprint = {2207.04672},
  archivePrefix = {arXiv}
}
@article{goyal2022flores,
  title   = {The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author  = {Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm{\'a}n, Francisco and Fan, Angela},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {10},
  pages   = {522--538},
  year    = {2022}
}
@article{guzman2019two,
  title   = {Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author  = {Guzm{\'a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal = {arXiv preprint arXiv:1902.01382},
  year    = {2019}
}
@misc{ezeani2020igboenglish,
  title  = {Igbo-English Machine Translation: An Evaluation Benchmark},
  author = {Ezeani, Ignatius and Rayson, Paul and Onyenwe, Ikechukwu and Uchechukwu, Chinedu and Hepple, Mark},
  year   = {2020},
  eprint = {2004.00648},
  archivePrefix = {arXiv}
}
@inproceedings{adelani-etal-2022-thousand,
  title     = {A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for {A}frican News Translation},
  author    = {Adelani, David and Alabi, Jesujoba and Fan, Angela and Kreutzer, Julia and others},
  booktitle = {Proceedings of NAACL-HLT 2022},
  pages     = {3053--3070},
  year      = {2022},
  url       = {https://aclanthology.org/2022.naacl-main.223}
}
```

The full author lists are in [`archive/notes/v3_data/citations_for_langbridge_ai_v3.txt`](../archive/notes/v3_data/citations_for_langbridge_ai_v3.txt).
