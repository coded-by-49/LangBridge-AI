# Data

The data is not stored in this repository.

- **LangBridge AI Igbo 1.0 data** lives on Hugging Face: [coded-by-49/langbridge_ai_v3_igbo_datasets](https://huggingface.co/datasets/coded-by-49/langbridge_ai_v3_igbo_datasets). It has 12,009 train, 597 validation (589 after the 256-token filter) and 1,693 test English–Igbo pairs. Load it with:

  ```python
  from datasets import load_dataset
  train = load_dataset("coded-by-49/langbridge_ai_v3_igbo_datasets", split="train")
  ```

  Note that `split="validation"` matches two files (the full and the filtered validation set). Load the filtered file explicitly, as `src/langbridge/train.py` does.

- **LangBridge v1 data** (the "LangBridge Wazobia" dataset, about 3–4 million Hausa, Igbo and Yoruba ↔ English pairs) **was deleted and no longer exists.**

`samples/` holds two tiny JSONL files (14 rows). They look like test output of the de-duplication and concatenation functions in `scripts/data_v3/text_dataset_processing.py`.

Sources, cleaning steps, split sizes, licenses, and the test/train overlap issue: [docs/data.md](../docs/data.md).
