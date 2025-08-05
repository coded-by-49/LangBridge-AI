# LangBridge AI MVP: Daily Update (August 5, 2025)

## Summary
Today, significant progress was made in the preprocessing phase of the LangBridge AI MVP project, focusing on tokenizing the combined Igbo-English dataset and evaluating tokenizer quality. A critical issue was identified with the tokenizer’s poor translation performance, which will be addressed tomorrow.

## Progress
- **Dataset Preprocessing**:
  - Loaded merged Igbo-English dataset from `data/Merged_data/Igbo/merged_igbo_eng.pkl`, combining texts from multiple sources (e.g., `CCAligned_ig_en.csv`, `JW300/ig_en.csv`, `Flores200/ig_en_dev.csv`).
  - Trained a BPE tokenizer (`Tokenizers/tokenizer_igbo_en.json`) with a vocabulary size of 60,000 and special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`).
  - Tokenized the merged dataset into PyTorch tensors, saved as `data/Tokenized_data/Igbo/tokenized_ibo_eng.pt`.
  - Fixed an issue with `<unk>` token being `null` by explicitly setting `tokenizer.unk_token = "<unk>"`.
- **Tokenizer Evaluation**:
  - Implemented and ran `round_trip_check` on 10,000 sampled sentences from `clean_merged_igbo_eng`.
  - Results: 5,280 disconcordant sentences (47.20% concordance rate, 0 failed encodings), indicating poor tokenizer performance.
- **Environment**:
  - Confirmed CUDA usage on RTX 3060 Laptop GPU with `torch.set_default_device('cuda')` for tensor operations.
  - Used `random.sample` for efficient round-trip testing, reducing runtime to ~1–2 minutes.

## Challenges
- **Poor Tokenizer Performance**: The round-trip check showed a low concordance rate (47.20%), suggesting the tokenizer fails to preserve ~52.8% of sentences. Possible causes include insufficient vocabulary coverage, data cleaning issues, or BPE merge rule inefficiencies.
- **Time Constraint**: With the deadline at midnight WAT, there’s limited time to retrain the tokenizer today.
- **Learning Curve**: Balancing debugging with deepening NLP knowledge (e.g., BPE mechanics, tokenizer optimization).

## Solutions
- Identified and fixed `<unk>` token issue by setting `unk_token` explicitly after loading the tokenizer.
- Used sampling (10,000 sentences) for `round_trip_check` to quickly diagnose tokenizer issues without processing the entire dataset.
- Plan to study a textbook (e.g., on NLP or tokenization) tomorrow to better understand BPE and tokenizer optimization.
- Scheduled tokenizer retraining for tomorrow, potentially increasing `vocab_size` or improving data cleaning.

## Next Steps
- **Tomorrow (August 6, 2025)**:
  - Investigate low concordance rate by analyzing disconcordant sentences and checking `<unk>` frequency with `frequency_unknown_ids`.
  - Retrain tokenizer with adjusted parameters (e.g., larger `vocab_size`, better pre-tokenization).
  - Study a recommended NLP textbook (e.g., *Speech and Language Processing* by Jurafsky & Martin) for deeper BPE understanding.
- **Today (Remaining)**:
  - Split tokenized data into train/validation sets using `split_datasets.py`.
  - Begin training the MarianMT model with `train_model.py`.
  - Evaluate model with `evaluate_model.py` and test translations with `translate.py`.

## Notes
- The low concordance rate (47.20%) is a critical issue that must be resolved to ensure robust NMT performance. Sampling 10,000 sentences was effective for quick diagnosis.
- Studying a textbook will enhance understanding of tokenizer design, crucial for low-resource languages like Igbo.
- GPU setup remains stable, supporting efficient tensor operations for training.