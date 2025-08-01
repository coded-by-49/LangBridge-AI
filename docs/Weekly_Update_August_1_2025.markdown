# LangBridge AI MVP: Weekly Update (Week Ending August 1, 2025)

## Summary
This week, the LangBridge AI MVP project transitioned into the preprocessing phase, focusing on tokenizing the QED Igbo-English dataset to prepare for neural machine translation (NMT) model training. Key achievements include setting up GPU-accelerated preprocessing and identifying challenges with vocabulary size.

## Progress
- **Dataset Preprocessing**:
  - Tokenized the QED Igbo-English dataset (`data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv`) using HuggingFace `tokenizers` with Byte-Pair Encoding (BPE).
  - Generated tokenized tensors (`source_tokens_s1.pt`, `target_tokens_s1.pt`) saved in `data/Tokenised_Data/Igbo/`.
  - Configured PyTorch to default to CUDA on NVIDIA RTX 3060 Laptop GPU, optimizing tensor operations.
- **Environment Optimization**:
  - Resolved VSCode import issues by selecting the correct WSL Python interpreter.
  - Confirmed GPU usage with `nvidia-smi` and `torch.set_default_device('cuda')`.
- **Analysis**:
  - Identified a small vocabulary size (621 tokens) for the QED dataset, likely due to its limited size (~325 sentences).
  - Evaluated potential impact on model performance (e.g., frequent `<unk>` tokens, long sequences).

## Challenges
- **Small Vocabulary Size**: The QED dataset’s 621-token vocabulary may limit translation quality. Plan to combine with larger datasets (e.g., JW300) to increase vocabulary diversity.
- **Runtime**: Tokenization took 2–3 minutes, likely due to CPU-bound `tokenizers` operations or WSL I/O. Investigating optimization strategies.
- **Learning Curve**: Balancing hands-on coding with NLP concept learning (e.g., BPE tokenization) under a tight deadline.

## Solutions
- Used `logging.getLogger("tokenizers").setLevel(logging.ERROR)` to suppress verbose tokenizer output if needed.
- Proposed combining QED with JW300 dataset to improve vocabulary size, with checks for `<unk>` frequency and sequence lengths.
- Profiled tokenization to identify bottlenecks, planning to use SSD for faster I/O in WSL.

## Next Steps
- Tokenize remaining datasets (Flores200, JW300, Opus, Tatoeba, Hypa_Fleurs).
- Combine QED with larger datasets to address vocabulary size issue.
- Split tokenized data into train/validation sets using `scikit-learn`.
- Begin NMT model training with PyTorch and HuggingFace transformers.
- Continue diagnostics for Wikititles dataset processing.

## Notes
- The small QED dataset highlights the importance of dataset size in NMT. Combining datasets will be critical for robust model performance.
- GPU setup is successful, paving the way for efficient training in the next phase.