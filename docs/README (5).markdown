# LangBridge AI MVP

This repository contains the development environment, datasets, and code for the LangBridge AI MVP, a project aimed at building a multilingual neural machine translation (NMT) model for Nigerian languages (Hausa, Igbo, Yoruba) paired with English. The project leverages parallel corpora and modern NLP tools to preprocess, train, and evaluate the model.

## Project Overview
- **Start Date**: July 19, 2025
- **Objective**: Develop a multilingual NMT model to translate between Hausa, Igbo, Yoruba, and English, focusing on low-resource language challenges.
- **Tools**: Python, PyTorch, CUDA, HuggingFace (transformers, tokenizers), scikit-learn, pyarrow
- **Environment**: Windows Subsystem for Linux (WSL) with NVIDIA GPU (RTX 3060 Laptop GPU)
- **Datasets**: Flores200, JW300, Opus (CCMatrix, Wikititles, XLEnt), Tatoeba, Hypa_Fleurs

## Progress Updates
- **Week 1 (July 19–21, 2025)**:
  - Set up development environment: Installed Python, PyTorch, CUDA, HuggingFace libraries, and scikit-learn.
  - Created GitHub repository and initial `README.md`.
  - Downloaded datasets (JW300, Tatoeba, Flores200, Opus, Hypa_Fleurs) and organized in `data/` directory.
  - Configured WSL environment for GPU support.
- **Week 2 (July 23–28, 2025)**:
  - Converted Flores200 Parquet/FLAC files to CSVs (e.g., `data/Flores200/ha_en_dev.csv`) using `pyarrow` and `pandas`.
  - Processed JW300 XML files, switching to line-aligned `.txt` files (e.g., `jw300.ig.txt`, `jw300.en.txt`) for Igbo-English pairs, saved as `data/JW300/ig_en.csv`.
  - Cleaned large Opus CCMatrix Hausa-English files (5.8M lines) into `data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv`, addressing 78 malformed lines, header swaps, and non-printable characters (e.g., £, €).
  - Investigated slow processing of Wikititles dataset due to encoding issues; diagnostics ongoing.
  - Filtered non-Latin characters (e.g., `ﾒｷ`) in Igbo XLEnt dataset using `unicodedata`, preserving Nigerian Latin characters (e.g., `ẹ`, `ọ`).
- **Week 3 (July 30–August 1, 2025)**:
  - Completed cleaning of all datasets (Flores200, JW300, Opus, Tatoeba, Hypa_Fleurs).
  - Began preprocessing phase: Tokenized QED Igbo-English dataset (`data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv`) using HuggingFace `tokenizers` with BPE, producing `source_tokens_s1.pt` and `target_tokens_s1.pt`.
  - Configured PyTorch to default to CUDA on NVIDIA RTX 3060 Laptop GPU, optimizing tensor operations.
  - Identified small vocabulary size (621 tokens) for QED dataset, planning to combine with larger datasets (e.g., JW300) to improve vocabulary coverage.

## Folder Structure
```
LANGBRIDG_AI/
├── .vscode/
├── data/
│   ├── Flores200/
│   │   ├── ha_en_dev.csv
│   │   ├── ig_en_dev.csv
│   │   └── yo_en_dev.csv
│   ├── Hypa_fleurs/
│   │   ├── ha_en.csv (pending)
│   │   ├── ig_en.csv (pending)
│   │   └── yo_en.csv (pending)
│   ├── JW300/
│   │   ├── ig_en.csv
│   │   ├── jw300.ig.txt
│   │   └── jw300.en.txt
│   ├── Opus/
│   │   ├── Hausa/
│   │   │   ├── clean_files/
│   │   │   │   ├── en-ha.txt(1)
│   │   │   │   ├── en-ha.txt(2)
│   │   │   │   └── en-hau.txt(3)
│   │   │   └── unclean_files/
│   │   ├── Igbo/
│   │   │   ├── clean_files/
│   │   │   │   ├── en-ig.txt(4) to en-ig.txt(15)
│   │   │   │   └── QED_ig_en.csv
│   │   │   └── unclean_files/
│   │   ├── Yoruba/
│   │   │   ├── en-yo.txt(1) to en-yo.txt(14)
│   │   └── Preprocessed_csv/
│   │       ├── Hausa/
│   │       │   ├── CCMatrix_ha_en.csv
│   │       ├── Igbo/
│   │       │   ├── QED_ig_en.csv
│   │       └── Yoruba/
│   ├── Tatoeba/
│   │   ├── hau_sentences.tsv
│   │   ├── ibo_sentences.tsv
│   │   ├── yor_sentences.tsv
│   │   ├── links/
│   │   └── ha_en.csv
│   ├── Tokenised_Data/
│   │   ├── Igbo/
│   │       ├── source_tokens_s1.pt
│   │       ├── target_tokens_s1.pt
├── tokenizers/
│   ├── tokenizer_igbo_en.json
```

## Challenges and Solutions
- **Data Cleaning**:
  - **Challenge**: Non-printable characters (e.g., £, €) and non-Latin scripts (e.g., `ﾒｷ`) in Opus and Igbo datasets.
  - **Solution**: Added UTF-8 BOM, used `unicodedata.block()` to filter non-Latin scripts while preserving Nigerian Latin characters (e.g., `ẹ`, `ọ`).
- **Large File Processing**:
  - **Challenge**: Slow processing of Opus CCMatrix (957 MB) and Wikititles due to encoding issues.
  - **Solution**: Implemented chunked reading with `zip_longest`, binary mode (`"rb"`), and `errors="replace"`.
- **Small Vocabulary Size**:
  - **Challenge**: QED Igbo-English dataset produced a 621-token vocabulary, potentially limiting model performance.
  - **Solution**: Plan to combine with larger datasets (e.g., JW300) and adjust `vocab_size` if needed.

## Next Steps
- Preprocess remaining datasets (Flores200, JW300, Opus, Tatoeba, Hypa_Fleurs) for tokenization.
- Split tokenized data into train/validation sets using `scikit-learn`.
- Train NMT model using PyTorch and HuggingFace transformers.
- Optimize processing for Wikititles dataset based on encoding diagnostics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/LangBridge_AI.git
   ```
2. Install dependencies in a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch tokenizers pandas scikit-learn pyarrow
   ```
3. Set up WSL with CUDA for GPU support (see NVIDIA documentation).

## Contact
For questions or contributions, open an issue or contact the project maintainer.