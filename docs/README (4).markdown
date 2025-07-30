# LangBridge AI MVP

This repository contains the development environment and datasets for the LangBridge AI MVP, a project focused on building a multilingual model for Nigerian languages (Hausa, Igbo, Yoruba) paired with English. The project leverages various parallel corpora and tools to train and evaluate the model.

## Project Overview
- **Start Date**: July 19, 2025
- **Objective**: Process and preprocess datasets for training a language model supporting Hausa, Igbo, Yoruba, and English.
- **Tools**: Python, PyTorch, CUDA, HuggingFace, scikit-learn, pyarrow
- **Environment**: Windows Subsystem for Linux (WSL)

## Progress Updates
- **Day 1 (July 19, 2025)**: Installed Python, PyTorch, CUDA, and HuggingFace libraries. Created GitHub repository and initial `README.md`.
- **Day 2 (July 21, 2025)**: Downloaded JW300, Tatoeba, Flores200, Opus, and Hypa_Fleurs datasets. Organized in `data/`. Set up WSL environment.
- **Day 3 (July 23–28, 2025)**:
  - Converted Flores200 FLAC files (`hau_Latn.dev`, `eng_Latn.dev`, etc.) to CSVs (`data/Flores200/ha_en_dev.csv`, etc.) using `pd.read_csv()` or similar.
  - Processed JW300 XML (`JW300_latest_xml_en-ig.xml`) conversion; switched to pairing `jw300.ig.txt` and `jw300.en.txt` into `data/JW300/ig_en.csv` due to missing `<parallel>` tags.
  - Successfully processed large Opus CCMatrix files (`CCMatrix.en-ha.ha`, 957 MB; `CCMatrix.en-ha.en`, 357 MB) for Hausa-English pairs, creating `data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv`. Addressed 78 malformed lines, fixed header swap (`hausa`/`english` mislabeled), and mitigated non-printable characters (e.g., £, €) in Excel, likely due to UTF-8 BOM absence.
  - Diagnosed slow processing for smaller Wikititles.en-ha.ha dataset; investigating encoding issues with `wc -l`, `head`, and profiling.
  - Identified valid UTF-8 non-Latin characters (e.g., `ﾒｷﾒｷ`) in Igbo dataset (`XLEnt.en-ig.ig`). Proposed `unicodedata` solution to filter non-Latin scripts while preserving Nigerian Latin characters (e.g., `ẹ`, `ọ`).
  - Installed `scikit-learn` for train/validation splitting (pending completion).
- **Day 4 (July 30, 2025)**: Completed cleaning of all datasets (Flores200, JW300, Opus, Tatoeba, Hypa_Fleurs). Transitioning to preprocessing and training phases.

## Research Notes
- **Datasets**:
  - **Flores200**: Parquet files converted to CSVs with `hausa`, `igbo`, `yoruba`, and `english` columns. Processed using `pyarrow` for efficiency.
  - **JW300**: XML lacked `<parallel>` tags; used line-aligned `.txt` files (`jw300.ig.txt`, `jw300.en.txt`) for Igbo-English pairs, saved as `data/JW300/ig_en.csv`.
  - **Opus**:
    - **CCMatrix**: Large parallel Hausa-English text files (5,861,080 lines). Processed with chunked reading (`zip_longest`), binary mode (`"rb"`), and `errors="replace"` to handle non-UTF-8 bytes. Output: `data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv`.
    - **Wikititles**: Smaller dataset; processing bottlenecks under investigation due to encoding or filtering issues.
    - **XLEnt (Igbo)**: Contains valid UTF-8 non-Latin characters (e.g., Katakana `ﾒｷ`), requiring `unicodedata.block()` to filter while preserving Latin Extended ranges.
  - **Tatoeba**: Processed `hau_sentences.tsv`, `ibo_sentences.tsv`, `yor_sentences.tsv`, and `links/` into CSVs (e.g., `data/Tatoeba/ha_en.csv`).
  - **Hypa_Fleurs**: Imported and cleaned; output pending preprocessing (e.g., `data/Hypa_fleurs/ha_en.csv`).
- **Folder Structure**:
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
│   │   │   └── unclean_files/
│   │   ├── Yoruba/
│   │   │   ├── en-yo.txt(1) to en-yo.txt(14)
│   │   └── Preprocessed_csv/
│   │       ├── Hausa/
│   │       │   ├── CCMatrix_ha_en.csv
│   │       ├── Igbo/
│   │       └── Yoruba/
├── Tatoeba/
│   ├── hau_sentences.tsv
│   ├── ibo_sentences.tsv
│   ├── yor_sentences.tsv
│   ├── links/
│   └── ha_en.csv
  ```

## Challenges and Solutions
- **Challenges**:
- **Hypa_Fleurs Access**: Initial difficulties accessing the file were resolved through iterative troubleshooting.
- **Anomaly Characters**: Encountered non-printable characters (e.g., £, €) and valid UTF-8 non-Latin characters (e.g., `ﾒｷ`) in Opus and Igbo datasets. Mitigated with UTF-8 BOM addition and `unicodedata` filtering, preserving Nigerian Latin characters (e.g., `ẹ`, `ọ`).
- **Data Retention vs. Cleanliness (Yoruba File)**: Balanced retention of useful data with removal of noise, ensuring minimal loss of linguistic content.
- **Large File Processing**: Slow processing of Opus CCMatrix (957 MB) and Wikititles due to encoding or filtering; addressed with chunked reading and ongoing diagnostics.
- **Solutions**:
- Used `zip_longest` and binary mode (`"rb"`) for large files.
- Implemented `unicodedata.block()` to filter non-Latin scripts while retaining Latin Extended ranges.
- Added UTF-8 BOM and cleaned non-printable characters in Excel for better rendering.

## Next Steps
- Preprocess all cleaned datasets (Flores200, JW300, Opus, Tatoeba, Hypa_Fleurs) for model training.
- Perform train/validation splitting using `scikit-learn`.
- Initiate model training with PyTorch and HuggingFace libraries.
- Optimize processing for large datasets (e.g., Wikititles) based on diagnostic results.

## Contact
For questions or contributions, please open an issue or contact the project maintainer.

---