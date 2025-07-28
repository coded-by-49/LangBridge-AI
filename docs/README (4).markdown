## Progress Updates
- **Day 1 (July 19, 2025)**: Installed Python, PyTorch`, CUDA, HuggingFace` libraries. Created GitHub repo and initial `README.md`.
- **Day 2 (July 21, 2025)**: Downloaded JW300, Tatoeba, Flores200, Opus, and Hypa_Fleurs datasets. Organized in `data/`. Set up WSL environment.
- **Day 3 (July 23–28, 2025)**:
  - Converted Flores200 FLAC files (`hau_Latn.dev`, `eng_Latn.dev`, etc.) to CSVs (`data/flores200/ha_en_dev.csv`, etc.) using `pd.read_csv()` or similar.
  - Attempted JW300 XML (`JW300_latest_xml_en-ig.xml`) conversion, but found no `<parallel>` tags; switched to pairing `jw300.ig.txt` and `jw300.en.txt` into `data/jw300/ig_en.csv`.
  - Successfully processed large Opus CCMatrix files (`CCMatrix.en-ha.ha`, 957 MB; `CCMatrix.en-ha.en`, 357 MB) for Hausa-English pairs, creating `data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv`. Addressed 78 malformed lines in `.en` (initially detected by `grep -c "^$"`, later refined to ~9 via `is_valid_line`), fixed header swap (`hausa`/`english` mislabeled), and mitigated funny characters (e.g., £, €) in Excel, likely due to UTF-8 BOM absence or residual non-printable characters.
  - Encountered slow processing for smaller Wikititles.en-ha.ha dataset, potentially due to encoding issues or overly strict filtering; currently diagnosing with `wc -l`, `head`, and profiling.
  - Identified non-Latin UTF-8 characters (e.g., `ﾒｷﾒｷ...`) in Igbo dataset (`XLEnt.en-ig.ig`), undetected by `grep -aP '\xEF\xBF\xBD'` due to valid UTF-8 encoding. Proposed `unicodedata` solution to filter non-Latin scripts while preserving Nigerian Latin characters (e.g., `ẹ`, `ọ`).
  - Installed `scikit-learn` for train/validation splitting (pending completion).

## Research Notes
- **Datasets**:
  - **Flores200**: Parquet files converted to CSVs with `hausa`, `igbo`, `yoruba`, and `english` columns. Used `pyarrow` for efficient reading.
  - **JW300**: XML lacked `<parallel>` tags; used `<tuv>` or switched to `.txt` files for Igbo-English pairs. `.txt` files are line-aligned, simpler than XML’s nested structure.
  - **Opus**: 
    - CCMatrix: Large parallel Hausa-English text files (5,861,080 lines). Processed with chunked reading (`zip_longest`), binary mode (`"rb"`), and `errors="replace"` to handle non-UTF-8 bytes. Encountered malformed lines, header swaps, and Excel rendering issues (fixed via UTF-8 BOM and cleaning).
    - Wikititles: Smaller dataset, but processing is slow; investigating encoding or filtering bottlenecks.
    - XLEnt (Igbo): Contains valid UTF-8 non-Latin characters (e.g., Katakana `ﾒｷ`), requiring `unicodedata.block()` to filter while preserving Latin Extended ranges for Nigerian languages.
  - **Tatoeba/Hypa_Fleurs**: Pending conversion to CSVs.
- **Folder Structure**:
  ```
  LANGBRIDG_AI/
├── .vscode/
├── data/
│   ├── Flores200/
│   ├── Hypa_fleurs/
│   ├── JW300/
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
│   │       ├── Igbo/
│   │       └── Yoruba/
├── Tatoeba/
│   ├── hau_sentences.tsv
│   ├── ibo_sentences.tsv
│   ├── yor_sentences.tsv
│   └── links/
  ```

## Challenges and Questions
- **Challenges**:
  - Large Opus CCMatrix files (957 MB, 357 MB) processed last week contain various UTF-8 non-Latin Nigerian characters, undetectable by `grep -c` and initial non-lingual detection functions in code.
  - Similar issues with smaller Opus datasets (e.g., Wikititles, XLEnt), including slow processing and valid UTF-8 non-Latin characters (e.g., Katakana in Igbo dataset).
  - Excel rendering issues with funny characters (e.g., £, €) in CCMatrix CSV, likely due to missing UTF-8 BOM or unfiltered non-printable characters.
- **Questions**:
  - How to efficiently process large CCMatrix files and smaller datasets like Wikititles?
  - How to detect UTF-8 non-Latin Nigerian languages via bash or Python while preserving Latin Extended characters?

## Next Steps
- Preprocessing of all lingual datasets from Opus (e.g., Wikititles, XLEnt).
- Cleaning and reprocessing of all cleaned datasets for further detection of non-Latin characters.
- Importation and preprocessing of Hypa_Fleurs dataset.