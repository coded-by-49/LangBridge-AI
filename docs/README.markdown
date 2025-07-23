## Progress Updates
- **Day 1 (July 19, 2025)**: Installed Python, PyTorch, CUDA, HuggingFace libraries. Created GitHub repo and initial `README.md`.
- **Day 2 (July 21, 2025)**: Downloaded JW300, Tatoeba, Flores200, Opus, and Hypa_Fleurs datasets. Organized in `data/`. Set up WSL environment.
- **Day 3 (July 23, 2025)**:
  - Converted Flores200 Parquet files (`hau_Latn.dev`, `eng_Latn.dev`, etc.) to CSVs (`data/flores200/ha_en_dev.csv`, etc.) using `pd.read_parquet`.
  - Attempted JW300 XML (`JW300_latest_xml_en-ig.xml`) conversion, but found no `<parallel>` tags; switched to pairing `jw300.ig.txt` and `jw300.en.txt` into `data/jw300/ig_en.csv`.
  - Started processing Opus CCMatrix files (`CCMatrix.en-ha.ha`, 957 MB; `CCMatrix.en-ha.en`, 357 MB) for Hausa-English pairs, but paused due to large file sizes. Will continue tomorrow.
  - Installed `scikit-learn` for train/validation splitting (pending completion).

## Research Notes
- **Datasets**:
  - **Flores200**: Parquet files converted to CSVs with `hausa`, `igbo`, `yoruba`, and `english` columns. Used `pyarrow` for efficient reading.
  - **JW300**: XML lacked `<parallel>` tags; used `<tuv>` or switched to `.txt` files for Igbo-English pairs. `.txt` files are line-aligned, simpler than XML’s nested structure.
  - **Opus**: CCMatrix files are large parallel text files (Hausa-English). Processing requires streaming to handle size (e.g., `zip_longest` for line pairing).
  - **Tatoeba/Hypa_Fleurs**: Pending conversion to CSVs.
- **Folder Structure**:
  ```
  langbridge-ai/
  ├── data/
  │   ├── flores200/
  │   │   ├── ha_en_dev.csv
  │   │   ├── ig_en_dev.csv
  │   │   ├── yo_en_dev.csv
  │   ├── jw300/
  │   │   ├── ig_en.csv
  │   ├── opus/
  │   │   ├── ha_en.csv (in progress)
  │   ├── tatoeba/
  │   ├── hypa_fleurs/
  ├── src/
  │   ├── convert_flores_parquet.py
  │   ├── convert_jw300_txt.py
  │   ├── convert_ccmatrix.py
  ├── docs/
  │   └── README.md
  ```

## Challenges and Questions
- **Challenges**:
  - JW300 XML was empty (`<parallel>` tags missing); switched to `.txt` files.
  - Large Opus CCMatrix files (957 MB, 357 MB) require memory-efficient processing.
  - `scikit-learn` installation delay slowed train/validation splitting.
- **Questions**:
  - How to efficiently process large CCMatrix files?
  - Best way to combine datasets for training?

## Next Steps
- Continue processing Opus CCMatrix files into `ha_en.csv`.
- Convert Tatoeba TSV and Hypa_Fleurs to CSVs.
- Preprocess all CSVs (remove duplicates, missing values), split into train/validation, and load into HuggingFace `datasets`.