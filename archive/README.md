# Archive

Older work, kept for history. Nothing here is needed to use LangBridge AI Igbo 1.0.

**`v1/`**: LangBridge v1, the multilingual NLLB-200-distilled-600M + LoRA model for Hausa, Igbo and Yoruba ↔ English. It was superseded by LangBridge AI Igbo 1.0 after it scored below its base model in every direction. It contains:

- the old README (`README_v1.md`; its claims are outdated);
- the v1 results sheet (`langbridge_ai_inference_sheet.md`);
- the training notebook (`training/`);
- the data cleaning script and data pipeline notebook (`data_pipeline/`);
- the inference and AfriCOMET evaluation notebooks (`evaluation/`).

These files were moved with `git mv`, so their history is kept. The dataset they used has been deleted.

**`v3_kaggle_export/lb_train_script_main.ipynb`**: the raw Kaggle export of the Igbo 1.0 training script. It is a notebook JSON that was saved with a `.py` extension. Its code cell is now [`src/langbridge/train.py`](../src/langbridge/train.py).

**`notes/`**: working notes. The problem/solution notes from training and evaluation are rewritten as entries in the [Engineering log](../docs/engineering_log.md):

- `oom_error_when_loading_nllb.txt`
- `repetition_loop_in_generation.txt`
- `errors_in_the_langbridge_evaluation_process.txt`
- `untitled_empty_note.txt`: an empty file originally named `.txt`.

`notes/v3_data/` holds the v3 dataset citations (now in [docs/data.md](../docs/data.md)) and the list of datasets to consider (now in [docs/roadmap.md](../docs/roadmap.md)).

**`misc/tempCodeRunnerFile.py`**: an editor scratch file (VS Code Code Runner) from the v3 data folder, containing a single line. It is kept only because nothing is deleted in this overhaul.
