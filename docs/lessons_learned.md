# Lessons learned

Problems hit while building **LangBridge v1** and **LangBridge AI Igbo 1.0** (internal: v3), written as findings. Each entry has four parts: **Problem**, **Cause**, **Fix** and **What I learned**.

The original working notes are kept in [`archive/notes/`](../archive/notes). Entries marked *(from notes)* come from those files. The others come from notebook outputs, the code, or the repository overhaul audit.

---

## 1. Out-of-memory error when loading the model *(from notes)*

**Problem:** Loading `facebook/nllb-200-distilled-600M` onto one GPU crashed with CUDA out-of-memory, before or at the first forward pass.

**Cause:** `.to("cuda")` puts every layer on a single device. On a 16 GB Tesla T4 there was not enough room for the model plus activations, even with 4-bit weights.

**Fix:** Load with `device_map="balanced"` (or `"auto"`), so Hugging Face Accelerate spreads the layers across the available GPUs. Combine this with 4-bit NF4 loading. Later notebooks also cap generation at chunks of 100 sentences in batches of 8 ("GPU OOMs above this").

**What I learned:** Plan memory before loading. Check `nvidia-smi`, use `device_map`, and treat batch and chunk size as settings you tune. The note also warns that 4-bit evaluation can cost 1–2 BLEU points compared with full precision. Both Igbo 1.0 and its baseline were scored with the base model in 4-bit, while CPU deployment will run in float32 or int8.

## 2. Repetition loops in seq2seq generation *(from notes)*

**Problem:** Some translations repeated the same phrase until `max_new_tokens` ran out. Those sentences got near-zero scores and dragged the average down.

**Cause:** Beam search or greedy decoding can get stuck in a high-probability loop. Nothing in plain decoding penalises a phrase it has already produced.

**Fix:** Add `no_repeat_ngram_size=3` to `generate()`, which bans any 3-token sequence from appearing twice. If loops remain, add `repetition_penalty≈1.3`. All later evaluations use `no_repeat_ngram_size=3`.

**What I learned:** Read a sample of raw outputs before computing any corpus-level score. Repetition is obvious by eye and silently corrupts averages.

## 3. Errors in the evaluation process *(from notes, plus audit)*

**Problem:** Evaluation runs crashed (out of memory, item 1), produced loops (item 2), and broke when building PyTorch datasets from tokenizer output.

**Cause:** The note says `__len__` in a custom `Dataset` needs tensors. Without `padding=True, return_tensors="pt"` the encodings are Python lists, and `.shape` does not exist. The audit also found that the saved state of `notebooks/03_evaluation.ipynb` is not consistent: two adapters are attached to one base model, `.eval()` is called on one and `.generate()` on the other, and the sample output pairs an Igbo source with an English reference.

**Fix:** The v1 notebook tokenized with `padding=True, return_tensors="pt"`. The v3 notebook uses `len(encodings["input_ids"])` and pads per batch with `DataCollatorForSeq2Seq`. The notebook state still needs a clean, one-model-per-run pass (see [evaluation.md](evaluation.md#reproducing-the-igbo-10-numbers)).

**What I learned:** An evaluation notebook should be one function per run: model id, direction, source column and reference column as explicit arguments. Save the generated JSON next to the score. Then you can audit any number later.

## 4. Missing `forced_bos_token_id` / `src_lang`

**Problem:** NLLB needs two language signals. The **source** tag (`tokenizer.src_lang`) is added to the input. The **target** tag (`forced_bos_token_id`) forces the first generated token to be the target language. If either is wrong or missing, the model reads the wrong source language or answers in the wrong language.

**Cause (from the code):**

- The v1 training notebook loads `AutoTokenizer.from_pretrained(model_name)` with no languages. Its sample-translation callback calls `generate()` without `forced_bos_token_id`.
- The saved v1 inference notebook never sets `src_lang`, so Hausa, Igbo and Yoruba inputs get the default `eng_Latn` tag. That bug can only hurt the **into-English** directions, which are the directions where v1 lost the most (Hausa → English −46.13%). Whether the NLLB baseline was scored with the same bug is unknown, so this is a **likely contributor, not a confirmed cause**.
- The old README used `tokenizer.lang_code_to_id[...]`, which newer transformers versions removed.

**Fix:** Igbo 1.0 sets `src_lang="eng_Latn"` and `tgt_lang="ibo_Latn"` on the tokenizer for training. It sets `forced_bos_token_id=tokenizer.convert_tokens_to_ids("ibo_Latn")` at evaluation.

**What I learned:** Check the first token of `input_ids` and `labels` in a real batch, every time. The v1 pipeline printed them (`256073…` for the source, `256047` = `eng_Latn` for labels), and that check is cheap.

## 5. Long training runs failing (the "22-hour" failures)

**Problem:** Long training scripts failed after many hours, and the work was lost.

**Cause:** No note in the repository records the exact failures (TBD). What the code shows:

- Kaggle sessions have time limits.
- The v1 notebook evaluated with generation on a 93,319-row validation set every 500 steps. That is very slow on a T4 **(guess: a likely reason runs stretched past the limit)**.
- The v1 callback typo (`model.devic`) would crash the run at the first evaluation.
- The LR range test alone took 9 h 45 min.

**Fix (visible in the v3 code):**

- Training runs as a background subprocess with logs written to disk (`ddp_debug.log`).
- Metrics are written to JSONL after every log step.
- Checkpoints are saved every 167 steps with no deletion limit.
- The LR finder wrapper saves its history every 25 iterations and can redraw the plot after a crash ("crash-tolerant … built for Kaggle").
- The v1 notebook already wrapped `trainer.train()` in `try/except` to save a salvage adapter.

**What I learned:** Assume a long run will die. Write progress to disk often, keep evaluation cheap (a small validation set, loss rather than generation), and make every run resumable from its last checkpoint.

## 6. Failed DDP run: `decoder_input_ids` and `decoder_inputs_embeds`

**Problem:** The saved run of `notebooks/02_training_runner_kaggle.ipynb` (2026-11-05) crashed at step 0 of 4,710 on both GPUs with `ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time`.

**Cause:** Not recorded. The run used transformers 5.16.1. The adapter that was released had been trained before this, on 2026-11-03 or earlier, probably with an older library version **(guess)**.

**Fix:** TBD. Next step: re-run with the library versions from the successful run, and record `pip freeze` in every run.

**What I learned:** Pin library versions for training runs. `pip install -U` at the top of a notebook can change behaviour between runs.

## 7. Test split overlaps the training split *(audit, 2026-11-15)*

**Problem:** 1,506 of the 1,693 test rows (88.95%) have their exact English–Igbo pair in the training split.

**Cause:** MAFAND-MT dev (1,500 rows) was used to build the test split, and it also ended up in the training split. The cross-source de-duplication only compared some file pairs, and only ever removed rows from the second file.

**Fix:** Added [`scripts/check_split_overlap.py`](../scripts/check_split_overlap.py). Planned: rebuild the splits with a final check that test ∩ train = ∅, then re-score (see [roadmap.md](roadmap.md)).

**What I learned:** Run an automatic overlap check between every pair of splits as the last step of building any dataset, and again before reporting results.

## 8. Renamed split files and swapped sizes

**Problem:** The Hugging Face cards list Validation = 1,693 and Test = 597. The files have the opposite sizes.

**Cause:** The files were renamed twice on 2026-09-30, and the card text was not updated.

**Fix:** This repository uses the file sizes. The card text needs updating (listed in the PR).

**What I learned:** Generate split-size tables from the files, not by hand.

## 9. The LR finder's automatic suggestion was unusable

**Problem:** `torch-lr-finder` suggested a learning rate of 0.2287.

**Cause:** It picks the steepest point of the loss curve. At that learning rate the loss was already NaN.

**Fix:** The curve was read by hand. The loss started to fall at about 1e-5 and was lowest at 5.14e-4. `base_lr=2e-5` and `max_lr=3.3e-4` were chosen from that range.

**What I learned:** Always plot the curve and choose by eye. Automatic suggestions are only a starting point.

## 10. Hard-coded Hugging Face token in a notebook

**Problem:** `bundling_model_for_prod.ipynb` contained a Hugging Face access token in plain text. A second token sat in `langbridge_ai_v3_data/secrets.env`.

**Cause:** A quick test overwrote the token read from Kaggle secrets.

**Fix:** The token was removed from the notebook before it was added to git. `secrets.env` is git-ignored and was never committed. Both tokens must be revoked on Hugging Face.

**What I learned:** Read secrets only from the environment or the platform's secret store, and scan for `hf_` before every commit.

## 11. CTranslate2 conversion failed in the bundling notebook

**Problem:** Converting the merged model to CTranslate2 failed with `merged-eng-ibo is not a local folder and is not a valid model identifier`.

**Cause:** The output folder was renamed (`merged_eng_igbo_v1`) but the converter ran with the old name. The next line passes an undefined variable (`merged_eng_igbo_v1` without quotes) where the base model path should be. The merge cell also refers to `base_model_id`, but the variable is named `c`.

**Fix:** TBD. Use one constant for each path and re-run the notebook top to bottom.

**What I learned:** Restart and run all cells before trusting a notebook. Leftover state hides broken cells.

## 12. Empty note file

The working-notes folder held an empty file named `.txt`. It had no content to merge, so it is kept as [`archive/notes/untitled_empty_note.txt`](../archive/notes/untitled_empty_note.txt).
