# Engineering Log

Bugs, failures and findings from building LangBridge AI, recorded in one
standard format. Each entry explains what happened, why, how it was fixed,
and what changed so it doesn't happen again.

## How to read this log

- **Type:** Bug (something broke) or Finding (something we learned from results or data).
- **Severity:** how much it hurt the project (scale below).
- **Status:** where it stands today.

| Severity | Meaning |
|---|---|
| Critical | Blocked training or evaluation completely, or produced wrong published results |
| High | Wasted a full training run or many GPU hours, or gave misleading metrics |
| Medium | Slowed work down; a workaround existed |
| Low | Minor issue with a quick fix |

| Status | Meaning |
|---|---|
| Resolved | Fixed and checked |
| Workaround | Avoided, but the root cause is still there |
| Open | Not fixed yet |
| Won't fix | Accepted on purpose, with a reason |

Entries are grouped by version (LangBridge v1, then LangBridge AI Igbo 1.0). Within a version, dated entries come first, oldest first, followed by entries whose date was not recorded. IDs are permanent: an ID never changes and is never reused, and new entries take the next number.

Names: **LangBridge v1** is the archived multilingual model (NLLB-200-distilled-600M). **LangBridge AI Igbo 1.0** (internal: v3) is the current English → Igbo model (NLLB-200-distilled-1.3B). No entries were recorded for the intermediate v2.

## Index

<!-- INDEX:START -->
| ID | Title | Type | Category | Severity | Status | Version |
|---|---|---|---|---|---|---|
| [LB-001](#lb-001-v1-sample-translation-callback-has-a-typo-that-crashes-the-first-evaluation) | v1 sample-translation callback has a typo that crashes the first evaluation | Bug | Training | Medium | Won't fix | v1 |
| [LB-002](#lb-002-out-of-memory-error-when-loading-nllb-600m-on-one-gpu) | Out-of-memory error when loading NLLB-600M on one GPU | Bug | Environment | Medium | Resolved | v1 |
| [LB-003](#lb-003-missing-src_lang-and-forced_bos_token_id-setup-in-v1) | Missing src_lang and forced_bos_token_id setup in v1 | Bug | Evaluation | High | Resolved | v1 |
| [LB-004](#lb-004-repetition-loops-in-seq2seq-generation) | Repetition loops in seq2seq generation | Bug | Inference | Medium | Resolved | v1 |
| [LB-005](#lb-005-errors-in-the-evaluation-process) | Errors in the evaluation process | Bug | Evaluation | Medium | Workaround | v1 |
| [LB-006](#lb-006-long-kaggle-training-runs-failed-late) | Long Kaggle training runs failed late | Bug | Environment | High | Workaround | v1 |
| [LB-007](#lb-007-copy-paste-error-in-the-v1-results-sheet) | Copy-paste error in the v1 results sheet | Finding | Evaluation | High | Open | v1 |
| [LB-008](#lb-008-tokenizerlang_code_to_id-removed-in-newer-transformers) | tokenizer.lang_code_to_id removed in newer transformers | Bug | Inference | Low | Resolved | v1 |
| [LB-009](#lb-009-ddp-find_unused_parameters-warning-on-every-training-step) | DDP find_unused_parameters warning on every training step | Bug | Training | Low | Resolved | Igbo 1.0 |
| [LB-010](#lb-010-the-lr-finders-suggested-learning-rate-was-unusable) | The LR finder's suggested learning rate was unusable | Finding | Training | Low | Resolved | Igbo 1.0 |
| [LB-011](#lb-011-second-training-attempt-crashed-at-step-0-with-a-decoder-input-conflict) | Second training attempt crashed at step 0 with a decoder input conflict | Bug | Training | High | Open | Igbo 1.0 |
| [LB-012](#lb-012-most-test-sentences-also-appear-in-the-training-split) | Most test sentences also appear in the training split | Finding | Data | Critical | Open | Igbo 1.0 |
| [LB-013](#lb-013-validation-and-test-sizes-swapped-on-the-hugging-face-cards) | Validation and test sizes swapped on the Hugging Face cards | Bug | Data | Low | Open | Igbo 1.0 |
| [LB-014](#lb-014-hugging-face-token-hard-coded-in-the-bundling-notebook) | Hugging Face token hard-coded in the bundling notebook | Bug | Deployment | High | Open | Igbo 1.0 |
| [LB-015](#lb-015-tied-weight-conflict-between-transformers-versions-and-a-cublas-error) | Tied-weight conflict between transformers versions and a CUBLAS error | Bug | Training | High | Resolved | Igbo 1.0 |
| [LB-016](#lb-016-gradient-checkpointing-warning-about-use_reentrant) | Gradient checkpointing warning about use_reentrant | Bug | Training | Low | Resolved | Igbo 1.0 |
| [LB-017](#lb-017-fp16-gradient-scaling-failed-on-bf16-gradients) | FP16 gradient scaling failed on BF16 gradients | Bug | Training | High | Resolved | Igbo 1.0 |
| [LB-018](#lb-018-expected-float-got-half-with-an-fp16-model-and-fp16-compute) | Expected Float, got Half with an fp16 model and fp16 compute | Bug | Training | High | Resolved | Igbo 1.0 |
| [LB-019](#lb-019-out-of-memory-on-2-t4-at-512-tokens-with-fp32-activations) | Out-of-memory on 2× T4 at 512 tokens with fp32 activations | Bug | Training | High | Resolved | Igbo 1.0 |
| [LB-020](#lb-020-data-collator-failed-on-leftover-text-columns-in-the-lr-range-test) | Data collator failed on leftover text columns in the LR range test | Bug | Data | Medium | Resolved | Igbo 1.0 |
| [LB-021](#lb-021-lr-finder-passed-the-input-dictionary-to-nllb-as-one-argument) | LR finder passed the input dictionary to NLLB as one argument | Bug | Training | Medium | Resolved | Igbo 1.0 |
| [LB-022](#lb-022-target-padding-tokens-were-counted-in-the-training-loss) | Target padding tokens were counted in the training loss | Bug | Data | High | Resolved | Igbo 1.0 |
| [LB-023](#lb-023-igbo-10-is-worse-than-its-base-model-on-igbo-to-english) | Igbo 1.0 is worse than its base model on Igbo to English | Finding | Evaluation | Medium | Open | Igbo 1.0 |
| [LB-024](#lb-024-ctranslate2-conversion-failed-in-the-bundling-notebook) | CTranslate2 conversion failed in the bundling notebook | Bug | Deployment | Medium | Open | Igbo 1.0 |
<!-- INDEX:END -->

## Entries

---

### LB-001: v1 sample-translation callback has a typo that crashes the first evaluation

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | Medium |
| Status | Won't fix |
| Version | v1 |
| Date found | 2026-11-15 |
| Environment | Kaggle T4 (single GPU), Hugging Face `Seq2SeqTrainer`; library versions not recorded |

**Summary.** The custom callback that prints sample translations during v1 training reads `model.devic` instead of `model.device`. It would raise an error at the first evaluation step.

**Symptoms.** No run output was saved. The problem was found by reading the code:

```python
device = model.devic
```

**How to reproduce.** Run the v1 training notebook with `eval_steps=500`. The `on_evaluate` callback runs at step 500 and fails with an `AttributeError`.

**Impact.** Not confirmed. `trainer.train()` is wrapped in `try/except`, which saves a "crash salvage" adapter. If the published v1 adapter came from such a run, it saw at most 500 steps × 32 = 16,000 of 3,000,000 examples. This is a hypothesis; no logs exist.

**Root cause.** A typo in the attribute name.

**Fix.** None. v1 is archived. The current training script has no custom generation callback.

**Verification.** Not applicable.

**Prevention.** None yet. A short smoke run that reaches the first evaluation step (for example `max_steps` set just past `eval_steps`) would catch this kind of error in minutes.

**Lesson.** Code that only runs at evaluation time stays untested until the first evaluation. Trigger every callback once before a long run.

**References.** `archive/v1/training/lang_bridge_training_process.ipynb` (cell 15); [experiments.md, E1](experiments.md#e1-langbridge-v1-training).

---

### LB-002: Out-of-memory error when loading NLLB-600M on one GPU

| Field | Value |
|---|---|
| Type | Bug |
| Category | Environment |
| Severity | Medium |
| Status | Resolved |
| Version | v1 |
| Date found | Not recorded (note file last modified 2026-03-19) |
| Environment | Tesla T4 (16 GB), 4-bit quantization (bitsandbytes); versions not recorded |

**Summary.** Loading `facebook/nllb-200-distilled-600M` with `.to("cuda")` put the whole model on one GPU and ran out of memory.

**Symptoms.** A CUDA out-of-memory (OOM) error during model loading or the first forward pass. GPU memory was full straight after loading, before any data was processed.

**How to reproduce.** `AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")` on a single 16 GB T4, then run a forward pass.

**Impact.** Training and evaluation crashed before processing any data.

**Root cause.** `.to("cuda")` places every layer on one device. Even with 4-bit weights, the model plus activations did not fit on that device.

**Fix.** Let Hugging Face Accelerate place the layers:

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="balanced")
```

`device_map="balanced"` (or `"auto"`) spreads the layers across the available GPUs.

**Verification.** Not recorded. Later notebooks load models with `device_map` and run.

**Prevention.** Load large models with `device_map` and 4-bit NF4 by default. Check free memory with `nvidia-smi` first. Evaluation notebooks cap generation at chunks of 100 sentences ("GPU OOMs above this").

**Lesson.** Plan memory before loading. The note also warns that 4-bit evaluation can cost 1–2 BLEU points compared with full precision.

**References.** `archive/notes/oom_error_when_loading_nllb.txt`; `archive/notes/errors_in_the_langbridge_evaluation_process.txt`.

---

### LB-003: Missing src_lang and forced_bos_token_id setup in v1

| Field | Value |
|---|---|
| Type | Bug |
| Category | Evaluation |
| Severity | High |
| Status | Resolved |
| Version | v1 |
| Date found | Not recorded |
| Environment | Kaggle T4, transformers (version not recorded) |

**Summary.** NLLB needs two language signals: the source tag (`tokenizer.src_lang`) and the target tag (`forced_bos_token_id`). Parts of the v1 code did not set them, so the model was not told which language it was reading or which to write.

**Symptoms.** From the code:

- The v1 training notebook loads `AutoTokenizer.from_pretrained(model_name)` with no languages, and its sample-translation callback calls `generate()` without `forced_bos_token_id`.
- The saved v1 inference notebook never sets `src_lang`, so Hausa, Igbo and Yoruba inputs get the default `eng_Latn` tag.

**How to reproduce.** Tokenize Hausa text with a default NLLB tokenizer and print `input_ids[0]`. The first token is the `eng_Latn` tag.

**Impact.** Not measured. A wrong source tag can only affect the into-English directions, which are where v1 lost the most (Hausa → English −46.13%). Hypothesis: this is a likely contributor to the v1 results. It was not confirmed, because it is unknown whether the baseline was scored with the same notebook.

**Root cause.** Tokenizer and generation settings left at their defaults.

**Fix.** LangBridge AI Igbo 1.0 sets both tags everywhere:

```python
tokenizer = NllbTokenizer.from_pretrained(model_id, src_lang="eng_Latn", tgt_lang="ibo_Latn")
forced_bos_token_id = tokenizer.convert_tokens_to_ids("ibo_Latn")
```

**Verification.** The v3 evaluation notebook passes `forced_bos_token_id` on every `generate()` call. v1 was not re-scored.

**Prevention.** Print the first token of `input_ids` and `labels` for a real batch before training or evaluation.

**Lesson.** Multilingual models fail quietly when language tags are wrong. Check the tags in every stage, not only in training.

**References.** `archive/v1/training/lang_bridge_training_process.ipynb`; `archive/v1/evaluation/langbridge-ai-inference-datapipeline.ipynb` (cell 8); [evaluation.md](evaluation.md#3-langbridge-v1-vs-nllb-200-distilled-600m-archived).

---

### LB-004: Repetition loops in seq2seq generation

| Field | Value |
|---|---|
| Type | Bug |
| Category | Inference |
| Severity | Medium |
| Status | Resolved |
| Version | v1 |
| Date found | Not recorded (note file last modified 2026-03-20) |
| Environment | NLLB-200, beam search or greedy decoding; versions not recorded |

**Summary.** Some translations repeated the same phrase until the token limit. Those sentences scored near zero and pulled the averages down.

**Symptoms.** Generated token IDs showed a repeating block before the end-of-sequence token. The decoded text was one fragment repeated many times. BLEU on those sentences collapsed, especially the 2-, 3- and 4-gram precision.

**How to reproduce.** Generate with NLLB using `max_new_tokens=100` and no repetition control, then inspect outputs for repeated fragments. Exact inputs were not recorded.

**Impact.** Corrupted corpus-level scores without any error message.

**Root cause.** Decoding can get stuck in a high-probability loop. Plain decoding has nothing that penalises repeating an earlier phrase.

**Fix.** Ban repeated 3-token sequences:

```python
model.generate(**inputs, forced_bos_token_id=forced_bos_token_id,
               max_new_tokens=100, no_repeat_ngram_size=3)
```

If loops remain, the note suggests adding `repetition_penalty=1.3`.

**Verification.** Not recorded. All later evaluation notebooks use `no_repeat_ngram_size=3`.

**Prevention.** `no_repeat_ngram_size=3` is the default in every evaluation notebook.

**Lesson.** Read a sample of raw outputs before computing corpus scores. Repetition is easy to see and silently skews averages.

**References.** `archive/notes/repetition_loop_in_generation.txt`; `archive/notes/errors_in_the_langbridge_evaluation_process.txt`; `notebooks/03_evaluation.ipynb`.

---

### LB-005: Errors in the evaluation process

| Field | Value |
|---|---|
| Type | Bug |
| Category | Evaluation |
| Severity | Medium |
| Status | Workaround |
| Version | v1 |
| Date found | Not recorded (note file last modified 2026-06-27) |
| Environment | Kaggle T4; versions not recorded |

**Summary.** Evaluation runs hit several problems: out-of-memory crashes (LB-002), repetition loops (LB-004), and a dataset class that broke on untensored tokenizer output.

**Symptoms.** A custom PyTorch `Dataset` failed in `__len__` because the encodings were Python lists, which have no `.shape`.

**How to reproduce.** Tokenize without `padding=True, return_tensors="pt"`, then call a `__len__` that reads `encodings["input_ids"].shape[0]`.

**Impact.** Evaluation runs had to be restarted. Exact time lost was not recorded.

**Root cause.** Lists and tensors were mixed: the dataset class expected tensors.

**Fix.** The v1 notebook tokenizes with `padding=True, return_tensors="pt"`. The v3 notebook uses `len(encodings["input_ids"])` and pads each batch with `DataCollatorForSeq2Seq`. Generation runs in chunks of 100 sentences with batch size 8.

**Verification.** The v3 evaluation notebook completed all 17 chunks for 1,693 sentences (saved output).

**Prevention.** None yet. Still open: the saved state of `notebooks/03_evaluation.ipynb` attaches two adapters to one base model, and its sample output pairs an Igbo source with an English reference. One clean run per model and direction is needed before its numbers can be reproduced.

**Lesson.** Write evaluation as one function per run, with model, direction, source column and reference column as arguments. Save the generated text next to the score.

**References.** `archive/notes/errors_in_the_langbridge_evaluation_process.txt`; `notebooks/03_evaluation.ipynb`; [evaluation.md](evaluation.md#reproducing-the-igbo-10-numbers).

---

### LB-006: Long Kaggle training runs failed late

| Field | Value |
|---|---|
| Type | Bug |
| Category | Environment |
| Severity | High |
| Status | Workaround |
| Version | v1 |
| Date found | Not recorded |
| Environment | Kaggle GPU sessions (T4) |

**Summary.** Long training scripts, reported as running for about 22 hours, failed late, and the work was lost.

**Symptoms.** Not recorded. No error message or log from these failures was kept.

**How to reproduce.** Not recorded.

**Impact.** Full training runs and many GPU hours lost. Exact counts not recorded.

**Root cause.** Not confirmed. Hypotheses from the code:

- Kaggle sessions have time limits.
- The v1 notebook evaluated with generation on a 93,319-row validation set every 500 steps, which is very slow on a T4.
- The callback typo in LB-001 would crash at the first evaluation.

**Fix.** Later code is built to survive crashes:

- Training runs as a background subprocess, with logs written to disk (`ddp_debug.log`).
- Metrics are written to JSONL after every log step.
- Checkpoints are saved every 167 steps with no deletion limit.
- The LR finder saves its history every 25 iterations.

**Verification.** Not recorded for training. The LR range test completed a 9 h 45 min run (`results/lr_range_test/result.json`).

**Prevention.** Frequent checkpoints and on-disk logs. A small validation set scored by loss rather than by generation.

**Lesson.** Assume a long run will die. Save progress often, keep evaluation cheap, and make every run resumable.

**References.** `archive/v1/training/lang_bridge_training_process.ipynb`; `notebooks/02_training_runner_kaggle.ipynb`; `notebooks/01_lr_range_test_cyclic_lr.ipynb`.

---

### LB-007: Copy-paste error in the v1 results sheet

| Field | Value |
|---|---|
| Type | Finding |
| Category | Evaluation |
| Severity | High |
| Status | Open |
| Version | v1 |
| Date found | Not recorded (sheet committed 2026-06-25; checked 2026-11-15) |
| Environment | AfriCOMET (`masakhane/africomet-mtl`), unbabel-comet 2.2.7 |

**Summary.** In the v1 results sheet, the NLLB-600M Yoruba → English baseline scores are exact copies of the NLLB-600M Igbo → English scores.

**Observation.** All three Yoruba → English values (0.4021051446907222, 0.4159341849889606, 0.40285637013643283) match Igbo → English digit for digit. The saved output of the v1 AfriCOMET notebook shows that 0.4021051446907222 came from `nllb_igbo_eng_comet_eval_data_0_100.json`, an Igbo → English file. No notebook output, note or commit holds a separate Yoruba → English baseline.

**Impact.** The v1 comparison for Yoruba → English has no valid baseline. Shown as published, the row would have given a false change value.

**Root cause.** Not confirmed. Most likely the Igbo → English rows were copied as a template and not overwritten (hypothesis).

**Decision.** The value is marked TBD in every CSV, table and chart, and the copied numbers are not shown anywhere. The Yoruba test file still exists in a local Hugging Face cache, so the baseline could be recomputed.

**Verification.** `results/v1_vs_nllb600m.csv` and both v1 summary tables show TBD.

**Prevention.** Generate results tables from saved score files by script, never by hand.

**Lesson.** Identical numbers in two rows of a results table are a warning sign. Check every value against its source file.

**References.** `archive/v1/langbridge_ai_inference_sheet.md`; `archive/v1/evaluation/afri-comet-evaluation-for-lang-bridge-ai.ipynb`; [`results/v1_vs_nllb600m.csv`](../results/v1_vs_nllb600m.csv); [evaluation.md, Issue 1](evaluation.md#issue-1-copied-yoruba--english-baseline).

---

### LB-008: tokenizer.lang_code_to_id removed in newer transformers

| Field | Value |
|---|---|
| Type | Bug |
| Category | Inference |
| Severity | Low |
| Status | Resolved |
| Version | v1 |
| Date found | Not recorded |
| Environment | transformers versions that no longer provide `lang_code_to_id` (exact version not recorded) |

**Summary.** The usage example in the old v1 README gets the target language ID with `tokenizer.lang_code_to_id[...]`. Newer transformers versions removed that attribute, so the example breaks.

**Symptoms.** The README example fails at this line:

```python
forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
```

The exact error text was not recorded.

**How to reproduce.** Run the old README example with a recent transformers release.

**Impact.** Anyone copying the old README example could not translate.

**Root cause.** An API change in transformers. The old README was written for an earlier version.

**Fix.** Look up the language token by name:

```python
forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
```

The old README was archived as `archive/v1/README_v1.md` and is not used.

**Verification.** The v3 evaluation notebook uses `convert_tokens_to_ids` and ran successfully (saved output).

**Prevention.** None yet. Docs code should come from a file that is run.

**Lesson.** Code in a README goes stale when libraries change. Prefer stable public APIs, and run docs examples when you upgrade.

**References.** `archive/v1/README_v1.md`; `notebooks/03_evaluation.ipynb`.

---

### LB-009: DDP find_unused_parameters warning on every training step

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | Low |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | 2026-10-08 (log timestamp `W1008`; year from the project timeline) |
| Environment | Kaggle 2× T4, PyTorch DDP via `accelerate` (ranks 0 and 1) |

**Summary.** **DDP** (Distributed Data Parallel: one model copy per GPU) was started with `find_unused_parameters=True`, but no parameters were unused. PyTorch warned that this adds extra work on every step.

**Symptoms.**

```
[rank1]:[W1008 13:10:13.455936698 reducer.cpp:1502] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance.
{'loss': '102.3', 'grad_norm': '17.91', 'learning_rate': '2e-05', 'epoch': '0.001998'}
  0%|          | 1/1503 [00:11<4:57:58, 11.90s/it]
```

**How to reproduce.** Launch training on two GPUs with `find_unused_parameters=True` for a model where all parameters are used. The exact launch settings of the logged run were not recorded.

**Impact.** Slower steps. The size of the slowdown was not recorded.

**Root cause.** The DDP flag was set to look for unused parameters, which this model does not have.

**Fix.** Not recorded in the notes. The current training script sets:

```python
ddp_find_unused_parameters=False,
```

**Verification.** Not recorded.

**Prevention.** The setting is fixed in `src/langbridge/train.py`.

**Lesson.** Read the warnings at the top of a distributed run. Some only cost speed, but they repeat on every step. The loss of 102.3 in this log is a symptom of LB-022.

**References.** Notion working notes, "ANOMALY BUGS" (DDP log); `src/langbridge/train.py`.

---

### LB-010: The LR finder's suggested learning rate was unusable

| Field | Value |
|---|---|
| Type | Finding |
| Category | Training |
| Severity | Low |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | 2026-10-14 |
| Environment | Kaggle T4 (one GPU), `torch-lr-finder`, NLLB-1.3B 4-bit + rsLoRA |

**Summary.** An **LR range test** trains briefly while raising the learning rate, to find a good range. `torch-lr-finder` suggested 0.2287, a value where the loss had already become NaN (not a number).

**Observation.** From `results/lr_range_test/result.json`:

- 501 iterations, learning rate from 1e-7 to 5.
- The loss started to fall near 1e-5 and was lowest (1.9904) at 5.14e-4.
- The loss became NaN from about 0.237.
- The tool's suggestion (steepest slope) was 0.2287.

**Impact.** Using the suggestion would have made training diverge immediately.

**Root cause.** The tool picks the steepest point of the curve without checking whether the loss there is still finite.

**Decision.** The curve was read by hand. CyclicLR uses `base_lr=2e-5` and `max_lr=3.3e-4`, inside the range where loss falls.

**Verification.** Training with these values produced a released checkpoint (epoch 1.333666).

**Prevention.** The LR finder wrapper saves the plot and full history every run, so the curve can always be inspected.

**Lesson.** Treat automatic learning-rate suggestions as a starting point. Always look at the curve.

**References.** [`results/lr_range_test/`](../results/lr_range_test); `notebooks/01_lr_range_test_cyclic_lr.ipynb`; [training.md](training.md#how-the-learning-rate-was-chosen).

---

### LB-011: Second training attempt crashed at step 0 with a decoder input conflict

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | High |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | 2026-11-05 |
| Environment | Kaggle 2× T4, transformers 5.16.1, accelerate 1.15.0 |

**Summary.** A later training run with the Igbo 1.0 script crashed at step 0 of 4,710 on both GPUs.

**Symptoms.**

```
[rank1]: ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time
  0%|          | 0/4710 [00:02<?, ?it/s]
```

**How to reproduce.** Run `notebooks/02_training_runner_kaggle.ipynb` with transformers 5.16.1 and accelerate 1.15.0.

**Impact.** The run produced no training. The released adapter (uploaded 2026-11-03) came from an earlier run and is not affected.

**Root cause.** Not confirmed. Hypothesis: a change in library behaviour, since the runner installs the latest `accelerate` and `bitsandbytes` at the top of the notebook.

**Fix.** Not recorded.

**Verification.** Not applicable.

**Prevention.** None yet. Planned: pin library versions and save `pip freeze` with every run.

**Lesson.** `pip install -U` at the top of a notebook can change behaviour between runs. Pin the versions of a run that worked.

**References.** `notebooks/02_training_runner_kaggle.ipynb` (saved output); [training.md](training.md#known-problem-with-the-saved-runner); [experiments.md, E9](experiments.md#e9-second-training-attempt-failed).

---

### LB-012: Most test sentences also appear in the training split

| Field | Value |
|---|---|
| Type | Finding |
| Category | Data |
| Severity | Critical |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | 2026-11-15 |
| Environment | `coded-by-49/langbridge_ai_v3_igbo_datasets` (Hugging Face) |

**Summary.** 1,506 of the 1,693 test rows (88.95%) have their exact English–Igbo pair in the training split. The published English → Igbo gain therefore mostly measures sentences the model was trained on.

**Observation.** All 1,500 test rows that come from MAFAND-MT dev are also in the training split. Only 187 test rows (mostly from Ezeani et al.) are not. Validation has no overlap with training.

| Split | Rows | Rows also in train |
|---|---|---|
| Validation (used in training) | 589 | 0 (0.00%) |
| Test | 1,693 | 1,506 (88.95%) |

**Impact.** The headline result (+1.88% AfriCOMET on English → Igbo) and the Hugging Face card's description of a held-out test set are not supported as published.

**Root cause.** MAFAND-MT dev was used to build the test split and was also included in the training split. The de-duplication step only compared some pairs of files.

**Decision.** Keep the recorded scores and add a warning wherever they appear. Re-score on the 187 clean rows and the validation split, then rebuild the splits with zero overlap.

**Verification.** Reproducible with `scripts/check_split_overlap.py` against the live Hugging Face files.

**Prevention.** None yet. Planned: run `scripts/check_split_overlap.py` as the last step of every dataset build and before reporting results.

**Lesson.** Check train/test overlap automatically. It is cheap to test and costly to miss.

**References.** [`results/v3_split_overlap.csv`](../results/v3_split_overlap.csv); [`scripts/check_split_overlap.py`](../scripts/check_split_overlap.py); [evaluation.md, Issue 4](evaluation.md#issue-4-test-split-overlaps-the-training-split).

---

### LB-013: Validation and test sizes swapped on the Hugging Face cards

| Field | Value |
|---|---|
| Type | Bug |
| Category | Data |
| Severity | Low |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | 2026-11-15 |
| Environment | Hugging Face dataset and model cards |

**Summary.** Both cards list Validation = 1,693 and Test = 597. The published files have the opposite sizes.

**Symptoms.** `igbo_eng_lv1_test_.jsonl` has 1,693 rows. `igbo_eng_lv1_validation.jsonl` has 597 rows (589 after the 256-token filter).

**How to reproduce.** Count the lines of each split file on Hugging Face and compare with the card tables.

**Impact.** Readers of the cards get the wrong size for each split.

**Root cause.** The split files were renamed twice on 2026-09-30 (dataset commit history). The card text was not updated.

**Fix.** This repository uses the file sizes. The card text still needs updating.

**Verification.** Not applicable until the cards are updated.

**Prevention.** None yet. Generate split-size tables from the files.

**Lesson.** Tables written by hand drift away from the data. Generate them.

**References.** [evaluation.md, Issue 5](evaluation.md#issue-5-validation-and-test-sizes-swapped-on-the-hugging-face-cards); [data.md](data.md#splits).

---

### LB-014: Hugging Face token hard-coded in the bundling notebook

| Field | Value |
|---|---|
| Type | Bug |
| Category | Deployment |
| Severity | High |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | 2026-11-15 |
| Environment | Kaggle notebook (production bundling) |

**Summary.** The bundling notebook contained a Hugging Face access token in plain text. A second token was stored in a local `secrets.env` file.

**Symptoms.** One cell overwrote the token read from Kaggle secrets with a literal string. The token value is not repeated here.

**How to reproduce.** Not applicable.

**Impact.** Anyone with access to the notebook could use the token. Neither token was ever committed to git.

**Root cause.** A quick test replaced the secret-store lookup with a pasted value.

**Fix.** The literal token was removed before the notebook was added to git. The notebook now relies on the Kaggle secret `HF_TOKEN`. `secrets.env` is git-ignored. Both tokens still have to be revoked on Hugging Face, which is why the status is Open.

**Verification.** A scan of all tracked files and of the git history found no Hugging Face tokens.

**Prevention.** Read secrets only from the environment or the platform's secret store, and scan for `hf_` before every commit.

**Lesson.** A token in a notebook is a published token. Revoke it, then remove it.

**References.** `notebooks/05_bundling_for_production.ipynb`; `.gitignore`.

---

### LB-015: Tied-weight conflict between transformers versions and a CUBLAS error

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | High |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | NLLB-200-distilled-1.3B in 4-bit; transformers 5.0.0 and 4.57.6 |

**Summary.** The model config says to tie (share) the embedding and output weights. The checkpoint stores them separately. The two transformers versions handled this differently, and a CUBLAS error followed under 4.57.6.

**Symptoms.** Transformers 5.0.0 printed (cut to one of three similar lines):

```
The tied weights mapping and config for this model specifies to tie model.shared.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them.
```

The same message appeared for `model.decoder.embed_tokens.weight` and `model.encoder.embed_tokens.weight`. Under transformers 4.57.6 the config's `tie_word_embeddings=True` was followed, and a CUBLAS error occurred later. CUBLAS is NVIDIA's GPU linear-algebra library. Its exact text was not recorded.

**How to reproduce.** Load the quantized model with transformers 4.57.6 and the default config, then train.

**Impact.** Training failed with the CUBLAS error. Time lost was not recorded.

**Root cause.** The config and the checkpoint disagree about weight tying. Whether wrong tying caused the CUBLAS error is not confirmed; the notes say it was suspected but not isolated.

**Fix.**

```python
model_config = AutoConfig.from_pretrained(model_id)
model_config.tie_word_embeddings = False
```

**Verification.** With `tie_word_embeddings=False`, the separate weights were kept and the CUBLAS error did not occur again.

**Prevention.** The setting is in `src/langbridge/train.py`.

**Lesson.** When a loader warns about tied weights, decide explicitly. Library versions may make different default choices.

**References.** Notion working notes, "ANOMALY BUGS"; `src/langbridge/train.py`; `configs/igbo_1.0_qlora.yaml`.

---

### LB-016: Gradient checkpointing warning about use_reentrant

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | Low |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | PyTorch (version before 2.9; exact version not recorded) |

**Summary.** **Gradient checkpointing** saves memory by recomputing activations during the backward pass. PyTorch warned that its `use_reentrant` option should be passed explicitly and will become required.

**Symptoms.**

```
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception.
```

**How to reproduce.** Train without configuring gradient checkpointing in `Seq2SeqTrainingArguments`.

**Impact.** A warning only. It would become an error from PyTorch 2.9.

**Root cause.** Per the notes, gradient checkpointing had not been explicitly configured in the training arguments.

**Fix.** Added to `Seq2SeqTrainingArguments`:

```python
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
```

The notes record the first line. The current script also sets the second.

**Verification.** The warning stopped (notes).

**Prevention.** Both settings are in `src/langbridge/train.py`.

**Lesson.** Deprecation warnings are future errors. Fix them while they are still warnings.

**References.** Notion working notes, "ANOMALY BUGS"; `src/langbridge/train.py`.

---

### LB-017: FP16 gradient scaling failed on BF16 gradients

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | High |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | 4-bit quantized NLLB, mixed precision; versions not recorded |

**Summary.** Training used `fp16=True`, which turns on automatic gradient scaling. Gradients from the quantized model were in BF16, a format the scaling step does not support here, so training crashed.

**Symptoms.** A `NotImplementedError` raised from PyTorch's `_amp_foreach_non_finite_check_and_unscale_cuda` operation. The full error text was not recorded.

**How to reproduce.** Train with `fp16=True` while the quantized model produces BF16 gradients. The exact configuration was not recorded (hypothesis: a BF16 compute dtype).

**Impact.** Training could not start.

**Root cause.** The FP16 gradient-scaling path (`GradScaler`) met BF16 gradients, and that operation does not accept them.

**Fix.** Per the notes:

```python
fp16=False,
bf16=True,
```

**Verification.** Training then started and ran normally:

```
[54/4506 02:06 < 3:00:34, 0.41 it/s, Epoch 0.04/3]
```

**Prevention.** None recorded. Note that the current `src/langbridge/train.py` uses `fp16=True, bf16=False` with a float32 compute dtype, so no BF16 gradients reach the scaler. That later change is not recorded in the notes.

**Lesson.** The mixed-precision flag and the model's compute dtype must agree. Check gradient dtypes when changing either.

**References.** Notion working notes, "ANOMALY BUGS"; `src/langbridge/train.py`.

---

### LB-018: Expected Float, got Half with an fp16 model and fp16 compute

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | High |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | NLLB QLoRA on Kaggle 2× T4 |

**Summary.** Loading the model in fp16 with an fp16 4-bit compute dtype caused a dtype mismatch error.

**Symptoms.**

```
Expected Float, got Half
```

**How to reproduce.** Load NLLB with `torch_dtype=torch.float16` and `bnb_4bit_compute_dtype=torch.float16`, then train (step 3 of the notes' timeline).

**Impact.** Training stopped. It was one of five configurations tried before training ran (see LB-019).

**Root cause.** The notes say a layer inside NLLB requires fp32 and rejects fp16 inputs. The specific layer was not recorded (Not confirmed).

**Fix.** Load in fp32 with fp32 compute:

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float32,
                                bnb_4bit_use_double_quant=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config,
                                              torch_dtype=torch.float32)
```

**Verification.** The fp32 configuration ran (step 4 of the timeline), and it then hit out-of-memory (LB-019).

**Prevention.** `src/langbridge/train.py` loads in fp32 with fp32 compute.

**Lesson.** `bnb_4bit_compute_dtype` must match the model's loaded dtype. fp16 does not always work with NLLB + QLoRA on a T4.

**References.** Notion working notes, "ANOMALY BUGS" (Bug Log: NLLB QLoRA Fine-Tuning on 2× T4); `src/langbridge/train.py`.

---

### LB-019: Out-of-memory on 2× T4 at 512 tokens with fp32 activations

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | High |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | Kaggle 2× T4 (16 GB each), NLLB-1.3B 4-bit QLoRA |

**Summary.** Training ran out of GPU memory several times before the right combination of sequence length, dtype and batch size was found.

**Symptoms.** The notes' timeline:

| Step | Action | Result |
|---|---|---|
| 1 | fp32 model + fp32 compute | OOM at 512 tokens |
| 2 | `expandable_segments:True`, tokens reduced to 256 | OOM |
| 3 | fp16 model + fp16 compute | `Expected Float, got Half` (LB-018) |
| 4 | fp32 model + fp32 compute | Ran, then OOM again |
| 5 | Batch 4 → 3, accumulation 2 → 4 | Training runs |

**How to reproduce.** Train with 512-token sequences and a per-GPU batch of 4 on a T4.

**Impact.** Four failed configurations before a working one.

**Root cause.** A combination of fp32 activations (4 bytes per value), long sequences (512 tokens), and a batch too large for 16 GB.

**Fix.**

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
per_device_train_batch_size=3, gradient_accumulation_steps=4
```

Max length is 256 tokens. **Gradient accumulation** sums gradients over several small batches before each update. The effective batch is 3 × 4 × 2 GPUs = 24, up from 16.

**Verification.** Training ran end to end (notes).

**Prevention.** These settings are in `src/langbridge/train.py` and the runner notebook.

**Lesson.** Lower the micro-batch and raise accumulation to cut peak memory without shrinking the effective batch. The notes add that a larger effective batch may need slight learning-rate tuning.

**References.** Notion working notes, "ANOMALY BUGS" (Bug Log: NLLB QLoRA Fine-Tuning on 2× T4); `notebooks/02_training_runner_kaggle.ipynb`.

---

### LB-020: Data collator failed on leftover text columns in the LR range test

| Field | Value |
|---|---|
| Type | Bug |
| Category | Data |
| Severity | Medium |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded (before the LR range test on 2026-10-14) |
| Environment | Kaggle T4, `DataCollatorForSeq2Seq`, Hugging Face Datasets |

**Summary.** `DataCollatorForSeq2Seq` raised a `ValueError` while building tensors. The tokenized dataset still held the raw-text columns.

**Symptoms.** A `ValueError` when creating tensors from batched examples. The exact message was not recorded.

**How to reproduce.** Tokenize with `dataset.map(tokenize_function, batched=True)` without `remove_columns`, then batch with the collator.

**Impact.** The LR range test could not start until fixed.

**Root cause.** Tokenization adds `input_ids`, `attention_mask` and `labels`, but keeps the original `id` and `translation` columns. The nested `translation` dictionaries hold raw text that cannot become tensors.

**Fix.**

```python
igbo_dataset_train = igbo_dataset_train.map(
    tokenize_function,
    batched=True,
    remove_columns=igbo_dataset_train.column_names,
)
```

The same change was applied to the validation set.

**Verification.** The LR range test completed 501 iterations.

**Prevention.** The notebook removes the original columns during mapping. Expected columns: `input_ids`, `attention_mask`, `labels`.

**Lesson.** `map()` adds new columns but does not remove old ones. Drop unneeded columns before batching.

**References.** Notion working notes, "ANOMALY BUGS" (Bugs encountered during the cyclical learning rate process); `notebooks/01_lr_range_test_cyclic_lr.ipynb`.

---

### LB-021: LR finder passed the input dictionary to NLLB as one argument

| Field | Value |
|---|---|
| Type | Bug |
| Category | Training |
| Severity | Medium |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded (before the LR range test on 2026-10-14) |
| Environment | Kaggle T4, `torch-lr-finder`, NLLB-1.3B |

**Summary.** `torch-lr-finder` called the model with the whole batch dictionary as a single positional argument. NLLB treated the dictionary as `input_ids` and failed in its embedding layer.

**Symptoms.**

```
TypeError: embedding(): argument 'indices' must be Tensor, not dict
```

**How to reproduce.** Pass NLLB directly to `LRFinder`, with batches shaped as `{"input_ids", "attention_mask", "labels"}`.

**Impact.** The LR range test could not run until fixed.

**Root cause.** `LRFinder` calls `self.model(inputs)`. A dictionary passed this way is not unpacked into keyword arguments, so it reached the embedding layer as if it were the token tensor.

**Fix.** A small wrapper unpacks the dictionary. A criterion returns the loss NLLB already computes when labels are given:

```python
class NLLBModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs):
        return self.model(**inputs)

def nllb_criterion(outputs, labels):
    return outputs.loss
```

**Verification.** The LR range test completed 501 iterations (`results/lr_range_test/result.json`).

**Prevention.** The wrapper is part of `notebooks/01_lr_range_test_cyclic_lr.ipynb`.

**Lesson.** `model(inputs)` and `model(**inputs)` are different calls. When a tool and a model disagree on input format, fix the interface with a thin wrapper instead of changing either one.

**References.** Notion working notes, "ANOMALY BUGS" (Debugging torch-lr-finder with NLLB); `notebooks/01_lr_range_test_cyclic_lr.ipynb`.

---

### LB-022: Target padding tokens were counted in the training loss

| Field | Value |
|---|---|
| Type | Bug |
| Category | Data |
| Severity | High |
| Status | Resolved |
| Version | Igbo 1.0 |
| Date found | Not recorded |
| Environment | NLLB-200-distilled-1.3B, PEFT/QLoRA, Kaggle 2× T4 |

**Summary.** Target labels were padded during tokenization with the padding token (ID `1`). The loss only ignores `-100`, so the model was trained to predict padding after each sentence ended.

**Symptoms.**

- Training loss (about 57) was far above evaluation loss (about 13.25).
- Training loss dropped by 9–10 points at each epoch boundary (57.57 → 48.27 → 56.82 around epoch 1.0).
- The stored training labels held 1,042,087 padding IDs out of 1,442,513 positions, and 0 positions marked `-100`.

**How to reproduce.** Tokenize targets with `padding=True`, store the dataset, and train with `DataCollatorForSeq2Seq(label_pad_token_id=-100)`. Padding already in the data is not replaced.

**Impact.** Misleading loss values, and a training objective that included predicting padding.

**Root cause.** `padding=True` at tokenization stored padding ID `1` in the labels. The collator only marks padding it adds itself as `-100`.

**Fix.**

- Tokenize with `padding=False` and regenerate the dataset.
- Let the collator pad each batch, using `-100` for labels.
- Separately, set `trainer.model_accepts_loss_kwargs = False`, because the model's `forward()` does not use `num_items_in_batch`.

The notes say the observations do not separate the effect of each change.

**Verification.**

- Training loss fell to 1.825 / 1.791, against an evaluation loss of 1.981 at epoch 0.3337.
- The epoch-boundary change shrank to 0.092 (1.584 → 1.676).
- The training script now prints a padding audit before training.

**Prevention.** The `inspect_stored_labels` and `inspect_actual_batch` checks in `src/langbridge/train.py`.

**Lesson.** A large, regular gap between training and evaluation loss is a data bug until proven otherwise. Check what the loss is actually computed on.

**References.** Notion working notes, "ANOMALY BUGS" (Bug Report: Target Padding Tokens Were Included in NLLB Training Loss); `src/langbridge/train.py`.

---

### LB-023: Igbo 1.0 is worse than its base model on Igbo to English

| Field | Value |
|---|---|
| Type | Finding |
| Category | Evaluation |
| Severity | Medium |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | Not recorded (evaluation run around 2026-11-06) |
| Environment | Kaggle T4, base model in 4-bit, AfriCOMET (`masakhane/africomet-mtl`) |

**Summary.** On the v3 test split, LangBridge AI Igbo 1.0 scores 0.5741 on Igbo → English against the base model's 0.6440, a drop of −10.86%.

**Observation.**

| Direction | NLLB-1.3B | Igbo 1.0 | Change |
|---|---|---|---|
| English → Igbo | 0.6459 | 0.6580 | +1.88% |
| Igbo → English | 0.6440 | 0.5741 | −10.86% |

**Impact.** The adapter cannot serve both directions.

**Root cause.** Not confirmed. Every training example had English as the source and Igbo as the target, and the adapter changes weights shared by both directions. Hypothesis: one-directional fine-tuning shifted the model away from the reverse task.

**Decision.** Use the adapter for English → Igbo only. Route Igbo → English and all other directions to the base NLLB model.

**Verification.** Not applicable. Routing is planned, not deployed.

**Prevention.** Evaluate both directions for every adapter, and promote a model per direction only when it beats the current one.

**Lesson.** Fine-tuning one direction can hurt the reverse direction. Measure the directions you did not train, too.

**References.** [`results/igbo1_vs_nllb1.3b.csv`](../results/igbo1_vs_nllb1.3b.csv); [evaluation.md](evaluation.md#2-langbridge-ai-igbo-10-vs-nllb-200-distilled-13b); [deployment.md](deployment.md#routing-plan).

---

### LB-024: CTranslate2 conversion failed in the bundling notebook

| Field | Value |
|---|---|
| Type | Bug |
| Category | Deployment |
| Severity | Medium |
| Status | Open |
| Version | Igbo 1.0 |
| Date found | Not recorded (November 2026) |
| Environment | Kaggle CPU, torch 2.11.0+cpu, transformers 5.20.2, peft 0.21.2, ctranslate2 4.9.0 |

**Summary.** Converting the merged model to **CTranslate2** (a fast inference engine for CPUs) failed, because the converter was pointed at a folder name that no longer existed.

**Symptoms.**

```
OSError: merged-eng-ibo is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
```

**How to reproduce.** Run `notebooks/05_bundling_for_production.ipynb` top to bottom.

**Impact.** No int8 production bundle exists yet, which blocks the CPU deployment plan.

**Root cause.** The output folder was renamed to `merged_eng_igbo_v1`, but the converter ran with the old name. In the same notebook, the next conversion line passes an undefined variable, and the merge cell refers to `base_model_id` while the variable is named `c`.

**Fix.** Not done yet. Use one constant per path and re-run the whole notebook.

**Verification.** Not applicable.

**Prevention.** None yet. Restart the kernel and run all cells before trusting a notebook.

**Lesson.** Leftover notebook state hides broken cells. A clean top-to-bottom run is the only real test.

**References.** `notebooks/05_bundling_for_production.ipynb`; [deployment.md](deployment.md#what-the-bundling-notebook-does).

## Adding a new entry

Copy this template below the last entry, give it the next free ID, and add a matching row to the index. For a Finding, rename **Symptoms** to **Observation** and **Fix** to **Decision**, and leave out **How to reproduce** if it does not apply.

```markdown
---

### LB-00X: Short, specific title (what broke, not how it felt)

| Field | Value |
|---|---|
| Type | Bug / Finding |
| Category | Data / Training / Evaluation / Inference / Environment / Deployment |
| Severity | Critical / High / Medium / Low |
| Status | Resolved / Workaround / Open / Won't fix |
| Version | v1 / v2 / Igbo 1.0 |
| Date found | YYYY-MM-DD, or "Not recorded" |
| Environment | e.g. Kaggle 2× T4, transformers 4.x, peft 0.x, or "Not recorded" |

**Summary.** One or two sentences a visitor can read in five seconds.

**Symptoms.** What was seen. Put error messages in a code block, cut to the key lines.

**How to reproduce.** Short steps, or "Not recorded".

**Impact.** What it cost: hours, GPU time, failed runs, wrong numbers.

**Root cause.** Why it happened. If not confirmed, write "Not confirmed" and label the hypothesis.

**Fix.** What changed. Add a short code snippet or a before/after where it helps.

**Verification.** How we know the fix worked (a rerun, a metric, a test).

**Prevention.** What now stops it from coming back. If nothing, write "None yet".

**Lesson.** One or two sentences, general enough to help another engineer.

**References.** Source notes, notebooks, repo files and commits.
```
