# Experiment log

One entry per run found in the code, notebooks, notes, git history and Hugging Face. Dates come from git commits, notebook outputs (`nvidia-smi` timestamps), file dates or Hugging Face commits. "TBD" means not recorded.

Names: **LangBridge v1** (archived, multilingual) and **LangBridge AI Igbo 1.0** (internal: v3, current).

---

## E1: LangBridge v1 training

| Field | Value |
|---|---|
| Date | Data work 2025-07-19 → 2025-09-15 (git). Validation data zipped 2025-11-30 (Kaggle path `igbo_test_val-20251130T173420Z`). Training run date TBD |
| Version | LangBridge v1 |
| Base model | `facebook/nllb-200-distilled-600M` |
| Data | LangBridge Wazobia (Hausa, Igbo, Yoruba ↔ English). 3,000,000 tokenized training examples in the saved pipeline (about 4M pairs per the old README). Validation: 93,319 Igbo rows. **Dataset since deleted** |
| Key settings (saved notebook) | 4-bit NF4, double quant, compute dtype bfloat16. LoRA r=32, alpha=64, dropout 0.05, targets `q_proj, k_proj, v_proj, out_proj`, bias none. Batch 16 × accumulation 2, lr 2e-4, 1 epoch, eval/save every 500 steps with generation (max 60 tokens), fp16, single T4. Hub id `coded-by-49/Lang_bridge_AI` |
| Results | See E2 |
| Outcome | Adapter published (private now). Code: [`archive/v1/training/`](../archive/v1/training/lang_bridge_training_process.ipynb) |

Notes:

- The old README says the LoRA targets also included `fc1` and `fc2`. The saved notebook does not include them.
- The saved notebook's evaluation callback has a typo (`device = model.devic`). It would raise an error at the first evaluation (step 500), and the `try/except` around `trainer.train()` would then save a "crash salvage" adapter. If the published adapter came from that run, v1 trained on at most 500 × 32 = 16,000 examples. **Not confirmed** (no outputs were saved).

## E2: LangBridge v1 evaluation vs NLLB-600M

| Field | Value |
|---|---|
| Date | 2026-06-25 (git commit "completetion of langbridge_ai_inference"); AfriCOMET runs around 2026-06-26 |
| Version | LangBridge v1 |
| Base model | NLLB-200-distilled-600M (as baseline) |
| Data | 2,000 random pairs per direction (seed 42) from the v1 `*_test_val.jsonl` files; scored in slices 0–100, 100–600, 600–1600 |
| Key settings | AfriCOMET `masakhane/africomet-mtl`. Beam 4, max 256 new tokens, `no_repeat_ngram_size=3`, `forced_bos_token_id` set, `src_lang` not set |
| Results (Sentences 600–1600) | Igbo → English 0.3764 vs 0.4029 (−6.57%); Hausa → English 0.2804 vs 0.5205 (−46.13%); Yoruba → English 0.2285 vs TBD; English → Igbo 0.5637 vs 0.5985 (−5.80%); English → Hausa 0.6538 vs 0.6697 (−2.38%); English → Yoruba 0.4509 vs 0.4940 (−8.73%). Full table: [evaluation.md](evaluation.md#3-langbridge-v1-vs-nllb-200-distilled-600m-archived) |
| Outcome | **v1 scored below its baseline in all six directions** (five with a known baseline; the Yoruba → English baseline is TBD). This led to the pivot |

What I learned:

1. More data is not better when it is noisy and off-domain.
2. Six directions at once made it impossible to tell which part failed.
3. Language tags (`src_lang`, `forced_bos_token_id`) must be checked in every step: training, validation and evaluation.
4. Note that the saved v1 inference notebook loads `coded-by-49/LANG_BRIDGE_AI_2.0`, not `Lang_bridge_AI`. See E3.

## E3: LangBridge v2 (intermediate, partly recorded)

| Field | Value |
|---|---|
| Date | TBD (Hugging Face cache shows PEFT 0.18.0 was used) |
| Version | Probably "v2" **(guess)** |
| Base model | `facebook/nllb-200-distilled-600M` |
| Data | TBD. The project brief says FLORES-200, MAFAND-MT and the Ezeani corpus |
| Key settings | From the cached `adapter_config.json` of `coded-by-49/LANG_BRIDGE_AI_2.0`: LoRA r=8, alpha=16, dropout 0.05, targets `q_proj, k_proj, v_proj, out_proj`, bias none |
| Results | TBD (no saved scores) |
| Outcome | No training code for this run is in the repository. The adapter is private. It is referenced by the v1 inference notebook (loaded as a full model) and by `notebooks/03_evaluation.ipynb` (loaded as a second adapter on the 1.3B base, which does not match its 600M base) |

## E4: LangBridge v1 "overfitting evaluation"

| Field | Value |
|---|---|
| Date | Around 2026-06-26 (notebook 04 `nvidia-smi` timestamp) |
| Version | LangBridge v1 |
| Base model | NLLB-200-distilled-600M + v1 adapter (assumed) |
| Data | Files `l1_{lang}_eng_comet_overfit_eval_data_0_1000.json` (1,000 sentences each) from the private Kaggle dataset `henryaloh/l1-yor-eng-comet-overfit-eval-data` |
| Key settings | AfriCOMET `masakhane/africomet-mtl`, batch 8 |
| Results | Hausa → English 0.2623; Yoruba → English 0.2365 (matches the saved output of notebook 04); Igbo → English 0.3625. Source: [`results/v1_overfitting_eval.csv`](../results/v1_overfitting_eval.csv) |
| Outcome | **Purpose not confirmed.** The file names suggest a check for overfitting (perhaps scoring on training sentences), but no code shows how the files were built. The scores are close to the v1 test-slice scores for the same directions (0.2804, 0.2285, 0.3764) |

## E5: v3 dataset build

| Field | Value |
|---|---|
| Date | 2026-09-22 (dataset created); split files renamed 2026-09-30; filtered validation uploaded 2026-10-05 (Hugging Face commits) |
| Version | Igbo 1.0 data |
| Data | FLORES-101/200, Ezeani et al., MAFAND-MT → train 12,009 / validation 597 (589 filtered) / test 1,693 |
| Key settings | English-side de-duplication across sources; 256-token validation filter |
| Outcome | Published as `coded-by-49/langbridge_ai_v3_igbo_datasets`. A later audit (2026-11-15) found that 1,506 of 1,693 test rows also appear in train. See [data.md](data.md#splits) |

## E6: LR range test (Igbo 1.0)

| Field | Value |
|---|---|
| Date | 2026-10-14 (result files) |
| Version | Igbo 1.0 |
| Base model | NLLB-200-distilled-1.3B, 4-bit NF4 + rsLoRA r=32 (same as training) |
| Data | v3 train split; v3 filtered validation |
| Key settings | `torch-lr-finder`, 501 iterations, lr 1e-7 → 5 (exponential), batch 6 × 4 accumulation = 24, one T4, AdamW (weight decay 0.01) |
| Results | Lowest loss 1.9904 at lr 5.14e-4. Loss 1.9937 near lr 3.36e-4. Loss NaN from lr ≈ 0.237. Tool suggestion 0.2287 (unusable). Runtime 9 h 45 min. Source: [`results/lr_range_test/result.json`](../results/lr_range_test/result.json) |
| Outcome | Chose CyclicLR `base_lr=2e-5`, `max_lr=3.3e-4`. Lesson: read the curve; do not trust the automatic suggestion |

## E7: LangBridge AI Igbo 1.0 training ("first main run")

| Field | Value |
|---|---|
| Date | Before 2026-11-03 (adapter upload date) |
| Version | LangBridge AI Igbo 1.0 (internal: v3) |
| Base model | `facebook/nllb-200-distilled-1.3B` |
| Data | v3 train (12,009), English → Igbo only; validation 589 |
| Key settings | 4-bit NF4 QLoRA, rsLoRA r=32 alpha=32 dropout 0.05 all-linear. 2 × T4, batch 3 × 4 accumulation × 2 GPUs = 24. AdamW + CyclicLR (2e-5 → 3.3e-4, step_size_up 1002). Label smoothing 0.1. Eval/save every 167 steps, best by `eval_loss`, early stopping patience 12, up to 9.4 epochs. Details: [training.md](training.md) |
| Results | Released checkpoint: epoch 1.333666 (= step 668, derived). Validation losses: TBD (logs not saved) |
| Outcome | Adapter published as `coded-by-49/LANGBRIDGE_AI_IGBO_1.0` (2026-11-03) |

## E8: LangBridge AI Igbo 1.0 evaluation vs NLLB-1.3B

| Field | Value |
|---|---|
| Date | Around 2026-11-06 (notebook 03 `nvidia-smi` timestamp); model card updated 2026-11-12 |
| Version | LangBridge AI Igbo 1.0 |
| Base model | NLLB-200-distilled-1.3B (baseline), 4-bit at evaluation time |
| Data | v3 test split, 1,693 pairs |
| Key settings | Beam 4, max 256 new tokens, `no_repeat_ngram_size=3`, `forced_bos_token_id` from `convert_tokens_to_ids`; AfriCOMET `masakhane/africomet-mtl` |
| Results | English → Igbo 0.6580 vs 0.6459 (+0.0122, +1.88%). Igbo → English 0.5741 vs 0.6440 (−0.0699, −10.86%). Source: [`results/igbo1_vs_nllb1.3b.csv`](../results/igbo1_vs_nllb1.3b.csv) |
| Outcome | Small gain in the trained direction and a drop in the reverse direction → per-direction routing. **Caveat found later:** 88.95% of test rows are also in train, so the gain is not a clean held-out result. No significance test |

## E9: Second training attempt (failed)

| Field | Value |
|---|---|
| Date | 2026-11-05 (notebook 02 output) |
| Version | Igbo 1.0 setup |
| Key settings | Same script; transformers 5.16.1, accelerate 1.15.0 |
| Results | Crashed at step 0/4710: `ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time` |
| Outcome | Cause not recorded. See [lessons_learned.md](lessons_learned.md#6-failed-ddp-run-decoder_input_ids-and-decoder_inputs_embeds) |

## E10: Production bundling (in progress)

| Field | Value |
|---|---|
| Date | November 2026 (notebook 05) |
| Version | Igbo 1.0 |
| Key settings | CPU runtime (torch 2.11.0+cpu, transformers 5.20.2, peft 0.21.2, ctranslate2 4.9.0). Merge adapter into base → `merged_eng_igbo_v1`. Convert to CTranslate2 int8 (Igbo model and base model) |
| Results | Merge cell ran. CTranslate2 conversion failed (wrong model path). See [deployment.md](deployment.md) |
| Outcome | Not deployed yet |
