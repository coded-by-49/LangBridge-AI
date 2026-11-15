# Changelog

Dates come from git commits, Hugging Face commits and notebook outputs. See [docs/experiments.md](docs/experiments.md) for details.

## [Unreleased]: repository overhaul (2026-11-15)

- Reorganised the repository: `src/`, `notebooks/`, `scripts/`, `configs/`, `results/`, `docs/`, `archive/`.
- Added a CPU-only inference script, results CSVs and charts, a split-overlap check, and detailed documentation.
- Found that 1,506 of 1,693 test pairs (88.95%) also appear in the training split. Documented in [docs/evaluation.md](docs/evaluation.md).
- Removed a hard-coded Hugging Face token from a notebook. Secrets, virtual environments, IDE files, weights and local data are git-ignored.

## LangBridge AI Igbo 1.0 (internal: v3): 2026-11-03

- English → Igbo LoRA adapter on `facebook/nllb-200-distilled-1.3B`. 4-bit NF4 QLoRA with rsLoRA, r=32, alpha=32, all linear layers.
- Trained on `coded-by-49/langbridge_ai_v3_igbo_datasets` (12,009 pairs from FLORES, MAFAND-MT and Ezeani et al.).
- 2 × T4 with `accelerate`; AdamW + triangular CyclicLR (2e-5 → 3.3e-4), chosen with an LR range test (2026-10-14).
- Released checkpoint: epoch 1.333666 (step 668), best validation loss.
- AfriCOMET vs NLLB-1.3B: English → Igbo 0.6580 vs 0.6459 (+1.88%); Igbo → English 0.5741 vs 0.6440 (−10.86%).
- Model card updated 2026-11-12. Deployment in progress.

## LangBridge v2 (intermediate): date TBD

- Partly recorded. `coded-by-49/LANG_BRIDGE_AI_2.0` (private) is a LoRA adapter on NLLB-200-distilled-600M (r=8, alpha=16, attention layers).
- The project notes describe it as trained on FLORES-200, MAFAND-MT and the Ezeani corpus. No training code or scores are in the repository.

## LangBridge v1: 2025

- 2025-07-19: repository created.
- 2025-07 to 2025-09: data collection and cleaning for Hausa, Igbo and Yoruba ↔ English (JW300, OPUS, FLORES-200, Tatoeba, Hypa_Fleurs, MAFAND-MT, web scraping). 3,000,000 tokenized training examples.
- 2025-09-15: final data formatting and training setup.
- Training date not recorded (after 2025-09-15; validation data packaged 2025-11-30): LoRA adapter (`coded-by-49/Lang_bridge_AI`) trained on NLLB-200-distilled-600M.
- 2026-06-25: AfriCOMET evaluation published. v1 scored below its base model in all six directions.
- The v1 dataset has since been deleted. Code is archived in [archive/v1/](archive/v1).
