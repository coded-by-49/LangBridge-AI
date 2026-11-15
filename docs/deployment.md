# Deployment

## Current status: in progress, not live

**LangBridge AI Igbo 1.0** (internal: v3) is **not deployed**. There is no public API or demo URL yet. The adapter is public on Hugging Face ([coded-by-49/LANGBRIDGE_AI_IGBO_1.0](https://huggingface.co/coded-by-49/LANGBRIDGE_AI_IGBO_1.0)) and can be run locally with [`src/langbridge/inference.py`](../src/langbridge/inference.py).

## Target: free, CPU-only hosting

The plan is to serve the model from a free hosting tier with no GPU. That sets the design:

- **No bitsandbytes.** 4-bit loading needs a CUDA GPU, so the CPU path loads the base model in float32 and merges the adapter into it.
- **Memory.** The 1.3B model needs about 5.5 GB of RAM in float32. CTranslate2 int8 (below) is the planned way to shrink this for small hosts. The int8 size is TBD until the conversion runs.
- **Speed.** Beam search on CPU is slow. Expect seconds per sentence. Measured numbers are TBD.

## What the bundling notebook does

[`notebooks/05_bundling_for_production.ipynb`](../notebooks/05_bundling_for_production.ipynb) runs on a Kaggle CPU runtime (torch 2.11.0+cpu, transformers 5.20.2, peft 0.21.2, ctranslate2 4.9.0). It:

1. **Loads** `facebook/nllb-200-distilled-1.3B` in float32 and the tokenizer from the adapter repo.
2. **Attaches** the `coded-by-49/LANGBRIDGE_AI_IGBO_1.0` adapter (`revision="main"`).
3. **Merges** the adapter into the base weights with `merge_and_unload()`. The result is a plain NLLB model with the LoRA changes folded in, so serving needs no PEFT.
4. **Saves** the merged model and tokenizer to `merged_eng_igbo_v1/`.
5. **Converts** to **CTranslate2** with int8 quantization. CTranslate2 is a fast inference engine for transformer models on CPU, and int8 means 8-bit integer weights, about 4× smaller than float32. It writes `ct2_eng_ibo_int8/` for the Igbo model and `ct2_nllb_base_1.3b_int8/` for the base model used by the other directions.
6. Imports `HfApi`, which suggests the bundles will be uploaded to Hugging Face. No upload cell exists yet.

**Status of the last run:** steps 1–4 ran. Step 5 failed because the converter was given an old folder name (`merged-eng-ibo`). The second conversion line passes an undefined variable. The merge cell also refers to `base_model_id` while the variable is named `c`. See [lessons_learned.md](lessons_learned.md#11-ctranslate2-conversion-failed-in-the-bundling-notebook).

## Routing plan

The adapter improved its trained direction (English → Igbo) on the project's test split, but it scored lower than the base model on Igbo → English. So each request is routed by direction:

| Direction | Model |
|---|---|
| English → Igbo | LangBridge AI Igbo 1.0 |
| Igbo → English | base NLLB-1.3B |
| All other directions | base NLLB-1.3B |

Two copies of a 1.3B model use a lot of memory. One option is to keep a single base model and attach or detach the LoRA adapter per request, using PEFT's `disable_adapter()` context. Another is two CTranslate2 int8 models. Which fits the free tier's memory is TBD.

Before promotion: the English → Igbo gain was measured on a test split that overlaps the training split by 88.95% (see [evaluation.md](evaluation.md#issue-4-test-split-overlaps-the-training-split)). Re-score on clean data before routing real traffic to the adapter. If the gain does not hold, route English → Igbo to the base model too.

## Checklist

**Done**

- [x] Adapter trained and published on Hugging Face (2026-11-03)
- [x] Hugging Face model card written (updated 2026-11-12)
- [x] Merge code written (notebook 05, steps 1–4)
- [x] CPU-only inference script without bitsandbytes ([`src/langbridge/inference.py`](../src/langbridge/inference.py)). See the README for whether it was tested.
- [x] Routing decided per direction

**Left**

- [ ] Re-score English → Igbo on a test set with no training overlap (promotion gate)
- [ ] Fix the CTranslate2 conversion in notebook 05 and record the int8 model sizes
- [ ] Check that the int8 model's output matches the float32 model on a sample
- [ ] Upload the merged and/or CTranslate2 bundles to Hugging Face
- [ ] Choose the free CPU host and check its memory limit
- [ ] Build a small API (for example `POST /translate {text, src, tgt}`) with the routing table above
- [ ] Measure latency and memory on the host
- [ ] Add input length limits (256 tokens) and sentence splitting for long text
- [ ] Add basic logging and a health check
- [ ] Publish the URL and update the README status badge
