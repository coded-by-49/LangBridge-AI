---
language:
  - en
  - ig
license: cc-by-nc-4.0
library_name: peft
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-1.3B
base_model_relation: adapter
datasets:
  - coded-by-49/langbridge_ai_v3_igbo_datasets
tags:
  - english-igbo
  - igbo
  - nllb
  - low-resource
  - african-languages
  - machine-translation
  - peft
  - lora
  - qlora
---

# LangBridge AI Igbo 1.0

LangBridge AI Igbo 1.0 (internal: v3) is a LoRA adapter for Meta AI's [NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B). It was fine-tuned for **English → Igbo** (`eng_Latn` → `ibo_Latn`) translation.

This repository holds the adapter only. To use it, load the NLLB base model and attach the adapter. The tokenizer files are included.

> This file is kept in sync with the model card at [huggingface.co/coded-by-49/LANGBRIDGE_AI_IGBO_1.0](https://huggingface.co/coded-by-49/LANGBRIDGE_AI_IGBO_1.0). Source repository: [github.com/coded-by-49/LangBridge-AI](https://github.com/coded-by-49/LangBridge-AI).

## Model details

| Field | Value |
|---|---|
| Model | LangBridge AI Igbo 1.0 (internal: v3) |
| Developer | Fabian Bernard ([coded-by-49](https://huggingface.co/coded-by-49)) |
| Task | Machine translation |
| Trained direction | English (`eng_Latn`) → Igbo (`ibo_Latn`) only |
| Base model | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B) |
| Method | 4-bit NF4 QLoRA with rank-stabilized LoRA (rsLoRA) |
| Adapter | r=32, alpha=32, dropout 0.05, bias `lora_only`, on `q_proj, k_proj, v_proj, out_proj, fc1, fc2` |
| Adapter file size | 188.9 MB (`adapter_model.safetensors`) |
| Released checkpoint | Epoch 1.333666 (global step 668), lowest validation loss |
| Max sequence length in training | 256 tokens (source and target) |
| Release date | 2026-11-03 |
| Earlier version | LangBridge v1 (`coded-by-49/Lang_bridge_AI`, NLLB-600M, multilingual), now archived |

## Intended use

- **English → Igbo translation only:** research, experiments and draft translations of short English sentences.
- Teaching and further fine-tuning research on low-resource African languages.

Have a fluent Igbo speaker review any translation before it is used for anything important.

## Out-of-scope use

- **Igbo → English.** The adapter scores *lower* than the base model in this direction (−10.86% AfriCOMET). Use the base NLLB model instead.
- **Any other language pair.** The adapter was not trained or evaluated for other pairs. Use the base NLLB model.
- **Commercial use.** The base model is licensed CC-BY-NC 4.0 (non-commercial), and so is this adapter.
- **High-stakes use without human review:** medical, legal, financial, safety-critical, official or certified translation.

## Training data

[`coded-by-49/langbridge_ai_v3_igbo_datasets`](https://huggingface.co/datasets/coded-by-49/langbridge_ai_v3_igbo_datasets), built from three human-translated sources:

1. **FLORES-101/200**: professional translations of English Wikipedia sentences (CC-BY-SA 4.0).
2. **Igbo–English MT benchmark** (Ezeani et al., 2020): news-domain sentences.
3. **MAFAND-MT** (Adelani et al., 2022): African news translation corpus (CC-BY-NC 4.0).

Split sizes (line counts of the published files):

| Split | File | Pairs | Used for |
|---|---|---|---|
| Train | `igbo_eng_lv1_train.jsonl` | 12,009 | Fine-tuning |
| Validation | `igbo_eng_lv1_validation_256_filtered.jsonl` | 589 (597 before removing pairs over 256 tokens) | Checkpoint selection |
| Test | `igbo_eng_lv1_test_.jsonl` | 1,693 | Final evaluation |

Cleaning: line alignment into pairs, de-duplication across sources by the English sentence, and a 256-token length filter on validation.

**Overlap warning:** 1,506 of the 1,693 test pairs (88.95%) also appear in the training split. MAFAND-MT dev was used in both. The validation split has no overlap with training.

## Training procedure

- **Quantization:** base model in 4-bit NF4 with double quantization, float32 compute, prepared with `prepare_model_for_kbit_training`. `tie_word_embeddings=False`.
- **Hardware:** 2 × NVIDIA T4 (Kaggle), `accelerate launch --num_processes 2` (data parallel).
- **Batch:** 3 per GPU × 4 gradient-accumulation steps × 2 GPUs = 24.
- **Optimizer:** AdamW, weight decay 0.01.
- **Learning rate:** PyTorch `CyclicLR`, triangular, from 2e-5 to 3.3e-4, `step_size_up=1002` (≈ 2 epochs), `cycle_momentum=False`. The range came from an LR range test: the loss started to fall near 1e-5 and was lowest at 5.14e-4.
- **Regularisation:** label smoothing 0.1, gradient clipping 1.0, LoRA dropout 0.05.
- **Precision and memory:** fp16 mixed precision, gradient checkpointing.
- **Schedule:** up to 9.4 epochs (4,710 steps). Evaluation and saving every 167 steps. Best checkpoint by validation loss (`load_best_model_at_end`). Early stopping with a patience of 12 evaluations.
- **Released checkpoint:** epoch 1.333666 = step 668, the 4th evaluation point. Training logs were not kept, so whether the run ended by early stopping or another stop is not confirmed.

## Evaluation

**Metric:** AfriCOMET (`masakhane/africomet-mtl`, via `unbabel-comet` 2.2.7). It is a learned metric for African languages; higher is better, and the score is not a percentage.

**Test data:** the test split above, 1,693 pairs. Both models were run on the same sentences.

**Decoding:** beam size 4, `max_new_tokens=256`, `no_repeat_ngram_size=3`, `forced_bos_token_id` = target language code. Base model loaded in 4-bit on a T4 for both systems.

| Direction | NLLB-1.3B baseline | LangBridge AI Igbo 1.0 | Absolute change | Relative change |
|---|---|---|---|---|
| English → Igbo | 0.6459 | 0.6580 | +0.0122 | +1.88% |
| Igbo → English | 0.6440 | 0.5741 | −0.0699 | −10.86% |

Exact values: 0.645850 → 0.658012 (+0.012162) and 0.643982 → 0.574051 (−0.069931).

How to read this:

- The adapter scores a little higher than the base model in its trained direction. Because 88.95% of the test pairs were also training pairs, this is **not yet evidence of better translation of new text**.
- The adapter scores clearly lower in the reverse direction, which it was not trained on.
- No significance test, BLEU, chrF++ or human evaluation has been done.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
ADAPTER_ID = "coded-by-49/LANGBRIDGE_AI_IGBO_1.0"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
tokenizer.src_lang = "eng_Latn"

base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)  # float32, works on CPU
model = PeftModel.from_pretrained(base_model, ADAPTER_ID).merge_and_unload().eval()

inputs = tokenizer(["Good morning. How are you?"], return_tensors="pt", truncation=True, max_length=256)
with torch.inference_mode():
    out = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("ibo_Latn"),
        num_beams=4,
        max_new_tokens=256,
        no_repeat_ngram_size=3,
    )
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

Do not use `tokenizer.lang_code_to_id`; newer transformers versions removed it.

## Limitations and bias

- **One direction:** only English → Igbo was trained. Igbo → English is worse than the base model.
- **Evaluation:** the test split overlaps training (88.95%); the gain is small (+1.88% relative); there is no significance test.
- **Domain:** news and encyclopedic text only. Quality may be lower for conversation, slang, social media, specialist terms and dialects.
- **Orthography:** Igbo spelling, diacritics and tone marks vary across sources, and the model may copy these inconsistencies.
- **Length:** training truncated inputs to 256 tokens. Split long text into sentences.
- **Content:** names, numbers, negation and idioms may be mistranslated. Biases in the news sources and in NLLB may carry over.
- **Precision:** scores were measured with the base model in 4-bit. CPU deployment in float32 or int8 was not separately evaluated.

## License

**CC-BY-NC 4.0.** This follows the non-commercial license of the base model, [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B). One training source (MAFAND-MT) is also CC-BY-NC 4.0. The dataset repository is labelled MIT, but that label does not override the terms of its sources. This is not legal advice.

## Citation

```bibtex
@misc{bernard2026langbridge,
  author       = {Bernard, Fabian},
  title        = {LangBridge AI Igbo 1.0: an English-Igbo QLoRA adapter for NLLB-200},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/coded-by-49/LANGBRIDGE_AI_IGBO_1.0}}
}
```

Please also cite NLLB (NLLB Team, 2022, [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)), AfriCOMET ([Wang et al., 2024](https://aclanthology.org/2024.naacl-long.334/)), FLORES-101 ([Goyal et al., 2022](https://aclanthology.org/2022.tacl-1.30/)), Ezeani et al. ([2020](https://arxiv.org/abs/2004.00648)) and MAFAND-MT ([Adelani et al., 2022](https://aclanthology.org/2022.naacl-main.223/)).
