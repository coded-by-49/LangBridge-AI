"""English -> Igbo inference with LangBridge AI Igbo 1.0 (internal: v3).

Loads the base model facebook/nllb-200-distilled-1.3B, attaches the
LangBridge AI Igbo 1.0 LoRA adapter from Hugging Face, and translates
English text into Igbo.

Designed for free, CPU-only hosting:
- No bitsandbytes and no 4-bit loading. The model loads in float32 on CPU.
- By default the adapter is merged into the base weights after loading
  (merge_and_unload), so generation runs on a plain NLLB model.
- Needs about 6 GB of free RAM and a one-time download of about 5.5 GB.

Use this adapter for English -> Igbo only. On the project's test split it
scored below the base model for Igbo -> English, so other directions should
be served by the base NLLB model (see docs/deployment.md).

Command line:
    python src/langbridge/inference.py "Good morning. How are you?"

Python:
    from langbridge.inference import load_model, translate
    tokenizer, model = load_model()
    print(translate(["Good morning."], tokenizer, model))
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
ADAPTER_ID = "coded-by-49/LANGBRIDGE_AI_IGBO_1.0"

SRC_LANG = "eng_Latn"  # English, Latin script (NLLB language code)
TGT_LANG = "ibo_Latn"  # Igbo, Latin script (NLLB language code)

# Same generation settings as notebooks/03_evaluation.ipynb.
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
NO_REPEAT_NGRAM_SIZE = 3
MAX_INPUT_TOKENS = 256  # training truncated source and target to 256 tokens


def load_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = ADAPTER_ID,
    merge: bool = True,
    device: str = "cpu",
):
    """Load the tokenizer and the base model with the Igbo 1.0 adapter attached.

    Args:
        base_model_id: Hugging Face id of the NLLB base model.
        adapter_id: Hugging Face id of the LoRA adapter.
        merge: If True, fold the adapter into the base weights. This makes
            generation a little faster and removes the PEFT wrapper.
        device: "cpu" (default) or a torch device string such as "cuda".

    Returns:
        (tokenizer, model) ready for translate().
    """
    # The adapter repo ships the NLLB tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    tokenizer.src_lang = SRC_LANG

    # Plain float32 load: works on CPU without bitsandbytes.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_id)
    if merge:
        model = model.merge_and_unload()

    model.to(device)
    model.eval()
    return tokenizer, model


def translate(
    texts: str | Iterable[str],
    tokenizer,
    model,
    num_beams: int = NUM_BEAMS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
) -> list[str]:
    """Translate English text into Igbo.

    Args:
        texts: One English string or a list of English strings.
        tokenizer, model: Returned by load_model().

    Returns:
        A list of Igbo translations, one per input.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    # NLLB needs the source language set before tokenizing.
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)

    # Force the first generated token to be the Igbo language tag.
    # (tokenizer.lang_code_to_id was removed in newer transformers versions.)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate English text into Igbo with LangBridge AI Igbo 1.0."
    )
    parser.add_argument("text", nargs="+", help="English sentence(s) to translate.")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep the adapter separate instead of merging it into the base model.",
    )
    parser.add_argument("--device", default="cpu", help='Torch device, default "cpu".')
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    tokenizer, model = load_model(merge=not args.no_merge, device=args.device)
    loaded = time.perf_counter()

    outputs = translate(args.text, tokenizer, model, num_beams=args.num_beams)
    done = time.perf_counter()

    for source, target in zip(args.text, outputs):
        print(f"EN: {source}")
        print(f"IG: {target}")
    print(
        f"[load {loaded - start:.1f}s | translate {done - loaded:.1f}s | device {args.device}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
