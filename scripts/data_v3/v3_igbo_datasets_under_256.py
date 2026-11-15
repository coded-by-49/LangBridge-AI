import os
from huggingface_hub import login
from pathlib import Path
import json
from transformers import (
    NllbTokenizer
)


secret = os.getenv('HF_SECRET')

login(token=secret)

print("login successful") if login else print("login failed")


model_id = "facebook/nllb-200-distilled-1.3B"

tokenizer = NllbTokenizer.from_pretrained(
    model_id,
    src_lang="eng_Latn",
    tgt_lang="ibo_Latn",
)


# Repository root (this file lives in scripts/data_v3/). Was a hard-coded
# absolute Windows path before the repo restructure.
root = Path(__file__).resolve().parents[2]

input_file_path = (
    root
    / "langbridge_ai_v3_data/data/igbo_v3/val/igbo_eng_lv1_validation.jsonl"
)

output_file_path = (
    root
    / "langbridge_ai_v3_data/data/igbo_v3/val/igbo_eng_lv1_validation_256_filtered.jsonl"
)

removed = 0
kept = 0

with (
    input_file_path.open("r", encoding="utf-8") as input_file,
    output_file_path.open("w", encoding="utf-8") as output_file
):
    for line in input_file:
        entry = json.loads(line)

        eng_text = entry["translation"]["eng_Latn"]
        igbo_text = entry["translation"]["ibo_Latn"]

        eng_tokens = tokenizer.tokenize(eng_text)
        igbo_tokens = tokenizer.tokenize(igbo_text)

        if len(eng_tokens) > 256 or len(igbo_tokens) > 256:
            removed += 1
            continue

        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        kept += 1

print(f"file length : {kept+removed}")
print(f"Kept: {kept}")
print(f"Removed: {removed}")
