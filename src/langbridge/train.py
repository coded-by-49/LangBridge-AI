"""LangBridge AI Igbo 1.0 (internal: v3) training script.

Fine-tunes facebook/nllb-200-distilled-1.3B for English -> Igbo with 4-bit
NF4 QLoRA (rsLoRA, r=32, alpha=32, all linear layers).

Origin: this is the code cell of the Kaggle script notebook
`bernardchiagoziem/lb_train_script_main`, copied here unchanged. The raw
Kaggle export (a notebook JSON saved with a .py extension) is kept at
archive/v3_kaggle_export/lb_train_script_main.ipynb.

How it is run: notebooks/02_training_runner_kaggle.ipynb launches it on two
Kaggle T4 GPUs with `accelerate launch --num_processes 2`. That runner points
at the Kaggle path of the script, not at this file.

Requirements: Kaggle GPU runtime (uses kaggle_secrets for the HF token),
bitsandbytes, peft, transformers, accelerate, datasets.

Hyperparameters are mirrored in configs/igbo_1.0_qlora.yaml for reference.
This script does not read that file. Do not change values here without
re-running training and evaluation.
"""
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

import os
import subprocess
import torch

import kagglehub
import transformers
import inspect 

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR



print(f"transformers version : {transformers.__version__}")

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    logging,
    EarlyStoppingCallback,
    TrainerCallback,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from datasets import load_dataset


import json

class TrainingMetricsJSONLCallback(TrainerCallback):
    def __init__(self, train_log_path, eval_log_path):
        self.train_log_path = train_log_path
        self.eval_log_path = eval_log_path

    def _write_record(self, path, record):
        with open(path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        logs = logs or {}

        if any(key.startswith("eval_") for key in logs):
            path = self.eval_log_path
            event = "evaluation"
        elif "loss" in logs:
            path = self.train_log_path
            event = "training"
        else:
            return control

        self._write_record(path, {
            "event": event,
            "global_step": int(state.global_step),
            "epoch": state.epoch,
            **logs,
        })

        return control


token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=token)


print(f"CUDA AVAILABLE : {torch.cuda.is_available()}")

try:
    subprocess.run(["nvidia-smi", "-L"], check=False)
except FileNotFoundError:
    print("nvidia-smi not found. No NVIDIA GPU detected.")

model_id = "facebook/nllb-200-distilled-1.3B"

model_config = AutoConfig.from_pretrained(model_id)
model_config.tie_word_embeddings = False

tokenizer = NllbTokenizer.from_pretrained(
    model_id,
    src_lang="eng_Latn",
    tgt_lang="ibo_Latn",
    truncation_side="right",
    model_max_length=256,
    # truncation=True,
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32, 
)


nf4_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    config=model_config,
    quantization_config=bnb_config,
    torch_dtype = torch.float32,
)

param_properties = {
    "infinite_gradients": [],
}

for name, param in nf4_model.named_parameters():
    dt = str(param.dtype)
    param_properties[dt] = param_properties.get(dt, 0) + 1
    
    if param.grad is not None:
        if not torch.isfinite(param.grad).all():
            param_properties["infinite_gradients"].append(name)
    

print("Model weight properties : ")
print(param_properties)


lora_config = LoraConfig(
    target_modules="all-linear",
    use_rslora=True,
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="lora_only",
    task_type="SEQ_2_SEQ_LM",
)


nf4_model = prepare_model_for_kbit_training(nf4_model)

qlora_model = get_peft_model(nf4_model, lora_config)



igbo_dataset_train = load_dataset(
    "coded-by-49/langbridge_ai_v3_igbo_datasets",
    split="train",
)

igbo_dataset_validation = load_dataset(
    "coded-by-49/langbridge_ai_v3_igbo_datasets",
     data_files={
        "validation": "hf://datasets/coded-by-49/langbridge_ai_v3_igbo_datasets/igbo_eng_lv1_validation_256_filtered.jsonl"
    },
    split="validation",
)


def tokenize_function(batch):
    eng_inputs = [
        entry["eng_Latn"]
        for entry in batch["translation"]
    ]

    ibo_targets = [
        entry["ibo_Latn"]
        for entry in batch["translation"]
    ]

    model_inputs = tokenizer(
        text=eng_inputs,
        max_length=256,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=ibo_targets,
        max_length=256,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


igbo_dataset_train = igbo_dataset_train.map(
    tokenize_function,
    batched=True,
)

igbo_dataset_validation = igbo_dataset_validation.map(
    tokenize_function,
    batched=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=qlora_model, #for label smoothing use case
    # model = model_id,
    pad_to_multiple_of=8,
)


print("\n")
print(
    f"RANK={os.environ.get('RANK')} | "
    f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} | "
    f"CUDA_CURRENT={torch.cuda.current_device()} | "
    f"MODEL_DEVICE = {next(nf4_model.parameters()).device} |",
    # f"UNUSED PARAMETERS={ddp_kwargs.find_unused_parameters}",
    flush=True,
)



try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    pass



training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/langbridge_ai_checkpoints",
    eval_strategy="steps",
    eval_steps = 167,
    save_strategy="steps",
    save_steps=167,
    save_total_limit=None,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better = False,
    label_smoothing_factor = 0.1,

    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    fp16=True,
    bf16=False,
    report_to="none",
    max_grad_norm=1.0,
    logging_steps=3,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    num_train_epochs=9.4,
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, qlora_model.parameters()),
    lr=2e-5,               
    weight_decay=0.01,
)

scheduler = CyclicLR(
    optimizer,
    base_lr=2e-5,
    max_lr=3.3e-4,
    step_size_up=1002,
    mode="triangular",
    cycle_momentum=False, # i will go for simple right now 
)

trainer = Seq2SeqTrainer(
    model=qlora_model,
    args=training_args,
    train_dataset=igbo_dataset_train,
    eval_dataset=igbo_dataset_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers = (optimizer, scheduler),
    callbacks=[
    TrainingMetricsJSONLCallback(
        train_log_path="/kaggle/working/training_metrics.jsonl",
        eval_log_path="/kaggle/working/evaluation_metrics.jsonl",
    ),
    EarlyStoppingCallback(early_stopping_patience=12),
],
)
trainer.model_accepts_loss_kwargs = False


PAD_ID = tokenizer.pad_token_id
EOS_ID = tokenizer.eos_token_id
IGNORE_ID = -100

print("===== PADDING CONFIGURATION =====")
print("PAD_ID:", PAD_ID)
print("EOS_ID:", EOS_ID)
print("IGNORE_ID:", IGNORE_ID)
print("Collator:", type(trainer.data_collator).__name__)
print(
    "Collator label_pad_token_id:",
    getattr(trainer.data_collator, "label_pad_token_id", "Not exposed"),
)


# 1. Inspect labels stored in the datasets before batching.
def inspect_stored_labels(dataset, split_name, max_examples=5):
    print(f"\n===== {split_name}: STORED LABELS =====")

    if dataset is None or "labels" not in dataset.column_names:
        print("Dataset or labels column unavailable.")
        return

    total_positions = 0
    total_pad_ids = 0
    total_ignored = 0
    rows_with_pad = 0
    examples_shown = 0

    for index in range(len(dataset)):
        labels = [int(x) for x in dataset[index]["labels"]]

        total_positions += len(labels)
        total_pad_ids += labels.count(PAD_ID)
        total_ignored += labels.count(IGNORE_ID)

        pad_positions = [
            i for i, token_id in enumerate(labels)
            if token_id == PAD_ID
        ]

        if not pad_positions:
            continue

        rows_with_pad += 1

        if examples_shown < max_examples:
            first_pad = pad_positions[0]

            eos_positions = [
                i for i, token_id in enumerate(labels)
                if token_id == EOS_ID
            ]
            first_eos = eos_positions[0] if eos_positions else None

            trailing_pad_only = all(
                token_id == PAD_ID
                for token_id in labels[first_pad:]
            )

            print(f"\nExample index: {index}")
            print("Label sequence length:", len(labels))
            print("PAD_ID positions:", pad_positions[:20])
            print("First EOS position:", first_eos)
            print(
                "All tokens from first PAD onward are PAD_ID:",
                trailing_pad_only,
            )
            print(
                "PAD occurs after EOS:",
                first_eos is not None and first_pad > first_eos,
            )

            # Show the target text without pad and ignore tokens.
            decoded_ids = [
                token_id for token_id in labels
                if token_id not in (PAD_ID, IGNORE_ID)
            ]
            print(
                "Decoded target without PAD_ID:",
                tokenizer.decode(
                    decoded_ids,
                    skip_special_tokens=False,
                ),
            )

            examples_shown += 1

    print("\n--- Dataset totals ---")
    print("Examples:", len(dataset))
    print("Total stored label positions:", total_positions)
    print("Occurrences of PAD_ID:", total_pad_ids)
    print("Occurrences of IGNORE_ID:", total_ignored)
    print("Examples containing PAD_ID:", rows_with_pad)


inspect_stored_labels(trainer.train_dataset, "TRAIN")
inspect_stored_labels(trainer.eval_dataset, "EVALUATION")

def inspect_actual_batch(dataloader, split_name):
    print(f"\n===== {split_name}: COLLATED BATCH =====")

    batch = next(iter(dataloader))
    labels = batch["labels"].detach().cpu()

    active_pad_mask = labels.eq(PAD_ID)
    ignored_mask = labels.eq(IGNORE_ID)

    print("Label tensor shape:", tuple(labels.shape))
    print("Total label positions:", labels.numel())
    print("PAD_ID positions still present:", active_pad_mask.sum().item())
    print("Ignored positions (-100):", ignored_mask.sum().item())

    for row_index, row in enumerate(labels.tolist()):
        pad_positions = [
            i for i, token_id in enumerate(row)
            if token_id == PAD_ID
        ]

        if pad_positions:
            eos_positions = [
                i for i, token_id in enumerate(row)
                if token_id == EOS_ID
            ]

            print(f"\nBatch row: {row_index}")
            print("PAD_ID positions:", pad_positions[:20])
            print(
                "EOS positions:",
                eos_positions[:10],
            )


inspect_actual_batch(
    trainer.get_train_dataloader(),
    "TRAIN",
)

inspect_actual_batch(
    trainer.get_eval_dataloader(),
    "EVALUATION",
)



trainer.train()



trainer.save_model("./testing_seq2seq_model")


print(f"\n Number of GPUs used in trainer {trainer.args.n_gpu}")
print(f"\n World size: {trainer.args.world_size}")

try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    pass
