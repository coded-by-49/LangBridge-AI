import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, PeftModel
from peft import get_peft_model, TaskType
from trl import SFTTrainer

bnb_config = BitsAndBytesConfig(
    oad_in_4bit=True,
    bnb_4bit_quant_type="nf4",               
    bnb_4bit_compute_dtype=torch.bfloat16,   
    bnb_4bit_use_double_quant=True, 
)

# The 600M model is called "facebook/nllb-200-distilled-600M"
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
    quantization_config = bnb_config,
    device_map = "auto")


training_args = Seq2SeqTrainingArguments(
    output_dir="src/Finetuned_model",          # Directory to save the model
    evaluation_strategy="epoch",     # Evaluate every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,              
    predict_with_generate=True,
    fp16=True,                       
    push_to_hub=False,               
)


# IMPLEMENATION OF QLORA 


lora_config = LoraConfig(
    r=16,                                    
    lora_alpha=32,                           
    target_modules=["q_proj", "v_proj"],    
    lora_dropout=0.05,                       
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  
)


peft_model = get_peft_model(model,lora_config)

# testing 
peft_model.print_trainable_parameters()