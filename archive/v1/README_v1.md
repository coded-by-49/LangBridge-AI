LangBridge AI: High-Performance NMT for Nigerian Languages

🌍 Overview

LangBridge AI is a specialized Neural Machine Translation (NMT) project designed to bridge the communication gap for the three major Nigerian languages: Hausa, Igbo, and Yoruba (Wazobia), paired with English.

While general-purpose models often underperform on low-resource African languages due to data scarcity, LangBridge AI mitigates this by leveraging a massively curated dataset of over 4 million high-quality sentence pairs and fine-tuning Meta's state-of-the-art NLLB-200 model using Low-Rank Adaptation (LoRA).

🚀 Key Features

Massive Scale: Trained on 4M+ cleaned and tokenized parallel sentences.

Efficient Architecture: Fine-tuned nllb-200-distilled-600M using QLoRA (4-bit quantization) for optimal performance-to-resource ratio.

Multi-Domain Coverage: Data sourced from religious texts, news, web scrapes, and conversational corpora.

📊 The Dataset: LangBridge Wazobia

The foundation of this model is the custom-built LangBridge Wazobia Dataset. Creating this dataset involved a rigorous pipeline of data mining, cleaning, and formatting.

Sources include:

FLORES-200: High-quality benchmark data.

JW300: Extensive parallel religious texts.

OPUS Corpus: Including CCMatrix, Wikititles, and XLEnt.

Tatoeba: Community-sourced translation pairs.

Hypa_Fleurs: Multi-reference datasets.

Custom Web Scraping: Targeted extraction from localized Nigerian web content.

Preprocessing Pipeline:

Deduplication: Removal of overlapping entries across sources.

Cleaning: Regex-based filtering to remove non-text artifacts and noise.

Tokenization: Processed using the NLLB tokenizer (SentencePiece) for immediate transformer usage.

👉 Access the Dataset on Hugging Face

🛠️ Model Architecture & Training

Base Model: facebook/nllb-200-distilled-600M

Fine-Tuning Method: LoRA (Low-Rank Adaptation)

Rank: 32

Alpha: 64

Target Modules: q_proj, v_proj, k_proj, out_proj, fc1, fc2

Quantization: 4-bit NormalFloat (NF4) via bitsandbytes to reduce memory footprint while maintaining precision.

Infrastructure: Trained on NVIDIA Tesla T4 GPUs.

💻 Usage

To use the model for inference, you need to load the base NLLB model and attach the LangBridge adapters.

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Setup Configuration
base_model_id = "facebook/nllb-200-distilled-600M"
adapter_model_id = "coded-by-49/Lang_bridge_AI"

# 2. Load Tokenizer & Base Model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Load LangBridge Adapters
model = PeftModel.from_pretrained(model, adapter_model_id)
model.eval()

# 4. Inference Function
def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # NLLB Language Codes: 
    # Igbo: "ibo_Latn", Hausa: "hau_Latn", Yoruba: "yor_Latn", English: "eng_Latn"
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id, 
        max_new_tokens=100
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Example: Translate English to Igbo
print(translate("How are you doing today?", "eng_Latn", "ibo_Latn"))
# Output: Kedu ka ị na-eme taa?


📈 Evaluation

The model was evaluated using SacreBLEU scores to ensure translation accuracy and fluency across all three language pairs.

📜 License

This project is open-source and available under the Apache 2.0 License.

🤝 Acknowledgements

Special thanks to the open-source community, specifically the creators of the FLORES-200 and OPUS projects, which made the data collection for this low-resource task possible.