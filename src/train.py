'''MODEL PREPROCESSING '''
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch 
import numpy as np

# Set CUDA as default device
# if torch.cuda.is_available():
#     torch.set_default_device('cuda')
#     print("Default device set to CUDA:", torch.cuda.get_device_name(0))
# else:
#     torch.set_default_device('cpu')
#     print("CUDA not available, default device set to CPU")

df = pd.read_csv("data/Opus/Preprocessed_csv/Igbo/CCAligned_ig_en.csv")

#intialisation of tokeniser
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# configuration of tokenizer trainer
trainer = BpeTrainer(vocab_size = 60000, special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"])
texts = df['igbo'].tolist() + df['english'].tolist()

# training tokenizer 
tokenizer.train_from_iterator(texts,trainer)
tokenizer.save("tokenizers/tokenizer_igbo_en.json")



"""Pretokenization tests -> optimal maximum length check """
def max_length_evaluator(source_df , target_df, source_lang_name, target_lang_name):
    target_raw_encodings = [tokenizer.encode(f"<sos> {s} <eos>").ids for s in target_df[target_lang_name].tolist()]
    source_raw_encodings = [tokenizer.encode(f"<sos> {s} <eos>").ids for s in source_df[source_lang_name].tolist()]
    all_lengths = [len(e) for e in source_raw_encodings] + [len(e) for e in target_raw_encodings]
    results = f"90th percentile length: {np.percentile(all_lengths, 90):.2f}\n"+f"95th percentile length: {np.percentile(all_lengths, 95):.2f}\n"
    return results

source_sequence = df['igbo']
target_sequence = df['english']

print(max_length_evaluator(source_sequence,target_sequence,"igbo","english"))

# fitting tokenizer to sentences of dataset
def tokenize_sentences(sentences, tokenizer, max_length = 128, device = "cuda"):
    # transformation of sentence to tokens with 
    encodings = [tokenizer.encode(f"<sos> {s} <eos>").ids for s  in sentences]
    # creation of empty tensor
    padded = torch.zeros(len(encodings), max_length, dtype=torch.long)
    # padding out our encodings

    for i, sen_tokens in enumerate(encodings):
        length = min(len(sen_tokens), max_length)
        padded[i, :length] = torch.tensor(sen_tokens[:length], dtype = torch.long)
    return padded 


# # testing
# print("Sample igbo sequence:", source_sequence[0])
# print("Sample English sequence:", target_sequence[0])
# print("Vocabulary size:", tokenizer.get_vocab_size())

# #saveing tokenized datasets
# torch.save(source_sequence, 'data/Tokenised_Data/Igbo/source_tokens_s1.pt')
# torch.save(target_sequence, 'data/Tokenised_Data/Igbo/target_tokens_s1.pt')


"""Posttokenization tests (tests to scrutinize tokenizers)"""
# '''Test for frequency of unknown token id '''
def frequency_unknown_ids(source_sequences, target_sequences):
    # Count of unknown tokens in both sequences 
    unk_token_id = tokenizer.token_to_id("<unk>")
    target_unk_count = (target_sequences == unk_token_id).sum().item()
    source_unk_count = (source_sequences == unk_token_id).sum().item()

    return (
        f"Total <unk> tokens in target: {target_unk_count}\n"
        f"Percentage <unk> in target: {target_unk_count / target_sequences.numel() * 100:.2f}%\n"
        f"Total <unk> tokens in source: {source_unk_count}\n"
        f"Percentage <unk> in source: {source_unk_count / source_sequences.numel() * 100:.2f}%"
    )

# Testng tokens ability to translate 
def round_trip_check(lang_name,df,):
    sample_indices = df.sample(n =min(len(df), 1000), random_state=42).index.tolist() #!!
    disconcordant_lang_check = 0
    disconcordant_eng_check = 0
    total_checks = 0

    for idx in sample_indices:
        original_lang = df[lang_name].iloc[idx]
        original_english = df["english"].iloc[idx]
        
        # encode the lan sentence sentence 
        lang_sentence_id = tokenizer.encode(f"<sos> {original_lang} <eos>").ids
        eng_sentence_id = tokenizer.encode(f"<sos> {original_english} <eos>").ids

        # decode your encoding 
        # how come skip_special_tokens is true here ?, our tokenizer trainer had special characters
        decoded_lang_sentence = tokenizer.decode(lang_sentence_id, skip_special_tokens = True)
        decoded_eng_sentence = tokenizer.decode(eng_sentence_id, skip_special_tokens = True)

        cleaned_orignal_lang_ = " ".join(original_lang.split())

        if cleaned_orignal_lang_ != decoded_lang_sentence:
            disconcordant_lang_check += 1
        total_checks +=1

        cleaned_orignal_english = " ".join(original_english.split())

        if cleaned_orignal_english != decoded_eng_sentence:
            disconcordant_eng_check += 1
        total_checks +=1 

    return (f" total sentences checked: {total_checks}\n total disconcordants found in english: {disconcordant_eng_check} \n total disconcordants found in {lang_name}: {disconcordant_lang_check}")