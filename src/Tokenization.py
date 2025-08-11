'''MODEL PREPROCESSING '''
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch 
import numpy as np
import random
import pickle 
import os
import math

#intialization of tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# configuration of tokenizer trainer
trainer = BpeTrainer(vocab_size = 60000, special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"])

# training tokenizer 
# tokenizer.train_from_iterator(texts,trainer)

# saving tokenizer 
# tokenizer.save("tokenizers/tokenizer_igbo_en.json")

""" Merging all data corpus for BPE tokenisation """
igbo_datasets = [
    "data/Flores200/Processed_csv/Igbo/ibo_en_dev.csv",
    "data/Flores200/Processed_csv/Igbo/ibo_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Igbo_eng.csv",
    "data/Jw300/Processed_csv/Igbo/ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/CCAligned_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/GNOME_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/Ubuntu_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/WikiTitles_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/Tatoeba_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/XLENT_ig_en.csv",
    "data/Tatoeba/preprocessed_csv/ibo_en.csv"
]

yoruba_datasets = [
    "data/Flores200/Processed_csv/Yoruba/yor_en_dev.csv",
    "data/Flores200/Processed_csv/Yoruba/yor_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Yoruba_eng.csv",
    "data/Opus/Preprocessed_csv/Yoruba/GlobalVoices_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/GNOME_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/NLLB_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/QED_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/Tatoeba_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/Ubuntu_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/wikimedia_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/XLENT_yo_en.csv",
    "data/Tatoeba/preprocessed_csv/yor_en.csv"
]

hausa_datasets = [
    "data/Flores200/Processed_csv/Hausa/ha_en_dev.csv",
    "data/Flores200/Processed_csv/Hausa/ha_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Hausa_eng.csv",
    "data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv",
    "data/Opus/Preprocessed_csv/Hausa/QED_ha_en.csv",
    "data/Opus/Preprocessed_csv/Hausa/WikiTitles_ha_en.csv",
    "data/Tatoeba/preprocessed_csv/hau_en.csv"
]

all_datasets = [
    "data/Flores200/Processed_csv/Yoruba/yor_en_dev.csv",
    "data/Flores200/Processed_csv/Yoruba/yor_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Yoruba_eng.csv",
    "data/Opus/Preprocessed_csv/Yoruba/GlobalVoices_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/GNOME_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/NLLB_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/QED_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/Tatoeba_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/Ubuntu_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/wikimedia_yo_en.csv",
    "data/Opus/Preprocessed_csv/Yoruba/XLENT_yo_en.csv",
    "data/Tatoeba/preprocessed_csv/yor_en.csv",
    "data/Flores200/Processed_csv/Igbo/ibo_en_dev.csv",
    "data/Flores200/Processed_csv/Igbo/ibo_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Igbo_eng.csv",
    "data/Jw300/Processed_csv/Igbo/ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/CCAligned_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/GNOME_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/Ubuntu_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/WikiTitles_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/Tatoeba_ig_en.csv",
    "data/Opus/Preprocessed_csv/Igbo/XLENT_ig_en.csv",
    "data/Tatoeba/preprocessed_csv/ibo_en.csv",
    "data/Flores200/Processed_csv/Hausa/ha_en_dev.csv",
    "data/Flores200/Processed_csv/Hausa/ha_en_devtest.csv",
    "data/Hypa_fleurs/Processed_csv/Hausa_eng.csv",
    "data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv",
    "data/Opus/Preprocessed_csv/Hausa/QED_ha_en.csv",
    "data/Opus/Preprocessed_csv/Hausa/WikiTitles_ha_en.csv",
    "data/Tatoeba/preprocessed_csv/hau_en.csv",
]
"""Merging dataset"""
def merge_datasets(lang_name,Datasets_for_lang):
    text = []
    for file in Datasets_for_lang:
        df = pd.read_csv(file)
        text.extend(df[lang_name].tolist())
    print(f"A total of {len(text)} lines of {lang_name} has been merged.")
    return text 

all_igbo = merge_datasets("igbo",igbo_datasets)
all_yoruba = merge_datasets("yoruba",yoruba_datasets)
all_hausa = merge_datasets("hausa",hausa_datasets)
all_english = merge_datasets("english", all_datasets)

"""Saving clean merged data as pickle file"""
def save_merged_file(file_path, list_of_items):
    try:
        with open(file_path,"wb") as file:
            pickle.dump(list_of_items,file)
            print(f"Dataset merged and saved to {file_path}.")
    except Exception as e :
        print(f"unable to store because of {e}")

save_merged_file("data/Merged_data/Hausa/all_hausa.pkl",all_hausa)
save_merged_file("data/Merged_data/Igbo/all_igbo.pkl",all_igbo)
save_merged_file("data/Merged_data/Yoruba/all_yoruba.pkl",all_yoruba)
save_merged_file("data/Merged_data/English/all_english.pkl",all_english)

"""Loading merged pickle files"""
def load_pickle_file(pickle_file_path):
    try:
        with open (pickle_file_path, "rb") as file:
            loaded_lists = pickle.load(file)
            type(loaded_lists)
            return loaded_lists
    except Exception as e:
        print(f"Error occured during pickle --- {e}")

# pickle_igbo_path = "data/Merged_data/Igbo/merged_igbo_eng.pkl"
# clean_merged_igbo_eng = load_pickle_file(pickle_igbo_path)


''' intialization of tokenizer '''
# tokenizer = Tokenizer(BPE())
# tokenizer.pre_tokenizer = Whitespace()
# # configuration of tokenizer trainer
# trainer = BpeTrainer(vocab_size = 60000, special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"])
# # training tokenizer 
# tokenizer.train_from_iterator(clean_merged_igbo_eng,trainer)
# # saving tokenizer 
# tokenizer.save("Tokenizers/tokenizer_igbo_en.json")


"""Pretokenization tests -> derivation of optimal maximum length for tokenization """
def max_length_evaluator(lang_eng_list_raw_text):
    encoded_lengths = []
    unencoded_strings = []
    for s in lang_eng_list_raw_text:
        try:
            encoded_lengths.append(len(tokenizer.encode(f"<sos> {s} <eos>").ids))
        except Exception as e:
            # print(f"Warning: Could not encode string '{s[:50]}...': {e}. Skipping this item.")
            unencoded_strings.append(s)
            continue
    results = (f"90th percentile length: {np.percentile(encoded_lengths, 90):.2f}\n" +
               f"95th percentile length: {np.percentile(encoded_lengths, 95):.2f}\n")
    
    return results, f"the number of uncoded string is {len(unencoded_strings)}\n the number of encoded strings is {len(encoded_lengths)}"
# print(max_length_evaluator(all_ibo_eng))



""" tokenization of datasets """
def tokenize_sentences(sentences, tokenizer, max_length = 60):
    # transformation of sentence to tokens with 
    encodings = [tokenizer.encode(f"<sos> {s} <eos>").ids for s  in sentences]
    # creation of empty tensor
    padded = torch.zeros(len(encodings), max_length, dtype=torch.long)
    # padding out our encodings
    for i, sen_tokens in enumerate(encodings):
        length = min(len(sen_tokens), max_length)
        padded[i, :length] = torch.tensor(sen_tokens[:length], dtype = torch.long)
    return padded 

#save tokenized data set 
loaded_tokenizer = Tokenizer.from_file("Tokenizers/tokenizer_igbo_en.json")
if "<unk>" in loaded_tokenizer.get_vocab():
    loaded_tokenizer.unk_token = "<unk>"  # Set unk_token explicitly
    print("unk_token set to:", loaded_tokenizer.unk_token)
else:
    print("Error: <unk> not in vocabulary! Retrain the tokenizer.")

# tokenzised_lang_tensor = tokenize_sentences(clean_merged_igbo_eng,loaded_tokenizer)
# print(torch.is_tensor(tokenzised_lang_tensor))

# torch.save(tokenzised_lang_tensor,"data/Tokenized_data/Igbo/tokenized_ibo_eng.pt")

"""Posttokenization  tests (tests to scrutinize tokenizers)"""
# Test for frequency of unknown token id 
def frequency_unknown_ids(lang_eng_list_tensors):
    print(f"Type of object received: {type(lang_eng_list_tensors)}")
    print(f"Content of object received (first 5 elements): {lang_eng_list_tensors[:5]}")
    # Count of unknown tokens in both sequences 
    unk_token_id = tokenizer.token_to_id("<unk>")
    unk_tag_count = (lang_eng_list_tensors == unk_token_id).sum().item()

    return (
        f"Total <unk> tokens in target: {unk_tag_count}\n"
        f"Percentage <unk> in target: {unk_tag_count / lang_eng_list_tensors.nume* 100:.2f}%\n\n"
    )

# loaded_lang_tensor = torch.load("data/Tokenized_data/Igbo/tokenized_ibo_eng.pt")
# print(frequency_unknown_ids(loaded_lang_tensor))

# Testng tokenizers encoding accuracy  
def round_trip_check(merged_lang_tensor, tokenizer, num_samples=10000):

    effective_samples = min(len(merged_lang_tensor), num_samples)

    # Randomly sample sentences from the provided list
    sample_sentences = random.sample(merged_lang_tensor, k=effective_samples)
    disconcordant_count = 0
    failed_encodings = 0

    for original_sentence in sample_sentences:
        try:
            encoded_ids = tokenizer.encode(f"<sos> {original_sentence} <eos>").ids
            cleaned_original = " ".join(original_sentence.split())
            decoded_sentence = tokenizer.decode(encoded_ids, skip_special_tokens=True) 
            if cleaned_original != decoded_sentence:
                disconcordant_count += 1
        except Exception as e:
            # print(f"Warning: Encoding failed for sentence '{original_sentence[:50]}...': {e}. Skipping.")
            failed_encodings += 1
            continue
    return (
        f"Total sentences checked: {effective_samples}\n"
        f"Total disconcordant sentences found: {disconcordant_count}\n"
        f"Total failed encodings found: {failed_encodings}\n"
        f"Concordance Rate: {((effective_samples - disconcordant_count) / effective_samples) * 100:.2f}%\n"
    )

# print(round_trip_check(clean_merged_igbo_eng , loaded_tokenizer))