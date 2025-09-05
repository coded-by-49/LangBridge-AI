import pandas as pd
import os
from itertools import zip_longest
import logging
import xml.etree.ElementTree as ET
import pickle 
import csv
from glob import glob
from pathlib import Path

def is_allowed_block(char) -> bool: 
    """Detection of non-nigerian-lingual-unicode characters"""
    allowed_ranges = [
    (0x0020, 0x007E),  # Basic Latin (printable ASCII, excluding control characters)
    (0x0080, 0x00FF),  # Latin-1 Supplement (e.g., é, ñ)
    (0x0100, 0x017F),  # Latin Extended-A (e.g., ŋ, ṣ)
    (0x0180, 0x024F),  # Latin Extended-B (e.g., ɓ, ẹ, Ọ)
    (0x1E00, 0x1EFF),  # Latin Extended Additional (e.g., ḿ, ọ)
        ]
    char_code = ord(char)

    return any(start <= char_code <= end for start, end in allowed_ranges)

def is_non_latin_scripts(line, non_latin_char_thresold = 0.1):
    """Detection of non-latin and empty lines """
    if not line.strip():
        return True
    
    un_allowed_chars = sum(1 for c in line if not is_allowed_block(c))
    degree_of_incorrectness = un_allowed_chars/len(line) # why not (len(strip(line)))

    return non_latin_char_thresold <= degree_of_incorrectness


def get_malformed_indices(file_path, control_threshold= 0.05, repeat_thresold = 0.2):
    """Get indices of malformed lines (empty or high control characters)."""
    malformed_indices = set()

    with open(file_path, "r") as f:
        for i,line in enumerate(f):
            if is_non_latin_scripts(line):
                malformed_indices.add(i)

    with open(file_path, "rb") as f:
        for i, line in enumerate(f):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                
                if not decoded or len(decoded) == 0:
                    malformed_indices.add(i)
                    continue
                
                replacement_char_count = sum(1 for c in decoded if ord(c) < 32 or c == "\uFFFD")
                if replacement_char_count / len(decoded) >= control_threshold:
                    malformed_indices.add(i)
                    continue

                repeated_char_count = max(decoded.count(c) for c in set(decoded))
                if ( repeated_char_count/len(decoded) ) >= repeat_thresold:
                    malformed_indices.add(i) 

            except Exception:
                malformed_indices.add(i)
    print(f"Total lines: {i + 1}, Malformed lines: {len(malformed_indices)}")
    return malformed_indices

"""REARRANGEMENT OF TEXT FILES AFTER REMOVAL OF CORRUPTED LINES"""
def filter_files_by_indices(input_en, input_lang, output_en, output_ha, malformed_indices):
    with open(input_en, "rb") as en_file, \
         open(input_lang, "rb") as lang_file, \
         open(output_en, "w", encoding="utf-8") as clean_eng_file, \
         open(output_ha, "w", encoding="utf-8") as clean_lang_file:
        for i, (en_line, lang_line) in enumerate(zip(en_file, lang_file)):
            if i not in malformed_indices:
                try:
                    clean_eng_file.write(en_line.decode("utf-8", errors="replace").strip() + "\n")
                    clean_lang_file.write(lang_line.decode("utf-8", errors="replace").strip() + "\n")
                except Exception:
                    continue
    print(f"Filtered files saved: {output_en}, {output_ha}")


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


"""MERGING DATASET"""
def merge_datasets(lang_name,Datasets_for_lang):
    text = []
    for file in Datasets_for_lang:
        df = pd.read_csv(file)
        text.extend(df[lang_name].tolist())
    print(f"A total of {len(text)} lines of {lang_name} has been merged.")
    return text 

# Saving clean merged data as pickle file
def save_merged_file(file_path, list_of_items):
    try:
        with open(file_path,"wb") as file:
            pickle.dump(list_of_items,file)
            print(f"Dataset merged and saved to {file_path}.")
    except Exception as e :
        print(f"unable to store because of {e}")

"""CONVERSION OF FILES TO CSV"""
def convert_to_csv(clean_en_file,clean_lang_file,lang_name,output_csv,batch_size = 10000):
    pd.DataFrame(columns=[lang_name,"english"]).to_csv(output_csv,index=False)
    with open(clean_en_file,"rb") as en_file,\
         open(clean_lang_file,"rb") as lang_file:
        chunk = []
        for i,(en_line, ha_line) in enumerate(zip(en_file,lang_file)):
            try:
                # decode each line and mark out errors withing them 
                en_text = en_line.decode("utf-8",errors="replace").strip()
                lang_text = ha_line.decode("utf-8", errors = "replace").strip()
                
               
                if en_text and lang_text: #ensuring both are non-empty
                    chunk.append({lang_name:lang_text,"english":en_text})
            except Exception as e:
                raise f'{e}'
                continue

            # append each chunk to the output_csv
            if len(chunk)==batch_size:
                try:
                    pd.DataFrame(chunk).to_csv(output_csv, mode = "a", header=False, index=False)
                    chunk = [] #rest chunk list again
                except Exception as e:
                    print(f"Unable to carry out operation due to error : {e}")

        if chunk: #append remaining lines (less that batch size)
            pd.DataFrame(chunk).to_csv(output_csv, mode = "a", header=False, index=False)
            print(f"preprocessed {i+1} lines (final batch)")

        try:
            df_final = pd.read_csv(output_csv) # Use a different variable name than 'df' from top
            print(f"Saved {len(df_final)} {lang_name}-English pairs to {output_csv} \n")
        except pd.errors.EmptyDataError:
            print(f"No data saved to {output_csv}, only headers present.")
        except Exception as e:
            print(f"Error reading final CSV for verification: {e}")
   

"""CONVERSION OF RAW CSV TO TEXT FILES"""
def convert_tsv_to_txt(file_paths):
    for file_path in file_paths:
        file_path = Path(file_path)
        df = pd.read_csv(file_path, header=None)

        if df.shape[1] < 2:
            print(df.columns)
            print(f"Skipping {file_path}: less than 3 columns")
            continue
        df = df.iloc[:, 1:3]

        df.to_csv(file_path.with_suffix('.txt'), sep='\t', header=False, index=False)

        print(f"Processed {file_path} -> {file_path.with_suffix('.txt')}")

""" CONVERSION OF PARQUET FILE TO CSV """
def hypafleurs_load_parquet(lang_file, eng_file, lang_name, output_csv):
    lang_df = pd.read_parquet(lang_file)
    eng_df = pd.read_parquet(eng_file)

    # confirm if the length matches its english translation
    assert len(lang_df) == len(eng_df), f"Error: {lang_file} ({len(lang_df)}) and {eng_file} ({len(eng_df)}) have different lengths!"

    print(f"{lang_name} columns:", lang_df.columns)
    print("\n")
    print("English columns:", eng_df.columns)

    df_lang_with_eng = pd.DataFrame({
        lang_name.lower(): lang_df["text"],
        "english": eng_df["text"]
    })
    
    df_lang_with_eng.to_csv(output_csv, index=False)
    
    print(f"Saved {len(df_lang_with_eng)} {lang_name}-English pairs to {output_csv}")

df = pd.read_parquet("data/parrallel_text/hausa_en_corpus/unprocessed_file/hau1.parquet")
# print(df.head)
df.to_csv("data/parrallel_text/igbo_en_corpus/csv_formatted/igbo_par.csv", index=False, encoding="utf-8")


"""EXTRACTION OF DATA EAF AND EAFL FILES"""
def extract_annotations(eaf_l_file_paths, destination_file_path):
    try:
        with open(destination_file_path, "rb") as f :
            large_lang_file = pickle.load(f)
    except Exception as e:
        print(f"either the file is non-existant or {e}")
    
    merged_file_len = len(large_lang_file)

    for file_path in eaf_l_file_paths :
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # for annotations file 
        for ann_val in root.findall(".//ANNOTATION_VALUE"):
            if ann_val.text:
                annotation_text = ann_val.text.strip()
                if 'X' in annotation_text or 'x' in annotation_text or '#' in annotation_text:
                        continue  
                large_lang_file.append(annotation_text)

        # for links file 
        id_to_text = {}
        for seg in root.findall(".//{*}seg"):
            seg_id = seg.attrib.get("id")
            text = seg.text.strip() if seg.text else ""
            id_to_text[seg_id] = text

        for link in root.findall(".//{*}link"):
            xtargets = link.attrib.get("xtargets", "")
            target_ids = xtargets.split(";")
            texts = [id_to_text.get(tid, "") for tid in target_ids]
            large_lang_file.append(texts)

    with open(destination_file_path, 'wb') as f:
        pickle.dump(large_lang_file, f)
    print(f"finished processing {len(eaf_l_file_paths)} files and \n the merged igbo dataset has now increased to  {len(large_lang_file) - merged_file_len}")

"""ADDING TEXT FILE TO MERGED LINGUAL DATA"""
def extend_merged_data(file,merged_data_path):
    try:
        with open(merged_data_path, "rb") as f :
            large_lang_file = pickle.load(f)
    except Exception as e:
        print(f"either the file is non-existant or {e}")
    
    start_len = len(large_lang_file)
    non_file_count = 0
    num_file = 0

    if file.is_file():
        if file.suffix.lower() == ".csv":
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    large_lang_file.extend(row)
        else:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip()]
            large_lang_file.extend(lines)
    else:
        print(f"This is not a file")
        
    end_len = len(large_lang_file)
    with open(merged_data_path, 'wb') as f:
        pickle.dump(large_lang_file, f)

    print(f"finished processing {num_file} files and \n the merged igbo dataset has now increased to  {end_len-start_len}")
    print(f"number of non_files = {non_file_count}")

def process_paraquet(paraquet_files):
    for file in paraquet_files:
        para_pd = pd.read_parquet(file)
        para_pd = para_pd[['igbo_cleaned','english_cleaned']]
        out_file = Path(file).with_suffix(".txt")
        para_pd.to_csv(out_file,header=False, sep= "\t")

"""TRANSFORMATION OF PARALLEL DATA TO SFTTRAINER FORMAT"""
def csvfiles_to_parrallel(csv_path,src_lang,target_lang):
    df = pd.read_csv(csv_path, sep = "\t")
    print(f"Succesfully read {csv_path}")
    df = df[[src_lang,target_lang]]
    df.to_csv(csv_path, index=False)

def merge_txtfiles_corpus(src_lang,target_lang,src_lang_txt,target_lang_txt,target_csv):
    src_txts = [line.strip() for line in open(src_lang_txt, "r", encoding="utf-8")]
    target_txts = [line.strip() for line in open(target_lang_txt, "r", encoding="utf-8")]
    
    assert len(src_txts) == len(target_txts), "disconcordant corpus"
    len_corpus = len(src_txts)
    parrallel_pairs = list(zip(src_txts,target_txts))
    df = pd.DataFrame(parrallel_pairs,columns=[src_lang,target_lang])
    if not df.empty:
        df.to_csv(target_csv,index = False)
        print(f"successfully processed {len_corpus} into {target_csv}")
    else:
        print(f"The dataframe is empty")

def txtfiles_to_corpus(file_path,output_csv):
    igbo_lines = [
        line.replace("Igbo:", "", 1).strip()
        for line in open(file_path, "r", encoding="utf-8")
        if line.startswith("Igbo:")
    ]
    english_lines = [
        line.replace("Eng:", "", 1).strip()
        for line in open(file_path, "r", encoding="utf-8")
        if line.startswith("Eng:")
    ]

    assert len(igbo_lines) == len(english_lines), f"Mismatch between Igbo and English lines! in {file_path}\n There are {len(igbo_lines)} igbo lines and {len(english_lines)} english lines"

    df = pd.DataFrame({
        "Igbo": igbo_lines,
        "English": english_lines
    })

    
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} translation pairs to {output_csv}")

merge_parameters = [
    ("data/parrallel_text/Yor_en_corpus/jw300.yo.txt", "data/parrallel_text/Yor_en_corpus/jw300.en.txt", "data/parrallel_text/Yor_en_corpus/csv_formatted/jw300_yo_en.csv")
]
txt_to_corp_parameters = [
    ("data/parrallel_text/igbo_en_corpus/gpt3.5_ted_talk_igbo.txt","data/parrallel_text/igbo_en_corpus/csv_formatted/gpt3.5_ted_talk_igbo.csv"),
    ("data/parrallel_text/igbo_en_corpus/gpt4_bbc_igbo.txt","data/parrallel_text/igbo_en_corpus/csv_formatted/gpt4_bbc_igbo.csv"),
    ("data/parrallel_text/igbo_en_corpus/gpt4_igbo.gov.txt","data/parrallel_text/igbo_en_corpus/csv_formatted/gpt4_igbo.gov.csv"),
    ("data/parrallel_text/igbo_en_corpus/gpt4_ted_talk_igbo.txt","data/parrallel_text/igbo_en_corpus/csv_formatted/gpt4_ted_talk_igbo.csv")
]

