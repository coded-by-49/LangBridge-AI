import pandas as pd
import os
from itertools import zip_longest
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def is_allowed_block(char):
    """Detection of non-nigerian-unicode characters"""
    try:
        block = unicodedata.block(char)
    allowed_blocks = 
def get_malformed_indices(file_path, threshold= 0.9):
    """Get indices of malformed lines (empty or high control characters)."""
    malformed_indices = []
    with open(file_path, "rb") as f:
        for i, line in enumerate(f):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                # append all empty lines here 
                if not decoded or len(decoded) == 0:
                    malformed_indices.append(i)
                    continue
                control_count = sum(1 for c in decoded if ord(c) < 32 or c == "\uFFFD")
                # if the percentage of control characters supasses thresold (it is considered a malformed line)
                if control_count / len(decoded) >= threshold:
                    malformed_indices.append(i)
            except Exception:
                # any line that gives us an error is considered a malformed line
                malformed_indices.append(i)
    print(f"Total lines: {i + 1}, Malformed lines: {len(malformed_indices)}")
    return malformed_indices

"""REARRANGEMENT OF TEXT FILES AFTER REMOVAL OF CORRUPTED LINES"""
def filter_files_by_indices(input_en, input_lang, output_en, output_ha, malformed_indices):
    """Remove lines at specified indices from both files."""
    with open(input_en, "rb") as en_file, \
         open(input_lang, "rb") as ha_file, \
         open(output_en, "w", encoding="utf-8") as clean_en, \
         open(output_ha, "w", encoding="utf-8") as clean_ha:
        for i, (en_line, ha_line) in enumerate(zip(en_file, ha_file)):
            if i not in malformed_indices:
                try:
                    clean_en.write(en_line.decode("utf-8", errors="replace"))
                    clean_ha.write(ha_line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
    print(f"Filtered files saved: {output_en}, {output_ha}")

def covert_to_csv(clean_en_file,clean_lang_file,lang_name,output_csv,batch_size = 10000):
    pd.DataFrame(columns=[lang_name,"english"]).to_csv(output_csv,index=False)
    with open(clean_en_file,"rb") as en_file,\
         open(clean_lang_file,"rb") as lang_file:
        chunk = []
        for i,(en_line, ha_line) in enumerate(zip(en_file,lang_file)):
            try:
                # decode each line
                en_text = en_line.decode("utf-8",errors="replace").strip()
                ha_text = ha_line.decode("utf-8", errors = "replace").strip()
                
               
                if en_text and ha_text: #ensuring both are non-empty
                    chunk.append({lang_name:ha_text,"english":en_text})
            except Exception:
                continue

            # append each chunk to the output_csv
            if len(chunk)>=batch_size:
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




"""MALFORMED FILE CLEANING AND STORAGE """

# Function to get malformed indices (you already have this)
malformed_indices = get_malformed_indices("data/Opus/Igbo/unclean_files/en-ig.txt (3)/XLEnt.en-ig.en")

# Store all file paths in a tuple that matches the order for filter_files_by_indices
file_pairs = [
    (
        "data/Opus/Igbo/unclean_files/en-ig.txt (3)/XLEnt.en-ig.en",  # original_en_file
        "data/Opus/Igbo/unclean_files/en-ig.txt (3)/XLEnt.en-ig.ig",  # original_lang_file
        os.path.join("data/Opus/Igbo/clean_files/en-ig.txt(3)/XLEnt.en-ig.en"),  # clean_en_file
        os.path.join("data/Opus/Igbo/clean_files/en-ig.txt(3)/XLEnt.en-ig.ig")   # clean_lang_file
    )
]

# Loop through all file sets and apply filtering
for original_en, original_lang, clean_en, clean_lang in file_pairs:
    filter_files_by_indices(original_en, original_lang, clean_en, clean_lang, malformed_indices)


"""CREATION AND STORAGE OF CSV"""
para_for_preprocessing = [
    # ("data/Opus/Hausa/clean_files/CCMatrix_filtered.en","data/Opus/Hausa/clean_files/CCMatrix_filtered.ha","hausa","data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv")
    # ("data/Opus/Hausa/en-ha.txt(1)/WikiTitles.en-ha.en","data/Opus/Hausa/en-ha.txt(1)/WikiTitles.en-ha.ha","hausa","data/Opus/Preprocessed_csv/Hausa/WikiTitles_ha_en.csv")
    # ("data/Opus/Hausa/en-hau.txt(3)/QED.en-hau.en","data/Opus/Hausa/en-hau.txt(3)/QED.en-hau.hau","hausa","data/Opus/Preprocessed_csv/Hausa/QED_ha_en.csv")
    # ("data/Opus/Igbo/en-ibo.txt/QED.en-ibo.en","data/Opus/Igbo/en-ibo.txt/QED.en-ibo.ibo","igbo","data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv")
    # ("data/Opus/Igbo/en-ig.txt/GNOME.en-ig.en","data/Opus/Igbo/en-ig.txt/GNOME.en-ig.ig","igbo","data/Opus/Preprocessed_csv/Igbo/GNOME_ig_en.csv")

    #PROCESSED

    #UNPROCESSED
    # ("data/Opus/Igbo/clean_files/en-ig.txt(1)/XLEnt.en-ig.en","data/Opus/Igbo/clean_files/en-ig.txt(1)/XLEnt.en-ig.ig","igbo","data/Opus/Preprocessed_csv/Igbo/XLEnt_ig_en.csv"),
    # ("data/Opus/Igbo/clean_files/en-ig.txt(2)/CCAligned.en-ig.en","data/Opus/Igbo/clean_files/en-ig.txt(2)/CCAligned.en-ig.ig","igbo","data/Opus/Preprocessed_csv/Igbo/CCAligned_ig_en.csv") 
    
]

for clean_en_file,clean_lang_file,lang_name,output_csv in para_for_preprocessing:
    try:
        covert_to_csv(clean_en_file,clean_lang_file,lang_name,output_csv,batch_size=100)
    except Exception as e:
        print(f"{e}")


