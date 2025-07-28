import pandas as pd
import os
from itertools import zip_longest
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# def is_allowed_block(char) -> bool: 
#     """Detection of non-nigerian-lingual-unicode characters"""
#     try:
#         char_code = unicodedata.block(char)
#     except ValueError:
#         return False 
#     allowed_unicodes = {
#         # this summaizes all unicode types of the 3 languages being processed
#         "Basic Latin",             # U+0000..U+007F (includes ASCII characters)
#         "Latin-1 Supplement",      # U+0080..U+00FF (e.g., é, ñ, also some control codes)
#         "Latin Extended-A",        # U+0100..U+017F (Crucial for many African languages, e.g., ŋ, ṣ)
#         "Latin Extended-B",        # U+0180..U+024F (e.g., ɓ, ẹ, Ọ)
#         "Latin Extended Additional" # U+1E00..U+1EFF (e.g., ḿ, ọ - more specific Latin extensions)
#     }
#     return char_code in allowed_unicodes

# def is_non_latin_scripts(line, thresold = 0.2):
#     """Detection of non-latin and empty lines """
#     if not line.strip():
#         return True
#     un_allowed_chars = sum(1 for c in line if not is_allowed_block(c))
#     degree_of_incorrectness = un_allowed_chars/len(line) # why not (len(strip(line)))

#     return thresold <= degree_of_incorrectness

    # adding all non-latin scripts to malformed_indices
    # with open(file_path, "r") as f:
    #     for i,line in enumerate(f):
    #         if is_non_latin_scripts(line):
    #             malformed_indices.add(i)
    #             logging.info(f"Line {i+1} flagged")

def get_malformed_indices(file_path, threshold= 0.4, repeat_thresold = 0.4):
    """Get indices of malformed lines (empty or high control characters)."""
    malformed_indices = set()

    with open(file_path, "rb") as f:
        for i, line in enumerate(f):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                
                # flag all empty lines here 
                if not decoded or len(decoded) == 0:
                    malformed_indices.add(i)
                    continue
                control_count = sum(1 for c in decoded if ord(c) < 32 or c == "\uFFFD")
                
                # if the percentage of control characters supasses thresold (it is considered a malformed line)
                if control_count / len(decoded) >= threshold:
                    malformed_indices.add(i)
                
                # flag lines with to much repeated character
                repeated_char_count = max(decoded.count(c) for c in set(decoded))
                if ( repeated_char_count/len(decoded) ) >= repeat_thresold:
                    malformed_indices.add(i) 
            except Exception:
                # any line that gives us an error is considered a malformed line
                malformed_indices.add(i)
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
                    clean_en.write(en_line.decode("utf-8", errors="replace").strip() + "\n")
                    clean_ha.write(ha_line.decode("utf-8", errors="replace").strip() + "\n")
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

# get malformed indices
malformed_indices = get_malformed_indices("data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.ha")

# Store all file paths in a tuple that matches the order for filter_files_by_indices
file_pairs = [
    (
        "data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.en",
        "data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.ha",
        os.path.join("data/Opus/Hausa/clean_files/en-ha.txt(2)/CCMatrix.en-ha.en"),
        os.path.join("data/Opus/Hausa/clean_files/en-ha.txt(2)/CCMatrix.en-ha.ha")
    )
]

# Loop through all file sets and apply filtering
for original_en, original_lang, clean_en, clean_lang in file_pairs:
    filter_files_by_indices(original_en, original_lang, clean_en, clean_lang, malformed_indices)


"""CREATION AND STORAGE OF CSV"""
para_for_preprocessing = [
    ("data/Opus/Hausa/clean_files/en-ha.txt(2)/CCMatrix.en-ha.en","data/Opus/Hausa/clean_files/en-ha.txt(2)/CCMatrix.en-ha.ha","hausa","data/Opus/Preprocessed_csv/Hausa/CCMatrix_ha_en.csv")
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


