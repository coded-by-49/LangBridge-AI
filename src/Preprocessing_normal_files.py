import pandas as pd
import os
from itertools import zip_longest
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')

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
    
    # flag empty lines 
    if not line.strip():
        return True
    
    un_allowed_chars = sum(1 for c in line if not is_allowed_block(c))
    degree_of_incorrectness = un_allowed_chars/len(line) # why not (len(strip(line)))

    return non_latin_char_thresold <= degree_of_incorrectness


def get_malformed_indices(file_path, control_threshold= 0.05, repeat_thresold = 0.2):
    """Get indices of malformed lines (empty or high control characters)."""
    malformed_indices = set()

    # flagging all scirpts with too much non-nigerian words
    with open(file_path, "r") as f:
        for i,line in enumerate(f):
            if is_non_latin_scripts(line):
                malformed_indices.add(i)
                # logging.info(f"Line {i+1} flagged")

    with open(file_path, "rb") as f:
        for i, line in enumerate(f):
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
                
                # flag all empty lines 
                if not decoded or len(decoded) == 0:
                    malformed_indices.add(i)
                    continue
                
                # flag lines with too much replacement characters
                replacement_char_count = sum(1 for c in decoded if ord(c) < 32 or c == "\uFFFD")
                if replacement_char_count / len(decoded) >= control_threshold:
                    malformed_indices.add(i)
                    continue

                # flag lines with to much repeated characters
                repeated_char_count = max(decoded.count(c) for c in set(decoded))
                if ( repeated_char_count/len(decoded) ) >= repeat_thresold:
                    malformed_indices.add(i) 

                # flag lines with too much non-foerign characters
            except Exception:
                # any line that gives us an error is considered a malformed line
                malformed_indices.add(i)
    print(f"Total lines: {i + 1}, Malformed lines: {len(malformed_indices)}")
    return malformed_indices

"""REARRANGEMENT OF TEXT FILES AFTER REMOVAL OF CORRUPTED LINES"""
def filter_files_by_indices(input_en, input_lang, output_en, output_ha, malformed_indices):
    """Remove lines at specified indices from both files."""
    with open(input_en, "rb") as en_file, \
         open(input_lang, "rb") as lang_file, \
         open(output_en, "w", encoding="utf-8") as clean_en, \
         open(output_ha, "w", encoding="utf-8") as clean_ha:
        for i, (en_line, lang_line) in enumerate(zip(en_file, lang_file)):
            if i not in malformed_indices:
                try:
                    clean_en.write(en_line.decode("utf-8", errors="replace").strip() + "\n")
                    clean_ha.write(lang_line.decode("utf-8", errors="replace").strip() + "\n")
                except Exception:
                    continue
    print(f"Filtered files saved: {output_en}, {output_ha}")

def convert_to_csv(clean_en_file,clean_lang_file,lang_name,output_csv,batch_size = 10000):
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




"""MALFORMED FILE CLEANING AND STORAGE """

# Store all file paths in a tuple that matches the order for filter_files_by_indices
# file_pairs = [
#     (
#         "data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.en", #original engish file path 
#         "data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.yo", #original language file path 
#         "data/Opus/Yoruba/clean_files/en-yo.txt(11)/NLLB.en-yo.en", # destination english file path
#         "data/Opus/Yoruba/clean_files/en-yo.txt(11)/NLLB.en-yo.yo", # destination language file path
#         get_malformed_indices("data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.en") # malformed indices from english file path
#     ),
#     (
#         "data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.en", #original engish file path 
#         "data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.yo", #original language file path 
#         "data/Opus/Yoruba/clean_files/en-yo.txt(11)/NLLB.en-yo.en", # destination english file path
#         "data/Opus/Yoruba/clean_files/en-yo.txt(11)/NLLB.en-yo.yo", # destination language file path
#         get_malformed_indices("data/Opus/Yoruba/en-yo.txt (11)/NLLB.en-yo.yo") # malformed indices from english file path
#     ),
  
# ]

# # Loop through all file sets and apply filtering
# for original_en, original_lang, clean_en, clean_lang,malformed_indices in file_pairs:
#     filter_files_by_indices(original_en, original_lang, clean_en, clean_lang, malformed_indices)


"""CREATION AND STORAGE OF CSV"""
para_for_preprocessing = [
    # (clean_en_file                      clean_lang_file                        language_name                         csvdestination)
    # ("data/Opus/Yoruba/clean_files/en-yo.txt(1)/XLEnt.en-yo.en","data/Opus/Yoruba/clean_files/en-yo.txt(1)/XLEnt.en-yo.yo","yoruba","data/Opus/Preprocessed_csv/Yoruba/XLENT_yo_en.csv"),
]

# for clean_en_file,clean_lang_file,lang_name,output_csv in para_for_preprocessing:

#     try:
#         convert_to_csv(clean_en_file,clean_lang_file,lang_name,output_csv,batch_size=100)
#     except Exception as e:
#         print(f"{e}")


