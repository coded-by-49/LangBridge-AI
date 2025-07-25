import pandas as pd
import os
from itertools import zip_longest

def get_malformed_indices(file_path, threshold=0.002):
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

def filter_files_by_indices(input_en, input_ha, output_en, output_ha, malformed_indices):
    """Remove lines at specified indices from both files."""
    with open(input_en, "rb") as en_file, \
         open(input_ha, "rb") as ha_file, \
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


malformed_indices = get_malformed_indices("data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.en")

# defining file path 
original_en_file = "data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.en"
original_ha_file = "data/Opus/Hausa/en-ha.txt(2)/CCMatrix.en-ha.ha"

# creating file path for clean output file
output_directory = "data/Opus/Hausa/clean_files"
clean_en_file = os.path.join(output_directory, "CCMatrix_filtered.en")
clean_ha_file = os.path.join(output_directory, "CCMatrix_filtered.ha")

filter_files_by_indices(original_en_file, original_ha_file, clean_en_file, clean_ha_file, malformed_indices)
