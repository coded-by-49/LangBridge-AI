import pandas as pd
from itertools import zip_longest
import os 

def precess_large_files(hausa_file,english_file,output_csv, chunk_size = 10000):
    if not (os.path.exists(hausa_file) and os.path.exists(english_file)):
        raise FileNotFoundError(f"one of the files is non-existant")
    
    # Initialize CSV file with headers
    # mode='w' creates a new file or overwrites existing
    pd.DataFrame(columns=["hausa", "english"]).to_csv(output_csv, index=False)

    with open(hausa_file, "r", encoding="utf-8") as ha_f, open(english_file, "r", encoding="utf-8") as en_f:
        chunk = []
        for i, (hausa_line, eng_line) in enumerate(zip_longest(ha_f, en_f, fillvalue=None)):
            # Skip empty or None lines
            if hausa_line and eng_line and hausa_line.strip() and eng_line.strip():
                chunk.append({"hausa": hausa_line.strip(), "english": eng_line.strip()})
            
            # When chunk is full, append to CSV
            if len(chunk) >= chunk_size:
                df_chunk = pd.DataFrame(chunk)
                df_chunk.to_csv(output_csv, mode="a", header=False, index=False)  # Append without headers
                print(f"Processed {i + 1} lines")
                chunk = []  # Clear chunk for next batch
        
        # Save any remaining lines
        if chunk:
            df_chunk = pd.DataFrame(chunk)
            df_chunk.to_csv(output_csv, mode="a", header=False, index=False)
            print(f"Processed {i + 1} lines (final chunk)")
    
    # Verify final CSV
    df = pd.read_csv(output_csv)
    print(f"Saved {len(df)} Hausa-English pairs to {output_csv}")
    print(df.head())  # Preview first 5 rows
    