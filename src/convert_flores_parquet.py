"""This function is was used to convert all parquet files from flores200 to csv"""
import pandas as pd
#loading and pairing of parquet files(flores 200)
def load_and_pair_parquet(lang_file, eng_file, lang_name, output_csv):
    
    #load eaach datafile 
    lang_df = pd.read_parquet(lang_file)
    eng_df = pd.read_parquet(eng_file)

    # confirm if the length matches its english translation
    assert len(lang_df) == len(eng_df), f"Error: {lang_file} ({len(lang_df)}) and {eng_file} ({len(eng_df)}) have different lengths!"

    print(f"{lang_name} columns:", lang_df.columns)
    print("\n")
    print("English columns:", eng_df.columns)

    # placing the language with its english translation side by side 
    df_lang_with_eng = pd.DataFrame({
        lang_name.lower(): lang_df["text"],
        "english": eng_df["text"]
    })
    

    # convert this counterpart into a csv
    df_lang_with_eng.to_csv(output_csv, index=False)
    
    print(f"Saved {len(df_lang_with_eng)} {lang_name}-English pairs to {output_csv}")


pairs = [
    # ("data/Flores200/Hausa/hau_latn(dev).parquet", "data/Flores200/English/eng_latn(dev).parquet", "hausa", "data/Flores200/ha_en_dev.csv"),
    # ("data/Flores200/Hausa/hau_latn(devtest).parquet", "data/Flores200/English/eng_latn(devtest).parquet", "hausa", "data/Flores200/ha_en_devtest.csv"),
    # ("data/Flores200/Igbo/ibo_latn(dev).parquet", "data/Flores200/English/eng_latn(dev).parquet", "igbo", "data/Flores200/ibo_en_dev.csv"),
    # ("data/Flores200/Igbo/ibo_latn(devtest).parquet", "data/Flores200/English/eng_latn(devtest).parquet", "igbo", "data/Flores200/ibo_en_devtest.csv"),
    # ("data/Flores200/Yoruba/yor_latn(dev).parquet", "data/Flores200/English/eng_latn(dev).parquet", "yoruba", "data/Flores200/yor_en_dev.csv"),
    ("data/Flores200/Yoruba/yor_latn(devtest).parquet", "data/Flores200/English/eng_latn(devtest).parquet", "yoruba", "data/Flores200/yor_en_devtest.csv")
]

for lang_file, eng_file, lang_name, output_csv in pairs:
    try: 
        load_and_pair_parquet(lang_file,eng_file,lang_name,output_csv)
    except Exception as e:
        print(f'An error occursed while processing {lang_file}: {e}')
