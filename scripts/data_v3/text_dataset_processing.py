import os 
import json
from datasets import load_dataset

def create_parallel_corpus(eng_file_path, igbo_file_path, outputfile):
    if os.path.exists(eng_file_path) and os.path.exists(igbo_file_path):
        print(f'loading files from {eng_file_path} and {igbo_file_path}....')
        with open(eng_file_path, "r", encoding = "utf-8") as f_en, \
            open(igbo_file_path, "r", encoding = "utf-8") as f_ig,\
            open(outputfile, "w", encoding="utf-8") as ig_en:
            id = 0
            for ig_line, en_line in zip(f_ig,f_en):
                id += 1
                ig_en.write(
                    json.dumps(
                    {
                        "id": id,
                        "translation":{
                            "eng_Latn": en_line.strip(),
                            "ibo_Latn": ig_line.strip()
                        }
                    }
                    , ensure_ascii = False ) + "\n"
                )

        with open(outputfile, "r", encoding="utf-8") as ig_en:
                line_count = len(ig_en.readlines())
                print(f"Total lines : {line_count}")


    else:
        print("This file path is non-existent")

def extend_one_json_file_with_another(json_file_one, json_file_two):
    with open(json_file_one, "r", encoding="utf-8") as f1, \
        open (json_file_two, "a+", encoding = "utf-8") as f2 :
        f2.seek(0)
        length_f2 = len(f2.readlines())
        length_f1 = len(f1.readlines())
        f1.seek(0)
        linecount = 1

        for jsonline in f1:
            jsonline_data = json.loads(jsonline)
            jsonline_data["id"] = length_f2 + linecount
            linecount += 1

            f2.write(json.dumps(
                jsonline_data, ensure_ascii = False
            ) + "\n")
        
        f2.seek(0)
        print(f"Intial length of file1 --> {length_f1} \n \
                Iniitlal length of file2 --> {length_f2} \n \
                New length of file2 --> {len(f2.readlines())}")

        f2.seek(0)
        if len(f2.readlines()) == length_f2 + length_f1:
            print("successfully concatenated")
        else:
            print("unsuccessfully concatenation")


def remove_duplicate_jsonlines(json_file_one, json_file_two):
    ds_1 = load_dataset("json", data_files = json_file_one)
    seen = set(data_line["translation"]["eng_Latn"] for data_line in ds_1["train"])

    ds_2 = load_dataset("json", data_files = json_file_two)
    len_dup_ds_2 = len(ds_2["train"])

    no_dup_ds_2_gen = (data_line for data_line in ds_2["train"] if data_line["translation"]["eng_Latn"] not in seen)

    with open(json_file_two, "w", encoding = "utf-8") as f2:
        for dataline in no_dup_ds_2_gen :
            f2.write(json.dumps(dataline, ensure_ascii = False) + "\n")

    with open(json_file_two, "r", encoding="utf-8") as f2:
        len_no_dup_ds_2 = len(f2.readlines())

    print(f"There where {len_dup_ds_2-len_no_dup_ds_2} duplicate datalines found in {json_file_two}. These data lines where already in {json_file_one}")






# The calls below are the last one-off step that was run (building the test
# split from Ezeani val + MAFAND dev). They are wrapped in a __main__ guard so
# that importing this module does not rewrite data files. Paths are relative
# to the repository root and point at the local, git-ignored data folder.
# Note: mafand_dev_igbo_eng.jsonl was later renamed to igbo_eng_lv1_test.jsonl.
if __name__ == "__main__":
    remove_duplicate_jsonlines("langbridge_ai_v3_data/data/igbo_v3/test/mafand_dev_igbo_eng.jsonl","langbridge_ai_v3_data/data/igbo_v3/test/ezeani_val_igbo_eng.jsonl")
    print(f"\n\n")
    extend_one_json_file_with_another("langbridge_ai_v3_data/data/igbo_v3/test/ezeani_val_igbo_eng.jsonl","langbridge_ai_v3_data/data/igbo_v3/test/mafand_dev_igbo_eng.jsonl")
