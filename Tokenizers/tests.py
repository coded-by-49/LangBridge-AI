from Byte_level_BPE import RegrexBpeTokenizer
import os
import pickle

# allowed special tokens 
special_tokens = {
    "<|startoftext|>": 100256,
    "<|endoftext|>": 100257,
    # "<|fim_prefix|>": 100258,
    # "<|fim_middle|>": 100259,
    # "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276
} 
tokenizer = RegrexBpeTokenizer(276)
tokenizer.vocab_size = 286
tokenizer.special_tokens = special_tokens


"""Testing"""
# Test for overall tokenization ability 
    # criterias :
        # encode/decode
        # seeing how merging takes place in multiple languages
def tokenizer_test(text_file_path):
    with open(text_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\n here are the merges \n")
    print(tokenizer.train(content)[:100])

    print(f"\n{tokenizer.decode(tokenizer.special_encode(content,"all")) == content}")

    list_item = list(tokenizer.merges.items())
    print(f"\n{len(tokenizer.vocab)}")
    

# tokenizer_test("data/Opus/Yoruba/clean_files/en-yo.txt(4)/wikimedia.en-yo.yo")
# Test for special token detection
with open("/home/fabia/LangBridge_MVP/data/oversampled_merged/all_in_one_corpus.pkl","rb") as f:
    items = pickle.load(f)

print(items[:10])