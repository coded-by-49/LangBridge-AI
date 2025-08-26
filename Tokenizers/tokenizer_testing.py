from batched_bpe import Batched_size_BpeTokenizer
import os
import pickle
from collections import Counter
import gc

gc.collect()
# allowed special tokens 
special_tokens = {
    "<|pad|>": 256,
    "<|unk|>": 257,
    "<|startoftext|>": 258,
    "<|endoftext|>": 259,
}
# 650 is the sweetspot
tokenizer = Batched_size_BpeTokenizer(vocab_size=50000,stop_list_size=None)

# tokenizer.register_special_tokens(special_tokens)
# print(f"Special tokens are : {tokenizer.special_tokens}\n the inversed versions are {tokenizer.inverse_special_tokens}")


"""Testing"""
# Test for overall tokenization ability 
    # criterias :
        # encode/decode
        # seeing how merging takes place in multiple languages

with open ("data/oversampled_merged/all_in_one_corpus.pkl","rb") as f:
        data = pickle.load(f)
        
txt_list2 = list(dict.fromkeys(data))
print(f"The length of lists without duplicates is {len(txt_list2)}")

# print(tokenizer.texts_to_ids_freq_table(txt_list,counts=None))
tranformer_file = "Tokenizers/Vocabularies/main_transformer_vocab.json"
human_readble_file = "Tokenizers/Vocabularies/main_vocab.vocab"

def test():
    tokenizer.register_special_tokens(special_tokens)

    tokenizer.train(txt_list2,max_batch_size=0,cap_divisor=2,verbose="Test")

    tokenizer.save(human_readble_file,tranformer_file)

test()

