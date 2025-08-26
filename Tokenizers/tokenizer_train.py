from Byte_level_BPE import RegrexBpeTokenizer
import pickle
from pathlib import Path
import random
from collections import Counter

# Load the pickle file

file_path = Path("data/oversampled_merged/all_in_one_corpus.pkl")
all_in_one_corpus = []

with open(file_path,"rb") as f:
    all_in_one_corpus = pickle.load(f)


print(len(all_in_one_corpus))

"""Training in batches """


Tokenizier = RegrexBpeTokenizer(8000)
vocab,merge = Tokenizier(all_in_one_corpus,150000)

