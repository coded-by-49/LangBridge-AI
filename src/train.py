'''MODEL PREPROCESSING '''
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch 

# Set CUDA as default device
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print("Default device set to CUDA:", torch.cuda.get_device_name(0))
else:
    torch.set_default_device('cpu')
    print("CUDA not available, default device set to CPU")

df = pd.read_csv("data/Opus/Preprocessed_csv/Igbo/QED_ig_en.csv")

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

# configuration of trainer
trainer = BpeTrainer(vocab_size = 60000, special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"])
texts = df['igbo'].tolist() + df['english'].tolist()

# training tokenizer 
tokenizer.train_from_iterator(texts,trainer)
tokenizer.save("tokenizers/tokenizer_igbo_en.json")

# applying tokenizer to sentences of dataset
def tokenize_sentences(sentences, tokenizer, max_length = 128, device = "cuda"):
    # transformation of sentence to tokens with 
    encodings = [tokenizer.encode(f"<sos> {s} <eos>").ids for s  in sentences]

    # creation of empty tensor
    padded = torch.zeros(len(encodings), max_length, dtype=torch.long)

    # padding out our encodings
    for i, sen_tokens in enumerate(encodings):
        length = min(len(sen_tokens), max_length)
        padded[i, :length] = torch.tensor(sen_tokens[:length], dtype = torch.long)
    
    return padded 

source_tokens = tokenize_sentences(df['igbo'].tolist(), tokenizer)
target_tokens = tokenize_sentences(df["english"].tolist(), tokenizer)

# testing
print("Sample igbo tokens:", source_tokens[0])
print("Sample English tokens:", target_tokens[0])
print("Vocabulary size:", tokenizer.get_vocab_size())

#saveing tokenized datasets
torch.save(source_tokens, 'data/Tokenised_Data/Igbo/source_tokens_s1.pt')
torch.save(target_tokens, 'data/Tokenised_Data/Igbo/target_tokens_s1.pt')