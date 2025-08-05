from tokenizers import Tokenizer
loaded_tokenizer = Tokenizer.from_file("Tokenizers/tokenizer_igbo_en.json")

print(loaded_tokenizer.get_vocab().get("<unk>", "Not found"))