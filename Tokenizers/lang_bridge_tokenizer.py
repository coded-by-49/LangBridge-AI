from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
# tokenizer decalaratoin
tokenizer = Tokenizer(models.BPE())

#
tokenizer.pre_tokenizer = pre_tokenizers.Bytelevel(add_prefix_space = False)

trainer = trainers.BpeTrainer(vocab_size = 25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# test preprocessor tokenizer 

# post processing 
tokenizer.post_processor = processors.Bytelevel(trim_offsets  = False)
tokenizer.decoder = decoders.Byteleve()

# test decoding abilities 