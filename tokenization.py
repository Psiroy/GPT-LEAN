import os
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders

code_file="lean_code.txt"
output_tokenizer="lean_tokenizer.json"
vocab_size=30000


processed_code_file="processed_"+code_file

def process_text(text):
    text = text.replace(" ", "⛀")
    text = text.replace("\n", "⏎")
    text = text.replace("\t", "⇥")
    text = text.replace("\x1c", "␚")
    return text

with open(code_file, "r") as f:
    raw_text = f.read()
processed_text = process_text(raw_text)

with open(processed_code_file, "w") as f:
    f.write(processed_text)


pre_tokenizer = pre_tokenizers.Whitespace()
model = models.BPE()
decoder = decoders.BPEDecoder()

tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.decoder = decoder

trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2)
tokenizer.train(files=[processed_code_file], trainer=trainer)
tokenizer.save(output_tokenizer)
