import os

train_file="train_code.txt"
val_file="val_code.txt"

processed_train_file="processed_train_code.txt"
processed_val_file="processed_val_code.txt"

def process_text(text):
    text = text.replace(" ", "⛀")
    text = text.replace("\n", "⏎")
    text = text.replace("\t", "⇥")
    text = text.replace("\x1c", "␚")
    return text


with open(train_file, "r") as f:
    raw_text = f.read()
processed_text = process_text(raw_text)

with open(processed_train_file, "w") as f:
    f.write(processed_text)

with open(val_file, "r") as f:
    raw_text = f.read()
processed_text = process_text(raw_text)

with open(processed_val_file, "w") as f:
    f.write(processed_text)