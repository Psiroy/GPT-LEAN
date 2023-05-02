import os
directory = "mathlib/src"
train_file = "train_code.txt"
val_file = "val_code.txt"
train_ratio=0.95

def load_and_concatenate_files(directory):
    text = ""
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".lean"):
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    text += f.read() + "\n\x1c\n"
    return text


def split_text_on_files(text, train_ratio):
    files = text.split("\n\x1c\n")
    n_train = int(len(files) * train_ratio)
    train_text = "\n\x1c\n".join(files[:n_train])
    val_text = "\n\x1c\n".join(files[n_train:])
    return train_text, val_text


def write_to_files(train_text, val_text, train_file, val_file):
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(val_text)


text = load_and_concatenate_files(directory)
train_text, val_text = split_text_on_files(text, train_ratio)
write_to_files(train_text, val_text, train_file, val_file)
