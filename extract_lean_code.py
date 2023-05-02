import os

root_dir = 'mathlib/src'
output_file = 'lean_code.txt'

def process_text(text):
    text = text.replace(" ", "⛀")
    text = text.replace("\n", "⏎")
    text = text.replace("\t", "⇥")
    text = text.replace("\x1c", "␚")
    return text

with open(output_file, 'w', encoding='utf-8') as outfile:
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.lean'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\x1c\n")