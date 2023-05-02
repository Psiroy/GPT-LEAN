import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import tensorflow as tf
from tokenizers import Tokenizer
from transformers import TFGPT2LMHeadModel
import threading
import os
import pyperclip

# Load the model and tokenizer
tokenizer = Tokenizer.from_file("lean_tokenizer.json")
model = TFGPT2LMHeadModel.from_pretrained("./config")


def process_text(text):
    return text.replace(" ", "⛀").replace("\n", "⏎").replace("\t", "⇥").replace("\x1c", "␚")


def decode_text(text):
    return text.replace("⛀", " ").replace("⏎", "\n").replace("⇥", "\t").replace("␚", "\x1c")


def prepare_input_text(input_text, input_mode):
    if input_mode == "Comment":
        if "\n" in input_text:
            input_text = "/--\n" + input_text + "\n-/\n"
        else:
            input_text = "/-- " + input_text + " -/\n"
    elif input_mode == "Head":
        input_text = "/-\nCopyright (c) 2023 Unknown. All rights reserved.\nReleased under Apache 2.0 license as described in the file LICENSE.\nAuthors: Unknown\n-/" + input_text
    return input_text


def clear_output():
    output_box.delete("1.0", "end")


def copy_output():
    pyperclip.copy(output_box.get("1.0", "end-1c"))


stop_generation = threading.Event()
output_box_lock = threading.Lock()

def generate_text():
    while True:
        with output_box_lock:
            input_text = prepare_input_text(input_box.get("1.0", "end-1c"), input_mode.get())
            output_text = output_box.get("1.0", "end-1c")
        combined_text = process_text(input_text + output_text)
        if combined_text == "":
            combined_text = process_text("\x1c\n")
        input_tokens = tokenizer.encode(combined_text).ids
        input_tokens = tf.convert_to_tensor([input_tokens], dtype=tf.int32)
        if stop_generation.is_set():
            break
        next_token = model.generate(
            input_tokens,
            max_length=input_tokens.shape[-1] + 1,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=1,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>")
        )
        next_token = next_token.numpy()[0][-1]
        if tokenizer.id_to_token(next_token) in ["</s>", "<pad>"]:
            break
        generated_word = tokenizer.decode([next_token], skip_special_tokens=False)
        decoded_word = decode_text(generated_word)
        output_box.insert("end", decoded_word)
        output_box.see("end")
        root.update()

        # Check for consecutive newlines or the document separator
        if decoded_word.endswith("\n\n") or decoded_word.endswith("\x1c"):
            break


def generate_thread():
    thinking_label.grid()
    root.update()
    generate_text()
    thinking_label.grid_remove()
    generate_button.config(text="Generate")
    if stop_generation.is_set():
        stop_generation.clear()


def toggle_generate_stop():
    if generate_button["text"] == "Generate":
        generate_button.config(text="Stop")
        stop_generation.clear()
        thread = threading.Thread(target=generate_thread)
        thread.start()
    else:
        generate_button.config(text="Generate")
        stop_generation.set()


root = tk.Tk()

style = ttk.Style()
style.configure("TButton", font=('Arial', 12))

root.title("GPT-LEAN Theorem Prover")
root.iconbitmap("gpt_lean_icon")

frame = tk.Frame(root, bg='white')
frame.pack(fill=tk.BOTH, expand=True)

input_label = ttk.Label(frame, text="Please enter a prompt:", font=("Arial", 12), background="white")
input_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=(5, 0))

input_box = tk.Text(frame, wrap=tk.WORD, height=10, font=('Consolas', 12), borderwidth=1, relief='solid')
input_box.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

type_label = ttk.Label(frame, text="Type:", font=("Arial", 12), background="white")
type_label.grid(row=0, column=1, sticky=tk.E, padx=(0, 5), pady=(5, 0))

input_mode = tk.StringVar(value="None")

type_menu = ttk.OptionMenu(frame, input_mode, "None", "None", "Comment", "Head")
type_menu.grid(row=0, column=2, sticky=tk.W, pady=(5, 0))

separator = ttk.Separator(frame, orient=tk.HORIZONTAL)
separator.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

output_label = ttk.Label(frame, text="GPT-LEAN:", font=("Arial", 12), background="white")
output_label.grid(row=3, column=0, padx=5, pady=(5, 0), sticky=(tk.W, tk.N))

thinking_label = ttk.Label(frame, text="Thinking...", font=("Arial", 12, 'bold'), foreground='green',
                           background="white")
thinking_label.grid(row=3, column=1, columnspan=2, padx=(0, 5), pady=(5, 0), sticky=(tk.E, tk.N))
thinking_label.grid_remove()

output_box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=10, font=("Consolas", 12))
output_box.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

generate_button = ttk.Button(frame, text="Generate", command=toggle_generate_stop)
generate_button.grid(row=5, column=0, pady=5, sticky=(tk.E, tk.N))
generate_button.config(style="TButton", width=10)


clear_button = ttk.Button(frame, text="Clear", command=clear_output)
clear_button.grid(row=5, column=0, pady=5, sticky=(tk.W, tk.N))
clear_button.config(style="TButton", width=10)

copy_button = ttk.Button(frame, text="Copy", command=copy_output)
copy_button.grid(row=5, column=2, pady=5, sticky=(tk.E, tk.N))
copy_button.config(style="TButton", width=10)


frame.rowconfigure(4, weight=1)
frame.columnconfigure(0, weight=1)

root.mainloop()
