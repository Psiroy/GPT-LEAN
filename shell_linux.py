import tensorflow as tf
from tokenizers import Tokenizer
from transformers import TFGPT2LMHeadModel

# File Configuration
tokenizer_file="lean_tokenizer.json"
model_dir="config"

def generate_text(model, tokenizer, input_text, max_length=48):
    input_tokens = tokenizer.encode(input_text).ids
    input_tokens = tf.convert_to_tensor([input_tokens], dtype=tf.int32)

    generated_text = []
    for _ in range(max_length):
        next_token = model.generate(
            input_tokens,
            max_length=input_tokens.shape[-1] + 1,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            eos_token_id=tokenizer.token_to_id("</s>")
        )
        next_token = next_token.numpy()[0][-1]
        if tokenizer.id_to_token(next_token) in ["</s>", "<pad>"]:
            break
        generated_text.append(tokenizer.decode([next_token]))
        input_tokens = tf.concat([input_tokens, tf.expand_dims([next_token], 0)], axis=-1)

    generated_text = "".join(generated_text).replace("⛀", " ").replace("⏎", "\n").replace("⇥", "\t").replace("␚", "\x1c")
    return generated_text


def main():
    print("GPT-LEAN Theorem Prover")
    print("Welcome to the GPT-LEAN! Please enter a text prompt to start generating text.")

    tokenizer = Tokenizer.from_file(tokenizer_file)
    model = TFGPT2LMHeadModel.from_pretrained("./"+model_dir)

    while True:
        input_text = input("Enter a text prompt:\n")
        if input_text.lower() == "quit":
            break

        generated_text = generate_text(model, tokenizer, input_text)
        print("\nGenerated text:\n")
        print(generated_text)

        while True:
            user_input = input(
                "\nPress Enter to continue generating text, type 'quit' to exit, or type anything else to start with a new text prompt: \n")

            if user_input.lower() == "quit":
                break
            elif user_input == "":
                input_text += generated_text
                generated_text = generate_text(model, tokenizer, input_text)
                print("\nGenerated text:\n")
                print(generated_text)
            else:
                input_text = user_input
                generated_text = generate_text(model, tokenizer, input_text)
                print("\nGenerated text:\n")
                print(generated_text)

if __name__ == "__main__":
    main()
