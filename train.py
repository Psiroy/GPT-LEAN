import tensorflow as tf
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from transformers import GPT2Config, TFGPT2LMHeadModel
import json
import os

# File configuration
train_file_path = "processed_train_code.txt"
val_file_path = "processed_val_code.txt"
tokenizer_file="lean_tokenizer.json"

output_name="gpt_lean"
output_dir="config"

loss_history_file="loss_history.json"
loss_history_plot="loss_history.png"


# Model configuration
tokenizer = Tokenizer.from_file(tokenizer_file)
config = GPT2Config(
    vocab_size=tokenizer.get_vocab_size(),
    bos_token_id=tokenizer.token_to_id("<s>"),
    eos_token_id=tokenizer.token_to_id("</s>"),
    pad_token_id=tokenizer.token_to_id("<pad>"),
    n_embd=768,
    n_layer=16,
    n_head=8,
    n_ctx=2048,
    n_positions=2048,
    dropout=0.3
)

# Training configuration
batch_size=32
epochs=10


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = None

    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


def plot_loss_history(train_loss_history, eval_loss_history, title, filename):
    iterations = range(1, len(train_loss_history) + 1)
    plt.plot(iterations, train_loss_history, 'b', label='Training Loss')
    plt.plot(iterations, eval_loss_history, 'r', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenized_text = tokenizer.encode(text).ids

    dataset = tf.data.Dataset.from_tensor_slices(tokenized_text)
    dataset = dataset.batch(block_size + 1, drop_remainder=True)

    def create_lm_inputs_and_targets(chunk):
        input_ids = chunk[:-1]
        lm_labels = chunk[1:]
        return input_ids, lm_labels

    dataset = dataset.map(create_lm_inputs_and_targets)
    return dataset


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        y_true, y_pred)
    return loss


def save_loss_history(loss_history_data, file_name=loss_history_file):
    with open(file_name, "w") as f:
        json.dump(loss_history_data, f)


def load_loss_history(file_name=loss_history_file):
    if os.path.isfile(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return {"train_loss_history": [], "eval_loss_history": []}


def train_model(model, train_dataset, eval_dataset, batch_size, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=loss_function)

    train_loss_history_callback = LossHistory()
    eval_loss_history_callback = LossHistory()

    checkpoint_filepath = "./"+output_name+"_checkpoints/checkpoint-{epoch:02d}-{batch:04d}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq=6000
    )

    initial_epoch = 0
    latest_checkpoint = tf.train.latest_checkpoint("./"+output_name+"_checkpoints")
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split("-")[-2])

    loss_history_data = load_loss_history()
    train_loss_history_callback.losses = loss_history_data["train_loss_history"]
    eval_loss_history_callback.losses = loss_history_data["eval_loss_history"]

    model.fit(
        train_dataset.batch(batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            train_loss_history_callback,
            eval_loss_history_callback,
            model_checkpoint_callback
        ],
        validation_data=eval_dataset.batch(batch_size)
    )

    return train_loss_history_callback, eval_loss_history_callback


model = TFGPT2LMHeadModel(config)

train_dataset = load_dataset(train_file_path, tokenizer)
eval_dataset = load_dataset(val_file_path, tokenizer)

train_loss_history_callback, eval_loss_history_callback = train_model(model, train_dataset, eval_dataset, batch_size=batch_size, epochs=epochs)

model.save_pretrained("./"+output_dir)
tokenizer.save(tokenizer_file)

plot_loss_history(train_loss_history_callback.losses, eval_loss_history_callback.losses, "Loss per Iteration", loss_history_plot)

loss_history_data = {
    "train_loss_history": train_loss_history_callback.losses,
    "eval_loss_history": eval_loss_history_callback.losses
}

save_loss_history(loss_history_data)
