# GPT-LEAN
A GPT-2 model trained to generate formal mathematical proofs in LEAN.

## How to train the model
This is a good tutorial to show how to train a GPT-2 model because its rather small dataset.

### Step 1 Prepare the environment
1. Make sure you have already installed Python. If not, please download and install it from https://www.python.org/downloads/ .  
Make sure the current directory is the one you want to use as the directory to put GPT-LEAN.
2. Copy and run the code `git clone https://github.com/Psiroy/GPT-LEAN.git` in your terminal. This will clone the GPT-LEAN project to your current directory.  
Run the code `cd GPT-LEAN` in your terminal. This will set your current directory to GPT-LEAN.
3. Run the code `git clone https://github.com/leanprover-community/mathlib.git` in your terminal. This will clone the Mathlib project to your current directory.
4. Run the code `pip install -r requirements_train.txt` in your terminal. This will install all the required libraries for the training.

### Step 2 Prepare the data
1. Run the code `python extract_lean_code.py`. This will put all the LEAN code file in *mathlib/src* to a file *lean_code.txt*. You can modify the range of files you want to use as the training data by edit the value of *root_dir* in *extract_lean_code.py* (for example, use 'root_dir="mathlib"').
2. Run the code `python split_data.py`. This will generate a dataset for training and a dataset for validation. If you have modify the *root_dir* in the file *extract_lean_code.py*, you should edit the value of *directory* in *split_data.py* as well.
3. Run the code `python process_data.py`. This will translate the data to the form what the model will use as an input.

### Step 3 Tokenization
Run the code `python tokenization.py`. This will do the BPE tokenization to the data *lean_code.txt* (also generate a translated data *processed_lean_code.txt*). The tokenizer will be saved as *lean_tokenizer.json*.

### Step 4 Train the model
Run the code `python train.py`. This will take hours to train the model and the output will be save in the directory *config*.  
During the training, it will save some checkpoints in the directory *gpt_lean_checkpoints*. You can edit the frequency of checkpoints by changing the value of *save_freq* in *train.py*. If the training is interrupted and you run the *train.py* again, you will start from the latest checkpoint. If you have finished training, you should remove the *gpt_lean_checkpoints* directory to release the space.   
At the end of training, it will also save the training loss history in the file *loss_history.json* and plot a figure of it in *loss_history.png*.

## How to run the GPT-LEAN theorem prover
Make sure you have trained a GPT-LEAN model or [download a pre-trained GPT-LEAN model](https://www.dropbox.com/scl/fo/tpo0hv5p2rsiiudjjgr2o/h?dl=0&rlkey=uk7jxx5hlz2csxhger0imtdgg).  
We assume that the model is placed in the *config* directory.  
Check *lean_tokenizer.json* is in your directory.  
If you want to run a demo of the model in terminal, you can run the code `python shell_linux.py`.  
If you want use the GUI of GPT-LEAN in Windows, you should run `python shell_windows.py`. (Make sure you have installed all the required library in *requirements_run.txt*. The icon file *gpt_lean_icon.ico* should be in your directory before you run the GUI.)
