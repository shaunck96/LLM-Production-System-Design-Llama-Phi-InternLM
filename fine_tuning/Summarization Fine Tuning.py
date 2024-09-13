# Databricks notebook source
!pip install peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate einops


# COMMAND ----------

#!pip install datasets --upgrade 

# COMMAND ----------

from datasets import load_dataset

# Load and save dataset locally
dataset = load_dataset("prsdm/medquad-phi2-1k", split="train")
dataset.save_to_disk("/tmp/dataset")

# Upload dataset to DBFS
dbutils.fs.cp("file:///tmp/dataset", "dbfs:/FileStore/phi/dataset", recurse=True)


# COMMAND ----------

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = load_from_disk("/dbfs/FileStore/phi/dataset")

# Define model and tokenizer
base_model = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with quantization configuration
compute_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Wrap the model with DistributedDataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# Training arguments
training_args = TrainingArguments(
    output_dir="/dbfs/FileStore/llama/results",
    num_train_epochs = 1,
    fp16 = False,
    bf16 = False,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 1,
    gradient_checkpointing = True,
    max_grad_norm = 0.3,
    learning_rate = 2e-4,
    weight_decay = 0.001,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    max_steps = -1,
    warmup_ratio = 0.03,
    group_by_length = True,
    save_steps = 0,
    logging_steps = 25,
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length= None,
    tokenizer=tokenizer,
    args=training_args,
)

# Train model
trainer.train()

# Save the trained model
model_save_path = "/dbfs/FileStore/phi/trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")


# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir results/runs
# MAGIC

# COMMAND ----------

import logging
# Run text generation pipeline with our next model
# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

prompt = "What are the treatments for Gastrointestinal Carcinoid Tumors?"
pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_length=200, 
                device="cuda")
result = pipe(f"### Instruction: {prompt}")
print(result[0]['generated_text'])


# COMMAND ----------

# Clear the memory
del model, pipe, trainer
torch.cuda.empty_cache()


# Reload model and merge it with LoRA parameters
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

# Reload tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# COMMAND ----------

#!/usr/bin/env python3
import argparse
import os
import subprocess

def clone_repo(repo_url, clone_dir):
    if not os.path.isdir(clone_dir):
        print(f"Cloning repo from {repo_url} to {clone_dir}")
        subprocess.run(f"git clone {repo_url} {clone_dir}", shell=True, check=True)
    else:
        print(f"Repository already exists at {clone_dir}")

def quantize_model(model_dir, output_dir, base_name, gguf_version="v3"):
    llama_base = "/your/path/llama.cpp"  # Ensure this path is correct

    # Path to llama.cpp repository and conversion tools
    quantize_script = f"{llama_base}/quantize"
    convert_script = f"{llama_base}/convert.py"

    os.makedirs(output_dir, exist_ok=True)

    # Load model from DBFS
    model_path = f"/dbfs/FileStore/llama/guanaco_llama_finetune"
    
    # Define paths for FP16 GGUF and quantized GGUF
    fp16_path = f"{output_dir}/{base_name}.gguf{gguf_version}.fp16.bin"
    if not os.path.isfile(fp16_path):
        print(f"Converting model to FP16 GGUF format at {fp16_path}")
        subprocess.run(f"python3 {convert_script} {model_path} --outtype f16 --outfile {fp16_path}", shell=True, check=True)
    else:
        print(f"FP16 GGUF model already exists at {fp16_path}")

    # Quantize the FP16 GGUF model
    quantization_types = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
    for qtype in quantization_types:
        quantized_path = f"{output_dir}/{base_name}.gguf{gguf_version}.{qtype}.bin"
        print(f"Quantizing to {qtype} format at {quantized_path}")
        subprocess.run(f"{quantize_script} {fp16_path} {quantized_path} {qtype}", shell=True, check=True)
    
    # Cleanup FP16 GGUF file
    os.remove(fp16_path)
    print("Quantization complete.")

def main(model_dir, outbase, outdir):
    # URL to llama.cpp repository
    repo_url = "https://github.com/ggerganov/llama.cpp.git"
    llama_base = "/your/path/llama.cpp"  # Ensure this path is correct

    # Clone the repository
    clone_repo(repo_url, llama_base)

    # Quantize the model
    quantize_model(model_dir, outdir, outbase)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize a model and save in GGUF format.')
    parser.add_argument('model_dir', help='Path to the fine-tuned model directory')
    parser.add_argument('outbase', help='Output base name for the quantized files')
    parser.add_argument('outdir', help='Directory to save the quantized GGUF files')
    args = parser.parse_args()
    
    main(args.model_dir, args.outbase, args.outdir)


# COMMAND ----------

# MAGIC %md
# MAGIC Fine Tune and Quantize Approach 2

# COMMAND ----------

!pip install "peft>=0.4.0" "accelerate>=0.27.2" "bitsandbytes>=0.41.3" "trl>=0.4.7" "safetensors>=0.3.1" "tiktoken"
!pip install "torch>=2.1.1" -U
!pip install "datasets" -U
!pip install -q -U git+https://github.com/huggingface/transformers.gitpyt


# COMMAND ----------

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from pytrl import SFTTrainer

# COMMAND ----------

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
dolly_15K= "databricks/databricks-dolly-15k"

# Fine-tuned model
new_model = "llama-2-7b-chat-dolly"

# Download the dataset
dataset = load_dataset(dolly_15K, split="train")

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

def create_prompt(row):
    prompt = f"Instruction: {row['instruction']}\\nContext: {row['context']}\\nResponse: {row['response']}"
    return prompt

dataset['text'] = dataset.apply(create_prompt, axis=1)
data = Dataset.from_pandas(dataset)


# COMMAND ----------

compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
   
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(r=32,
                        lora_alpha=64,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                      )

# Define the training arguments. For full list of arguments, check
#<https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>
args = TrainingArguments(
    output_dir='llama-dolly-7b',
    warmup_steps=1,
    num_train_epochs=10, # adjust based on the data size
    per_device_train_batch_size=2, # use 4 if you have more GPU RAM
    save_strategy="epoch", #steps
    logging_steps=50,
    optim="paged_adamw_32bit",
    learning_rate=2.5e-5,
    fp16=True,
    seed=42,
    max_steps =500,
    save_steps=50,  # Save checkpoints every 50 steps
    do_eval=False,   
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field = 'text',
    max_seq_length=None,
    tokenizer=tokenizer,
    args=args,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)



# COMMAND ----------

prompt = " "

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = new_model.generate(input_ids=input_ids,
                         max_new_tokens=200,
                         do_sample=True,
                         top_p=0.9,
                         temperature=0.1)
result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
print(result)


# COMMAND ----------

!huggingface-cli login
hf_model_repo = "<REPO PATH>"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)

# COMMAND ----------

