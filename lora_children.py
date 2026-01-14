# imports the pytorch library for tensors and GPU support.
import torch

# imports Python's math utilites
import math

# imports OS utilities
import os

# Imports the Hugging Face helper to load datasets from various sources.
from datasets import load_dataset

# Imports PyTorch's DataLoader
from torch.utils.data import DataLoader

# Imports Hugging Face Transformers classes
from transformers import (
    GPT2Tokenizer,              # tokenization for GPT-2
    GPT2LMHeadModel,            # GPT-2 model with language-model head
    TrainingArguments,          # config container for training
    Trainer,                    # higher-level training loop
    DataCollatorForLanguageModeling # collates batches for causal LM training
)

# Imports PEFT (Parameter-Efficient Fine-Tuning) utilities for LoRA:
# LoraConfig to configure LoRA adapters.
# get_peft_model to wrap base model with LoRA.
# TaskType enum specifying task (CAUSAL_LM here).
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. DATA PREPARATION (Internal Dataset)
# ==========================================
# Create a more substantial local dataset to ensure the model learns the tone
# toddler_sentences = [
#     "me want juice now", "mama look big truck", "no go bed", "doggy go woof woof",
#     "bobby want cookie", "it mine mine mine", "where dada go", "me tired now",
#     "look at dat", "no like peas", "want more milk", "me do it", "big ball roll",
#     "mama hold me", "kitty cat soft", "vroom vroom car", "no wear shoes",
#     "me thirsty mama", "up up up", "bye bye birdy", "me hungry now", "The cat is big",
#     "I like milk", "The dog is happy", "Mommy reads a book", "no want juice", "mama look doggy",
#     "me go park now", "big truck vroom vroom", "bobby tired", "want cookie please", "where dada go", "it mine"
# ] * 20  # Repeat sentences to give the model more steps to learn

# with open("toddler_data.txt", "w") as f:
#     for s in toddler_sentences:
#         f.write(s + "\n")

# dataset = load_dataset("text", data_files="toddler_data.txt")["train"]

from datasets import Dataset
from datasets import load_from_disk

# Loads a dataset saved at "child_dialogue_dataset" from disk and 
# selects the first 2000 examples. The dataset is assumed to have a "text" column.
dataset = load_from_disk("child_dialogue_dataset")#.select(range(2000))

# dataset = dataset['train']

# Loads the pretrained GPT-2 tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sets the pad token to the tokenizer's end-of-sequence token, 
# required because GPT-2 doesn't have a pad token by default.
tokenizer.pad_token = tokenizer.eos_token


# Defines a function to preprocess examples:
# Lowercases and strips text (matching toddler style).
# Tokenizes with truncation, fixed max length 32, and pads to that length.
def tokenize_fn(examples):
    # Toddler speech is lowercase and lacks complex punctuation
    texts = [t.lower().strip() for t in examples["text"]]
    return tokenizer(texts, truncation=True, max_length=32, padding="max_length")

# Applies tokenize_fn across the dataset in batches and removes the original "text" 
# column, leaving tokenized fields like input_ids, attention_mask.
tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Splits the tokenized dataset into train/test with 10% for evaluation.
splits = tokenized_ds.train_test_split(test_size=0.1, seed=42)
print(splits)


# ==========================================
# 2. MODEL SETUP WITH LoRA
# ==========================================
# Loads pretrained GPT-2 (with LM head) as the base model for fine-tuning.
base_model = GPT2LMHeadModel.from_pretrained("gpt2")


# Creates a LoRA configuration:
# task_type=CAUSAL_LM indicates autoregressive LM task.
# r=32: LoRA rank (size of low-rank matrices).
# lora_alpha=64: scaling factor for LoRA updates.
# lora_dropout=0.1: dropout on LoRA layers.
# target_modules=["c_attn"]: restricts LoRA to attention projection module(s) named "c_attn".
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=32,               # High rank to force "unlearning" of standard grammar
    lora_alpha=64,      # Higher alpha for stronger adapter influence
    lora_dropout=0.1, 
    target_modules=["c_attn"] 
)

# Wraps the base GPT-2 model with LoRA adapters as specified 
# in peft_config. model is now a PEFT-enabled model object.
model = get_peft_model(base_model, peft_config)

# ==========================================
# 3. TRAINING
# ==========================================

# Configures training:
# output_dir: where to save checkpoints.
# eval_strategy="epoch": run evaluation at each epoch.
# learning_rate=1e-3: training LR (relatively large).
# per_device_train_batch_size=8: batch size per GPU/CPU.
# num_train_epochs=10: number of epochs.
# fp16=torch.cuda.is_available(): use mixed precision if CUDA available.
# logging_steps=10: log every 10 steps.

training_args = TrainingArguments(
    output_dir="./toddler_gpt2_lora",
    eval_strategy="epoch",
    learning_rate=1e-3, 
    per_device_train_batch_size=8, 
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    logging_steps=10
)

# Instantiates a Hugging Face Trainer with:
# the PEFT model,
# training args,
# train and eval datasets,
# a data collator that prepares batches for causal LM (mlm=False).

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("\n--- Training the Toddler Brain ---")
trainer.train()

model.save_pretrained("./children_lora_adapter")
tokenizer.save_pretrained("./children_lora_adapter")

# ==========================================
# 4. SIDE-BY-SIDE COMPARISON
# ==========================================

def compare_tones(prompt, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Pre-process prompt to be lowercase for the toddler
    inputs = tokenizer(prompt.lower(), return_tensors="pt").to(device)
    
    # 1. Generate with Toddler LoRA
    model.base_model.enable_adapter_layers()
    toddler_out = model.generate(
        **inputs, 
        max_new_tokens=20, 
        do_sample=True, 
        temperature=1.2,
        repetition_penalty=1.2
    )
    
    # 2. Generate with Raw GPT-2
    with model.disable_adapter():
        raw_out = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=True, 
            temperature=0.8 # Lower temp for the "smarter" base model
        )
    
    print(f"\nPROMPT: {prompt}")
    print("-" * 50)
    print(f"BASE GPT-2: {tokenizer.decode(raw_out[0], skip_special_tokens=True)}")
    print(f"KID TONE:   {tokenizer.decode(toddler_out[0], skip_special_tokens=True)}")
    print("-" * 50)

# Run comparisons
compare_tones("What would you like to eat?", model, tokenizer)
compare_tones("Where is your father?", model, tokenizer)
compare_tones("Look at that airplane in the sky!", model, tokenizer) 


# Access a specific layer's LoRA weights
# Layer 0 is: transformer.h[0].attn.c_attn
lora_a_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_A['default'].weight
lora_b_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_B['default'].weight

print("LoRA A Matrix Shape:", lora_a_weight.shape) # Should be [rank, input_dim]
print("LoRA A Sample Values:\n", lora_a_weight[:2, :2]) 

print("\nLoRA B Matrix Shape:", lora_b_weight.shape) # Should be [output_dim, rank]
print("LoRA B Sample Values:\n", lora_b_weight[:2, :2])