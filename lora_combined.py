import torch
import os
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET 1: SHAKESPEARE ---
def get_shakespeare_data():
    ds = load_dataset("text", data_files="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", sample_by="document")
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length", return_overflowing_tokens=True, stride=32)
    return ds["train"].map(tokenize, batched=True, remove_columns=["text"]).remove_columns(["overflow_to_sample_mapping"])

# --- DATASET 2: POETRY ---
# Using a sample of public domain poetry
def get_poetry_data():
    # For this script, we'll simulate it with a few lines, replace with your poetry file if available
    poetry_text = ["O mistress mine, where are you roaming?", "The woods are lovely, dark and deep", "Shall I compare thee to a summer's day?"] * 50
    with open("poetry.txt", "w") as f: f.write("\n".join(poetry_text))
    ds = load_dataset("text", data_files="poetry.txt")["train"]
    return ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=64, padding="max_length"), batched=True)

# --- DATASET 3: KID TONE ---
def get_kid_data():
    kid_text = ["me want juice", "mama look truck", "no bed now", "doggy woof", "me go park"] * 50
    with open("toddler.txt", "w") as f: f.write("\n".join(kid_text))
    ds = load_dataset("text", data_files="toddler.txt")["train"]
    return ds.map(lambda ex: tokenizer(ex["text"].lower(), truncation=True, max_length=32, padding="max_length"), batched=True)

# ==========================================
# 2. SEQUENTIAL TRAINING
# ==========================================
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

styles = {
    "shakespeare": {"data": get_shakespeare_data(), "r": 16, "epochs": 1},
    "poetry":      {"data": get_poetry_data(),      "r": 16, "epochs": 3},
    "toddler":     {"data": get_kid_data(),         "r": 32, "epochs": 5}
}

for name, config in styles.items():
    print(f"\n>>> TRAINING ADAPTER: {name.upper()}")
    
    # Define LoRA config for this specific style
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["r"],
        lora_alpha=32,
        target_modules=["c_attn"]
    )
    
    # Add/Re-initialize the adapter
    if name == "shakespeare":
        model = get_peft_model(base_model, lora_config, adapter_name=name)
    else:
        model.add_adapter(name, lora_config)
    
    model.set_adapter(name) # Ensure we are training the current one
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./{name}_adapter",
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=4,
            logging_steps=50,
            report_to="none"
        ),
        train_dataset=config["data"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    model.save_pretrained(f"./{name}_lora_adapter")

# ==========================================
# 3. COMPARISON INFERENCE
# ==========================================
def run_comparison(prompt):
    print(f"\nPROMPT: {prompt}\n" + "="*50)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 1. Base GPT-2
    with model.disable_adapter():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
        print(f"[BASE]: {tokenizer.decode(out[0], skip_special_tokens=True).strip()}")

    # 2. Styles
    for name in styles.keys():
        model.set_adapter(name)
        # Use higher temp for toddler to make it more "random"
        temp = 1.3 if name == "toddler" else 0.8
        out = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=temp)
        print(f"[{name.upper()}]: {tokenizer.decode(out[0], skip_special_tokens=True).strip()}")

# Test it out!
run_comparison("The morning sun is")
run_comparison("I want to go")