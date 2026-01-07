import torch
import math
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. DATA PREPARATION (Internal Dataset)
# ==========================================
# Create a more substantial local dataset to ensure the model learns the tone
toddler_sentences = [
    "me want juice now", "mama look big truck", "no go bed", "doggy go woof woof",
    "bobby want cookie", "it mine mine mine", "where dada go", "me tired now",
    "look at dat", "no like peas", "want more milk", "me do it", "big ball roll",
    "mama hold me", "kitty cat soft", "vroom vroom car", "no wear shoes",
    "me thirsty mama", "up up up", "bye bye birdy", "me hungry now", "The cat is big",
    "I like milk", "The dog is happy", "Mommy reads a book", "no want juice", "mama look doggy",
    "me go park now", "big truck vroom vroom", "bobby tired", "want cookie please", "where dada go", "it mine"
] * 20  # Repeat sentences to give the model more steps to learn

with open("toddler_data.txt", "w") as f:
    for s in toddler_sentences:
        f.write(s + "\n")

dataset = load_dataset("text", data_files="toddler_data.txt")["train"]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    # Toddler speech is lowercase and lacks complex punctuation
    texts = [t.lower().strip() for t in examples["text"]]
    return tokenizer(texts, truncation=True, max_length=32, padding="max_length")

tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
splits = tokenized_ds.train_test_split(test_size=0.1, seed=42)

# ==========================================
# 2. MODEL SETUP WITH LoRA
# ==========================================
base_model = GPT2LMHeadModel.from_pretrained("gpt2")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=32,               # High rank to force "unlearning" of standard grammar
    lora_alpha=64,      # Higher alpha for stronger adapter influence
    lora_dropout=0.1, 
    target_modules=["c_attn"] 
)

model = get_peft_model(base_model, peft_config)

# ==========================================
# 3. TRAINING
# ==========================================
training_args = TrainingArguments(
    output_dir="./toddler_gpt2_lora",
    eval_strategy="epoch",
    learning_rate=1e-3, 
    per_device_train_batch_size=8, 
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("\n--- Training the Toddler Brain ---")
trainer.train()

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