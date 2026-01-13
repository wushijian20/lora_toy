import torch
import math
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. LOAD AND PROCESS DATASET
dataset = load_dataset(
    "text", 
    data_files="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    sample_by="document"
)
# print(dataset) 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    # 1. Tokenize the entire document
    tokenized = tokenizer(examples["text"])
    
    # 2. Flatten all input_ids into one long list
    # This ensures we don't just have one row with a list of lists
    concatenated_ids = [item for sublist in tokenized["input_ids"] for item in sublist]
    
    # 3. Cut the long list into blocks of 128
    block_size = 128
    total_length = len(concatenated_ids)
    # Floor to the nearest multiple of block_size
    total_length = (total_length // block_size) * block_size
    
    result = {
        "input_ids": [
            concatenated_ids[i : i + block_size] 
            for i in range(0, total_length, block_size)
        ],
        "attention_mask": [
            [1] * block_size 
            for _ in range(0, total_length, block_size)
        ]
    }
    return result

# We use batched=True but the function now returns multiple rows for each input row
tokenized_ds = dataset["train"].map(
    tokenize_fn, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

print(f"Total samples created: {len(tokenized_ds)}")

# Now splitting will work because len(tokenized_ds) will be ~2500-3000
splits = tokenized_ds.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
val_ds = splits["test"]


# Load the base GPT-2 model
base_model = GPT2LMHeadModel.from_pretrained("gpt2")

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,               # Rank: higher for complex Shakespearean grammar
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.1, 
    target_modules=["c_attn"] 
)

# Wrap the model with LoRA adapters
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()


# 3. TRAINING

training_args = TrainingArguments(
    output_dir="./shakespeare_gpt2_lora",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4, 
    num_train_epochs=3,    
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
    logging_steps=50,
    save_total_limit=1
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./shakespeare_lora_adapter")

# 4. METRICS & COMPARISON FUNCTIONS

def calculate_perplexity(eval_model, dataset, tokenizer, collator):
    eval_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = eval_model(**inputs) 
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return math.exp(avg_loss)

def compare_outputs(prompt, lora_model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with Shakespeare LoRA
    lora_model.base_model.enable_adapter_layers()
    ft_out = lora_model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8)
    
    # Generate with Raw GPT-2 (Disable adapter)
    with lora_model.disable_adapter():
        raw_out = lora_model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8)
    
    print(f"\nPROMPT: {prompt}")
    print("="*60)
    print(f"RAW GPT-2 OUTPUT:\n{tokenizer.decode(raw_out[0], skip_special_tokens=True)}")
    print("-" * 60)
    print(f"SHAKESPEARE LoRA OUTPUT:\n{tokenizer.decode(ft_out[0], skip_special_tokens=True)}")
    print("="*60)


# 5. EXECUTE EVALUATION
# Calculate Perplexity
lora_ppl = calculate_perplexity(model, val_ds, tokenizer, data_collator)
with model.disable_adapter():
    base_ppl = calculate_perplexity(model, val_ds, tokenizer, data_collator)

print(f"\nFinal Results:")
print(f"Base GPT-2 Perplexity: {base_ppl:.2f}")
print(f"Shakespeare LoRA Perplexity: {lora_ppl:.2f}")

# Side-by-Side Generation
compare_outputs("ROMEO: Shall I hear more,", model, tokenizer)
compare_outputs("KING HENRY: The day is", model, tokenizer)


# Access a specific layer's LoRA weights
# Layer 0 is: transformer.h[0].attn.c_attn
lora_a_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_A['default'].weight
lora_b_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_B['default'].weight

print("LoRA A Matrix Shape:", lora_a_weight.shape) # Should be [rank, input_dim]
print("LoRA A Sample Values:\n", lora_a_weight[:2, :2]) 

print("\nLoRA B Matrix Shape:", lora_b_weight.shape) # Should be [output_dim, rank]
print("LoRA B Sample Values:\n", lora_b_weight[:2, :2])