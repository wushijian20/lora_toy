from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
import math

# 1. Load Dataset
dataset = load_dataset("suayptalha/Poetry-Foundation-Poems") 
dataset = dataset["train"].select(range(2000))
splits = dataset.train_test_split(test_size=0.1)

# 2. Tokenizer Setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    # Add EOS token to help the model learn the end of a poem
    texts = [text + tokenizer.eos_token for text in examples["Poem"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# Applies the tokenize_fn to every poem in the training and val set.
train_tokenized = splits["train"].map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
val_tokenized = splits["test"].map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# 3. Load Model and Apply LoRA
model = GPT2LMHeadModel.from_pretrained("gpt2")

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Tells the library we are doing standard text generation
    inference_mode=False, 
    r=8,                # Rank: Higher = more parameters, lower = more efficient
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.1, 
    target_modules=["c_attn"] # For GPT-2, this targets the attention layers
)

# Wrap the model with LoRA

model = get_peft_model(model, peft_config) # Wraps the model by freezes the original parameters and injects the tiny LoRA matrices that will stay "trainable".
model.print_trainable_parameters() # show how few params are being trained

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./poetry_gpt2_lora",
    eval_strategy="epoch", # Better to see progress
    learning_rate=2e-4,    # LoRA often handles slightly higher learning rates
    per_device_train_batch_size=4, 
    num_train_epochs=3,    # Increased slightly as LoRA is efficient
    weight_decay=0.01,
    fp16=True,             # Uses "half-precision" numbers to speed up training by up to 3x and significantly reduce GPU memory usage.
    logging_steps=10
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # This is a specialized tool that takes your tokenized batches and prepares the "labels." In Causal LM (like GPT-2), the label is just the input shifted by one word—this tool handles that automatically.
)

trainer.train() # Starts the train

# 6. Saving the Adapter
# Note: This saves ONLY the small LoRA weights (A and B), not the whole GPT-2
model.save_pretrained("./poetry_lora_adapter")

from peft import PeftModel

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
lora_model = PeftModel.from_pretrained(base_model, "./poetry_lora_adapter") # Injects the saved poetry style back into the base model.

prompt = "Whispers of the night"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = lora_model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


########################################################################################################

from torch.utils.data import DataLoader

def calculate_perplexity(model, dataset, tokenizer, data_collator):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    eval_dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
    
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move all tensors in the batch to the device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # The collator already put 'labels' in 'inputs'. 
            # Just unpack it!
            outputs = model(**inputs) 
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(eval_dataloader)
    return math.exp(avg_loss)


# Update your call to include the tokenizer and data_collator
from transformers import DataCollatorForLanguageModeling

# You must define this so the perplexity function knows how to pad the sequences
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Now the call will work
base_ppl = calculate_perplexity(base_model, val_tokenized, tokenizer, data_collator)
lora_ppl = calculate_perplexity(model, val_tokenized, tokenizer, data_collator)

print(f"Base GPT-2 Perplexity: {base_ppl:.2f}")
print(f"Fine-tuned LoRA Perplexity: {lora_ppl:.2f}")


##############################################################################################################


def compare_models(prompt, base, fine_tuned, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
    
    # Generate from Base
    base_out = base.generate(**inputs, max_length=60, do_sample=True, temperature=0.8)
    # Generate from Fine-tuned
    ft_out = fine_tuned.generate(**inputs, max_length=60, do_sample=True, temperature=0.8)
    
    print(f"PROMPT: {prompt}")
    print("-" * 30)
    print(f"BASE GPT-2:\n{tokenizer.decode(base_out[0], skip_special_tokens=True)}")
    print("-" * 30)
    print(f"FINE-TUNED POETRY:\n{tokenizer.decode(ft_out[0], skip_special_tokens=True)}")

compare_models("The silent moon", base_model, lora_model, tokenizer)


# Access a specific layer's LoRA weights
# Layer 0 is: transformer.h[0].attn.c_attn
lora_a_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_A['default'].weight
lora_b_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_B['default'].weight

print("LoRA A Matrix Shape:", lora_a_weight.shape) # Should be [rank, input_dim]
print("LoRA A Sample Values:\n", lora_a_weight[:2, :2]) 

print("\nLoRA B Matrix Shape:", lora_b_weight.shape) # Should be [output_dim, rank]
print("LoRA B Sample Values:\n", lora_b_weight[:2, :2])