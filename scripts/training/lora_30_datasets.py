import os
import math
import torch
import gc
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration & Dataset List
MODEL_ID = "gpt2"
# 30+ Hugging Face dataset paths
DATASET_PATHS = [
    # "deven367/babylm-100M-children-stories"
    # ,
    # "gofilipa/bedtime_stories"
    # ,
    # "sarnab/Shakespeare_Corpus"
    # ,
    # "SzuTao/KingJamesVersionBible"    # Need to remove other columns.
    # ,
    # "azizsi/old_english_dataset"        # Need to remove other columns.
    # ,
    # "contemmcm/victorian_authorship"  # Need to remove other columns.
    # ,
    # "erhwenkuo/poetry-chinese-zhtw"
    # ,
    #  "aslicu/fairy_tales"
     # ,
    # "AJ69/Mythological"  
    # ,
    # "Smilyai-labs/ChatPILE-Casual"
    # ,
    # "phxdev/corporate-speak-dataset"
    # ,
    # "sumukshashidhar-testing/research-paper-abstracts"
    # ,
    # "Samarth0710/neurips-2024-peer-reviews"
    # ,
    # "nvidia/Nemotron-Math-Proofs-v1"             
    # ,
    # "emilpartow/reddit_finance_posts_sp500"
    # ,
    # "Thewillonline/reddit-sarcasm"
    # ,
    # "wenknow/reddit_dataset_44" # too large
    # ,  
    # "agentlans/reddit-logic"
    # , 
    # "cowWhySo/reddit_top_comments"
    # , # too large
    # "jonaskoenig/reddit-blogspot-twitter"
    # ,
    # "Osondu/reddit_autism_dataset"
    # ,
    # "Tlighteval/covid_dialogue"
    # ,
    # "sedthh/tv_dialogue"
    # , 
    # "Nexdata/American_English_Natural_Dialogue_Speech_Data"
    # ,
    # "erhwenkuo/medical_dialogue-chinese-zhtw"
    # ,
    # "jpeandrew/dialy_dialogue_with_recoginized_concept_raw",
    # "rony/soccer-dialogues"
    # ,
    # "pixelsandpointers/empathetic_dialogues_for_lm",
    # "Adapting/empathetic_dialogues_v2",
    # "suayptalha/Poetry-Foundation-Poems"
    # ... add all 30 datasets here
]

# 2. Shared Setup
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    # This handles datasets with different column names (Poem, text, content, etc.)
    possible_cols = ["Poem", "text", "Text", "TEXT", "content", "instruction", 
                     "body", "stories", "Output", "output", "chunk", 
                     "messages", "document_text", "reviews"]
    text_col = next((col for col in possible_cols if col in examples), None)
    
    if text_col is None:
        raise ValueError(f"Could not find a text column. Available: {examples.keys()}")
        
    texts = [str(t) + tokenizer.eos_token for t in examples[text_col]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# 3. LoRA Configuration (Shared)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)

# 4. The Loop
results_log = []

for ds_path in DATASET_PATHS:
    short_name = ds_path.split("/")[-1]
    print(f"\n{'='*20}\nTRAINING ADAPTER: {short_name}\n{'='*20}")
    
    try:
        # Load & Prep Data
        dataset = load_dataset(ds_path, split="train")

        # Select 10,000 or the maximum available rows
        limit = min(len(dataset), 10000)
        dataset = dataset.select(range(limit)) # Limiting for speed

        # Preprocess "SzuTao/KingJamesVersionBible" to remove all other columns expect the 'Text'
        # dataset = dataset.remove_columns(['Book ID', 'Book', 'Book Abbeviation', 'Chapter Number', 'Verse Number', 'Character Count'])
        
        # Preprocess "azizsi/old_english_dataset"
        # dataset = dataset.remove_columns(['Instruction', 'Input'])
        
        # Preprocess "contemmcm/victorian_authorship"
        # dataset = dataset.remove_columns(['author'])

        # Preprocess "aslicu/fairy_tales"
        # dataset = dataset.remove_columns(['source'])

        #"AJ69/Mythological" # ['book', 'canto', 'chapter', 'verse', 'instruction', 'input', 'output']
        # dataset = dataset.remove_columns(['book', 'canto', 'chapter', 'verse', 'instruction', 'input'])
        
        # "phxdev/corporate-speak-dataset"
        # dataset = dataset.remove_columns(['instruction', 'input', 'context', 'text', 'bidirectional'])
        
        #  "Samarth0710/neurips-2024-peer-reviews"
        # dataset = dataset.remove_columns( ['paper_id', 'title', 'abstract', 'pdf_url'])

        # "emilpartow/reddit_finance_posts_sp500"
        # dataset = dataset.remove_columns( ['id', 'title', 'created_utc', 'created_datetime', 'author', 'score', 'num_comments', 'upvote_ratio', 'flair', 'permalink', 'url', 'subreddit', 'company'])
      
        # "emilpartow/reddit_finance_posts_sp500"
        # dataset = dataset.remove_columns(['METADATA', 'SOURCE'])  'input', 'instruction'
      
      
      
       
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        
        train_tokenized = splits["train"].map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        val_tokenized = splits["test"].map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        # Load fresh model for each iteration to prevent style "bleeding"
        base_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
        model = get_peft_model(base_model, peft_config)

        # Training Args
        training_args = TrainingArguments(
            output_dir=f"./temp_out_{short_name}",
            eval_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            fp16=True, # Critical for speed/memory
            logging_steps=50,
            save_strategy="no", # Don't save large checkpoints
            report_to="none"    # Set to "wandb" if you want to track 30 runs online
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        # Train and Save
        trainer.train()
        
        adapter_path = f"./adapters/{short_name}_lora"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        print(f"✅ Finished {short_name}. Saved to {adapter_path}")

        # 5. MEMORY CLEANUP (The most important part for 30 runs)
        del base_model
        del model
        del trainer
        gc.collect()           # Python garbage collection
        torch.cuda.empty_cache() # Clear VRAM
        
    except Exception as e:
        print(f"❌ Failed to train {ds_path}: {e}")
        continue

print("\n🎉 All 30+ datasets processed!")