import os
import math
import torch
import gc
import json
import mlflow
import mlflow.pytorch
import pandas as pd
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
MODEL_ID = "gpt2"
MLFLOW_TRACKING_URI = "file:./mlruns"  # Change to remote server if needed
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoRA-1000-Datasets")

# Text column detection
POSSIBLE_TEXT_COLS = [
    "Poem", "text", "Text", "TEXT", "content", "instruction", 
    "body", "stories", "Output", "output", "chunk", 
    "messages", "document_text", "reviews", "dialogue", "summary"
]

# LoRA Configuration (Shared)
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)

# Training hyperparameters
TRAINING_PARAMS = {
    'learning_rate': 2e-4,
    'per_device_train_batch_size': 4,
    'num_train_epochs': 3,
    'max_seq_length': 128,
    'max_samples': 10000,
}

# ============================================================================
# 2. DATASET CATALOG MANAGEMENT
# ============================================================================
class DatasetCatalog:
    """Manage 1000 dataset configurations"""
    
    def __init__(self, catalog_path='datasets_catalog.csv'):
        self.catalog_path = catalog_path
        self.df = self._load_or_create_catalog()
    
    def _load_or_create_catalog(self):
        """Load existing catalog or create new one"""
        if os.path.exists(self.catalog_path):
            return pd.read_csv(self.catalog_path)
        else:
            # Create sample catalog with your 30 datasets
            data = {
                'dataset_id': range(1, 31),
                'name': [
                    "babylm-100M-children-stories",
                    "bedtime_stories",
                    "Shakespeare_Corpus",
                    "KingJamesVersionBible",
                    "old_english_dataset",
                    "victorian_authorship",
                    "poetry-chinese-zhtw",
                    "fairy_tales",
                    "Mythological",
                    "ChatPILE-Casual",
                    "corporate-speak-dataset",
                    "research-paper-abstracts",
                    "neurips-2024-peer-reviews",
                    "reddit_finance_posts_sp500",
                    "reddit-sarcasm",
                    "reddit-logic",
                    "reddit_top_comments",
                    "reddit-blogspot-twitter",
                    "reddit_autism_dataset",
                    "covid_dialogue",
                    "tv_dialogue",
                    "American_English_Natural_Dialogue_Speech_Data",
                    "medical_dialogue-chinese-zhtw",
                    "dialy_dialogue_with_recoginized_concept_raw",
                    "soccer-dialogues",
                    "empathetic_dialogues_for_lm",
                    "empathetic_dialogues_v2",
                    "Poetry-Foundation-Poems",
                    "sample_dataset_29",
                    "sample_dataset_30"
                ],
                'hf_path': [
                    "deven367/babylm-100M-children-stories",
                    "gofilipa/bedtime_stories",
                    "sarnab/Shakespeare_Corpus",
                    "SzuTao/KingJamesVersionBible",
                    "azizsi/old_english_dataset",
                    "contemmcm/victorian_authorship",
                    "erhwenkuo/poetry-chinese-zhtw",
                    "aslicu/fairy_tales",
                    "AJ69/Mythological",
                    "Smilyai-labs/ChatPILE-Casual",
                    "phxdev/corporate-speak-dataset",
                    "sumukshashidhar-testing/research-paper-abstracts",
                    "Samarth0710/neurips-2024-peer-reviews",
                    "emilpartow/reddit_finance_posts_sp500",
                    "Thewillonline/reddit-sarcasm",
                    "agentlans/reddit-logic",
                    "cowWhySo/reddit_top_comments",
                    "jonaskoenig/reddit-blogspot-twitter",
                    "Osondu/reddit_autism_dataset",
                    "Tlighteval/covid_dialogue",
                    "sedthh/tv_dialogue",
                    "Nexdata/American_English_Natural_Dialogue_Speech_Data",
                    "erhwenkuo/medical_dialogue-chinese-zhtw",
                    "jpeandrew/dialy_dialogue_with_recoginized_concept_raw",
                    "rony/soccer-dialogues",
                    "pixelsandpointers/empathetic_dialogues_for_lm",
                    "Adapting/empathetic_dialogues_v2",
                    "suayptalha/Poetry-Foundation-Poems",
                    "custom/sample_29",
                    "custom/sample_30"
                ],
                'domain': ['literature', 'literature', 'literature', 'literature', 'literature', 
                          'literature', 'literature', 'literature', 'literature', 'social_media',
                          'business', 'academic', 'academic', 'finance', 'social_media',
                          'social_media', 'social_media', 'social_media', 'social_media', 'dialogue',
                          'dialogue', 'dialogue', 'medical', 'dialogue', 'dialogue', 'dialogue', 'dialogue', 'literature', 'custom', 'custom'],
                'style': ['narrative', 'narrative', 'poetic', 'biblical', 'archaic',
                         'victorian', 'poetic', 'narrative', 'mythological', 'casual',
                         'formal', 'technical', 'academic', 'technical', 'sarcastic',
                         'logical', 'social', 'mixed', 'technical', 'dialogue',
                         'dialogue', 'dialogue', 'medical', 'dialogue', 'dialogue', 'empathetic', 'empathetic', 'poetic', 'mixed', 'mixed'],
                'status': ['pending'] * 30,
                'token_count': [0] * 30,
                'training_time_minutes': [0] * 30,
                'perplexity': [0.0] * 30
            }
            df = pd.DataFrame(data)
            df.to_csv(self.catalog_path, index=False)
            return df
    
    def get_pending_datasets(self):
        """Get datasets that haven't been trained yet"""
        return self.df[self.df['status'] == 'pending'].to_dict('records')
    
    def update_status(self, dataset_id, status, metrics=None):
        """Update dataset training status"""
        idx = self.df[self.df['dataset_id'] == dataset_id].index[0]
        self.df.at[idx, 'status'] = status
        if metrics:
            for key, val in metrics.items():
                if key in self.df.columns:
                    self.df.at[idx, key] = val
        self.df.to_csv(self.catalog_path, index=False)
    
    def get_dataset_by_id(self, dataset_id):
        """Retrieve dataset config"""
        return self.df[self.df['dataset_id'] == dataset_id].iloc[0].to_dict()

# ============================================================================
# 3. TOKENIZER & DATA PREPROCESSING
# ============================================================================
class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_text_column(self, dataset):
        """Find the text column in dataset"""
        for col in POSSIBLE_TEXT_COLS:
            if col in dataset.column_names:
                return col
        raise ValueError(f"No text column found. Available: {dataset.column_names}")
    
    def tokenize_fn(self, examples, text_col):
        """Tokenize text examples"""
        texts = [str(t) + self.tokenizer.eos_token for t in examples[text_col]]
        return self.tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=TRAINING_PARAMS['max_seq_length']
        )

# ============================================================================
# 4. TRAINING WITH MLFLOW
# ============================================================================
class LoRATrainer:
    def __init__(self, model_id, peft_config):
        self.model_id = model_id
        self.peft_config = peft_config
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_processor = DataProcessor(self.tokenizer)
    
    def train_on_dataset(self, dataset_config):
        """Train LoRA model on a single dataset with MLflow tracking"""
        dataset_id = dataset_config['dataset_id']
        dataset_name = dataset_config['name']
        hf_path = dataset_config['hf_path']
        
        with mlflow.start_run(run_name=f"lora_{dataset_name}"):
            try:
                logger.info(f"[{dataset_id}/1000] Training on {dataset_name}...")
                
                # Log parameters
                mlflow.log_params({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'domain': dataset_config['domain'],
                    'style': dataset_config['style'],
                    'model_id': self.model_id,
                    'r': self.peft_config.r,
                    'lora_alpha': self.peft_config.lora_alpha,
                    'learning_rate': TRAINING_PARAMS['learning_rate'],
                    'batch_size': TRAINING_PARAMS['per_device_train_batch_size'],
                    'num_epochs': TRAINING_PARAMS['num_train_epochs']
                })
                
                # Load dataset
                start_time = datetime.now()
                dataset = load_dataset(hf_path, split="train")
                
                # Limit samples
                limit = min(len(dataset), TRAINING_PARAMS['max_samples'])
                dataset = dataset.select(range(limit))
                
                # Get text column
                text_col = self.data_processor.get_text_column(dataset)
                
                # Preprocess and tokenize
                splits = dataset.train_test_split(test_size=0.1, seed=42)
                
                tokenize_fn_bound = lambda examples: self.data_processor.tokenize_fn(examples, text_col)
                train_tokenized = splits["train"].map(
                    tokenize_fn_bound, 
                    batched=True, 
                    remove_columns=dataset.column_names
                )
                val_tokenized = splits["test"].map(
                    tokenize_fn_bound, 
                    batched=True, 
                    remove_columns=dataset.column_names
                )
                
                # Load fresh model
                base_model = GPT2LMHeadModel.from_pretrained(self.model_id)
                model = get_peft_model(base_model, self.peft_config)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=f"./temp_out_{dataset_name}",
                    eval_strategy="epoch",
                    learning_rate=TRAINING_PARAMS['learning_rate'],
                    per_device_train_batch_size=TRAINING_PARAMS['per_device_train_batch_size'],
                    num_train_epochs=TRAINING_PARAMS['num_train_epochs'],
                    fp16=True,
                    logging_steps=50,
                    save_strategy="no",
                    report_to="none"
                )
                
                # Train
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_tokenized,
                    eval_dataset=val_tokenized,
                    data_collator=DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer, 
                        mlm=False
                    )
                )
                
                trainer.train()
                
                # Save adapter
                adapter_path = f"./adapters/{dataset_name}_lora"
                os.makedirs(adapter_path, exist_ok=True)
                model.save_pretrained(adapter_path)
                self.tokenizer.save_pretrained(adapter_path)
                
                # Log model
                mlflow.pytorch.log_model(model, f"lora_adapter_{dataset_id}")
                
                # Calculate metrics
                eval_result = trainer.evaluate()
                training_time = (datetime.now() - start_time).total_seconds() / 60
                
                # Log metrics
                mlflow.log_metrics({
                    'eval_loss': eval_result.get('eval_loss', 0),
                    'eval_perplexity': math.exp(eval_result.get('eval_loss', 0)),
                    'training_time_minutes': training_time,
                    'token_count': len(dataset)
                })
                
                logger.info(f"✅ Successfully trained {dataset_name} (Time: {training_time:.2f}min)")
                
                # Cleanup
                del base_model, model, trainer
                gc.collect()
                torch.cuda.empty_cache()
                
                return {
                    'status': 'completed',
                    'perplexity': math.exp(eval_result.get('eval_loss', 0)),
                    'training_time_minutes': training_time,
                    'token_count': len(dataset)
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to train {dataset_name}: {str(e)}")
                mlflow.log_param('error', str(e))
                return {'status': 'failed', 'error': str(e)}

# ============================================================================
# 5. MAIN ORCHESTRATOR
# ============================================================================
def main():
    logger.info("="*60)
    logger.info("Starting LoRA Training for 1000 Datasets")
    logger.info("="*60)
    
    # Initialize
    catalog = DatasetCatalog()
    trainer = LoRATrainer(MODEL_ID, PEFT_CONFIG)
    
    # Get pending datasets
    pending_datasets = catalog.get_pending_datasets()
    logger.info(f"Found {len(pending_datasets)} datasets to train")
    
    # Training loop
    completed = 0
    failed = 0
    
    for idx, dataset_config in enumerate(pending_datasets, 1):
        logger.info(f"\n[{idx}/{len(pending_datasets)}] Processing {dataset_config['name']}")
        
        result = trainer.train_on_dataset(dataset_config)
        
        if result['status'] == 'completed':
            catalog.update_status(
                dataset_config['dataset_id'], 
                'completed', 
                {
                    'perplexity': result['perplexity'],
                    'training_time_minutes': result['training_time_minutes'],
                    'token_count': result['token_count']
                }
            )
            completed += 1
        else:
            catalog.update_status(dataset_config['dataset_id'], 'failed')
            failed += 1
    
    logger.info("\n" + "="*60)
    logger.info(f"Training Complete!")
    logger.info(f"✅ Completed: {completed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total: {completed + failed}")
    logger.info(f"📁 MLflow UI: {MLFLOW_TRACKING_URI}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
