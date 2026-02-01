# LoRA-1000: Scalable LoRA Fine-tuning for 1000 Datasets

This project automates the training of 1000 LoRA adapters across diverse datasets with MLflow experiment tracking.

## Architecture

```
data/ ─────► preprocessing ─────► training ─────► mlflow ─────► registry
(1000)         (tokenize)      (LoRA adapter)   (tracking)    (best models)
```

## Files Overview

| File | Purpose |
|------|---------|
| `lora_1000_datasets.py` | Main training orchestrator with MLflow integration |
| `create_dataset_catalog.py` | Generate catalog of 1000 datasets |
| `mlflow_manager.py` | MLflow utilities and reporting |
| `datasets_catalog.csv` | Catalog of all 1000 datasets and their status |

## Quick Start

### 1. Create Dataset Catalog

```bash
python create_dataset_catalog.py
```

This creates `datasets_catalog.csv` with metadata for 1000 datasets.

### 2. Start MLflow Server (Optional)

For remote tracking and UI:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Then visit `http://localhost:5000` in your browser.

### 3. Train All 1000 Models

```bash
python lora_1000_datasets.py
```

The script will:
- Load datasets from HuggingFace Hub
- Preprocess and tokenize each dataset
- Train a LoRA adapter with GPT-2
- Log metrics, parameters, and models to MLflow
- Save adapters locally in `./adapters/`

### 4. View Results & Generate Reports

```bash
python mlflow_manager.py
```

This will:
- Print training statistics
- Export results to `mlflow_results.csv`
- Show top 10 best models
- Register best models to MLflow Registry

## Configuration

Edit `TRAINING_PARAMS` in `lora_1000_datasets.py`:

```python
TRAINING_PARAMS = {
    'learning_rate': 2e-4,           # Adjust learning rate
    'per_device_train_batch_size': 4, # Batch size
    'num_train_epochs': 3,            # Number of epochs
    'max_seq_length': 128,            # Sequence length
    'max_samples': 10000,             # Max samples per dataset
}
```

## Dataset Catalog Structure

`datasets_catalog.csv`:

```csv
dataset_id,name,hf_path,domain,style,status,token_count,training_time_minutes,perplexity
1,babylm-100M-children-stories,deven367/babylm-100M-children-stories,literature,narrative,pending,0,0,0.0
...
1000,dataset_1000_news_journalistic,custom/dataset_1000,news,journalistic,pending,0,0,0.0
```

## Key Features

✅ **Scalable**: Train 1000+ models with automated orchestration  
✅ **MLflow Integration**: Track all metrics, parameters, and artifacts  
✅ **Memory Efficient**: Automatic garbage collection between runs  
✅ **Error Recovery**: Graceful handling of failed datasets  
✅ **Model Registry**: Organize and version best models  
✅ **Comprehensive Logging**: Detailed logs for debugging  

## MLflow Dashboard

The MLflow UI shows:
- Run metrics (perplexity, loss, training time)
- Parameter comparisons across datasets
- Model artifacts and checkpoints
- Dataset domain/style analysis

Access at: `http://localhost:5000`

## Troubleshooting

**Out of Memory?**
- Reduce `max_samples` or `per_device_train_batch_size`
- Add `torch.cuda.empty_cache()` calls

**Dataset not found?**
- Verify HuggingFace dataset path in catalog
- Check internet connection
- Try manually: `huggingface-cli download <dataset_path>`

**Models not appearing in MLflow?**
- Check MLflow server is running
- Verify `MLFLOW_TRACKING_URI` is correct
- Check logs in `lora_training.log`

## Next Steps: LoRA-Rec

With 1000 trained adapters, implement the LoRA-Rec recommendation system:

```python
# Collect LoRA metadata
metadata = {
    'domain': 'literature',
    'style': 'poetic',
    'perplexity': 45.2,
    'tokens': 75000
}

# Train recommender to predict best LoRA for new user
# Use collaborative filtering or neural network
```

## Performance Expectations

- **Per-model training time**: 5-15 minutes (on V100 GPU)
- **Total time for 1000 models**: ~100-250 hours
- **Parallel training**: Use multiple GPUs/machines
- **Storage**: ~50-100 GB for all adapters

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09714)
- [MLflow Documentation](https://mlflow.org/docs)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PEFT Library](https://github.com/huggingface/peft)
