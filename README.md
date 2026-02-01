# LoRA Training Project

Scalable LoRA fine-tuning framework for training 1000+ adapters with MLflow tracking.

## Project Structure

```
lora-project/
├── config/              # Configuration files
├── data/                # Data storage
├── scripts/             # Utility scripts
│   ├── preprocessing/   # Data preprocessing
│   ├── training/        # Training scripts
│   ├── evaluation/      # Model evaluation
│   └── inspection/      # Adapter inspection
├── src/                 # Core source code
├── adapters/            # Trained LoRA adapters
├── mlruns/              # MLflow tracking
├── outputs/             # Training outputs
├── notebooks/           # Jupyter notebooks
└── docs/                # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train LoRA Adapters

```bash
# Train on single dataset
python scripts/training/lora_shakespeare.py

# Train on multiple datasets
python scripts/training/lora_1000_datasets.py
```

### 3. Inspect Adapters

```bash
# View adapter parameters
python scripts/inspection/inspect_adapters.py

# View LoRA A/B weights
python scripts/inspection/view_lora_detailed_values.py
```

### 4. View MLflow Dashboard

```bash
mlflow ui
```

Visit: http://localhost:5000

## Key Scripts

- **scripts/training/lora_1000_datasets.py** - Train 1000 LoRA adapters with MLflow
- **scripts/inspection/view_lora_detailed_values.py** - Inspect LoRA A/B matrices
- **src/utils/mlflow_manager.py** - MLflow utilities and reporting

## Configuration

Edit `config/datasets_catalog.csv` to manage your 1000 datasets.

## Documentation

See `docs/` for detailed guides and API reference.
