"""
Reorganize LoRA project structure
Run this script to automatically reorganize your project
"""

import os
import shutil
from pathlib import Path

def create_directory_structure(base_path='.'):
    """Create organized directory structure"""
    
    directories = [
        'scripts/preprocessing',
        'scripts/training',
        'scripts/evaluation',
        'scripts/inspection',
        'src/data',
        'src/models',
        'src/training',
        'src/utils',
        'config',
        'data/raw',
        'data/processed',
        'data/datasets',
        'outputs/logs',
        'outputs/checkpoints',
        'outputs/metrics',
        'experiments',
        'notebooks',
        'tests',
        'docs'
    ]
    
    print("📁 Creating directory structure...\n")
    
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
        
        # Create __init__.py for Python packages
        if dir_path.startswith('src/'):
            init_file = os.path.join(full_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Package initialization."""\n')

def move_files(base_path='.'):
    """Move files to their new locations"""
    
    print("\n📦 Moving files to new locations...\n")
    
    # File movements (source -> destination)
    movements = {
        # Inspection scripts
        'inspect_adapters.py': 'scripts/inspection/',
        'inspect_lora_weights.py': 'scripts/inspection/',
        'view_lora_detailed_values.py': 'scripts/inspection/',
        'diagnose_adapters.py': 'scripts/inspection/',
        
        # Training scripts
        'lora_30_datasets.py': 'scripts/training/',
        'lora_1000_datasets.py': 'scripts/training/',
        'lora_shakespeare.py': 'scripts/training/',
        'lora_poetry.py': 'scripts/training/',
        'lora_children.py': 'scripts/training/',
        'lora_combined.py': 'scripts/training/',
        'lora.py': 'scripts/training/',
        'lora_shakespeare_test.py': 'scripts/training/',
        
        # Data processing scripts
        'data_generator.py': 'scripts/preprocessing/',
        'extract_children_speaking.py': 'scripts/preprocessing/',
        'extract_data.py': 'scripts/preprocessing/',
        
        # MLflow utilities
        'mlflow_manager.py': 'src/utils/',
        'create_dataset_catalog.py': 'scripts/training/',
        
        # Data files
        'kids.txt': 'data/raw/',
        'toddler_data.txt': 'data/raw/',
        
        # Config files
        'datasets_catalog.csv': 'config/',
    }
    
    for src, dst_dir in movements.items():
        src_path = os.path.join(base_path, src)
        dst_path = os.path.join(base_path, dst_dir, os.path.basename(src))
        
        if os.path.exists(src_path):
            try:
                # Create backup
                backup_path = src_path + '.backup'
                shutil.copy2(src_path, backup_path)
                
                # Move file
                shutil.move(src_path, dst_path)
                print(f"  ✓ Moved: {src} -> {dst_dir}")
                
                # Remove backup if successful
                os.remove(backup_path)
            except Exception as e:
                print(f"  ⚠️  Failed to move {src}: {e}")
                # Restore from backup if exists
                if os.path.exists(backup_path):
                    shutil.move(backup_path, src_path)
        else:
            print(f"  ℹ️  Skipped: {src} (not found)")

def create_gitignore(base_path='.'):
    """Create .gitignore file"""
    
    print("\n📝 Creating .gitignore...\n")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/raw/
data/processed/
data/datasets/
*.csv
*.txt
*.json

# Models & Outputs
adapters/
temp_out_*/
*.bin
*.safetensors
*.pt
*.pth

# MLflow
mlruns/

# Outputs
outputs/
experiments/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# Backups
*.backup
"""
    
    gitignore_path = os.path.join(base_path, '.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    print(f"  ✓ Created: .gitignore")

def create_requirements(base_path='.'):
    """Create requirements.txt"""
    
    print("\n📦 Creating requirements.txt...\n")
    
    requirements_content = """# Core ML Libraries
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.12.0

# MLflow & Tracking
mlflow>=2.5.0

# Data Processing
pandas>=1.5.0
numpy>=1.24.0

# Model Serialization
safetensors>=0.3.0

# Development
jupyter>=1.0.0
ipython>=8.0.0

# Testing
pytest>=7.0.0
"""
    
    requirements_path = os.path.join(base_path, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    
    print(f"  ✓ Created: requirements.txt")

def create_readme(base_path='.'):
    """Create updated README.md"""
    
    print("\n📖 Creating README.md...\n")
    
    readme_content = """# LoRA Training Project

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
"""
    
    readme_path = os.path.join(base_path, 'README.md')
    
    # Backup existing README if it exists
    if os.path.exists(readme_path):
        shutil.copy2(readme_path, readme_path + '.backup')
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ Created: README.md")

def create_summary_report(base_path='.'):
    """Create summary report of reorganization"""
    
    print("\n" + "="*70)
    print("✅ REORGANIZATION COMPLETE!")
    print("="*70)
    
    print("\n📊 Summary:")
    print("  ✓ Directory structure created")
    print("  ✓ Files moved to new locations")
    print("  ✓ .gitignore created")
    print("  ✓ requirements.txt created")
    print("  ✓ README.md updated")
    
    print("\n📁 New Structure:")
    print("  - scripts/       # All utility scripts")
    print("  - src/           # Core source code")
    print("  - config/        # Configuration files")
    print("  - data/          # Data files")
    print("  - outputs/       # Training outputs")
    
    print("\n🚀 Next Steps:")
    print("  1. Review the new structure")
    print("  2. Update import paths in your scripts")
    print("  3. Run: pip install -r requirements.txt")
    print("  4. Test your training scripts")
    print("  5. Commit changes to git")
    
    print("\n💡 Useful Commands:")
    print("  - Train models: python scripts/training/lora_1000_datasets.py")
    print("  - Inspect adapters: python scripts/inspection/inspect_adapters.py")
    print("  - View MLflow: mlflow ui")
    
    print("\n" + "="*70)

def main():
    """Main reorganization function"""
    
    print("\n" + "="*70)
    print("🔧 LoRA Project Reorganization Script")
    print("="*70)
    
    base_path = '.'
    
    # Confirm before proceeding
    print("\n⚠️  This will reorganize your project structure.")
    print("   Backups will be created for moved files.")
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Reorganization cancelled.")
        return
    
    try:
        # Execute reorganization steps
        create_directory_structure(base_path)
        move_files(base_path)
        create_gitignore(base_path)
        create_requirements(base_path)
        create_readme(base_path)
        create_summary_report(base_path)
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        print("   Please restore from backups if needed.")
        return

if __name__ == '__main__':
    main()
