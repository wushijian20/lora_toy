"""
Inspect LoRA parameters from saved adapters in the adapters/ folder
"""

import os
import json
from pathlib import Path
import pandas as pd
from peft import PeftConfig, PeftModel
from transformers import GPT2LMHeadModel

class LoRAInspector:
    """Inspect and analyze LoRA adapters"""
    
    def __init__(self, adapters_dir='./adapters'):
        self.adapters_dir = adapters_dir
        self.adapters = self._find_adapters()
    
    def _find_adapters(self):
        """Find all LoRA adapters in directory"""
        adapters = []
        if not os.path.exists(self.adapters_dir):
            print(f"❌ Adapters directory not found: {self.adapters_dir}")
            return adapters
        
        for adapter_name in os.listdir(self.adapters_dir):
            adapter_path = os.path.join(self.adapters_dir, adapter_name)
            config_path = os.path.join(adapter_path, 'adapter_config.json')
            
            if os.path.isdir(adapter_path) and os.path.exists(config_path):
                adapters.append({
                    'name': adapter_name,
                    'path': adapter_path,
                    'config_path': config_path
                })
        
        return adapters
    
    def get_adapter_config(self, adapter_name):
        """Load adapter_config.json"""
        for adapter in self.adapters:
            if adapter['name'] == adapter_name:
                with open(adapter['config_path'], 'r') as f:
                    return json.load(f)
        return None
    
    def print_all_configs(self):
        """Print LoRA config for all adapters"""
        if not self.adapters:
            print("❌ No adapters found")
            return
        
        print(f"\n{'='*70}")
        print(f"Found {len(self.adapters)} LoRA Adapters")
        print(f"{'='*70}\n")
        
        for idx, adapter in enumerate(self.adapters, 1):
            print(f"{idx}. {adapter['name']}")
            print(f"   Path: {adapter['path']}")
            
            try:
                config = self.get_adapter_config(adapter['name'])
                self._print_config(config)
            except Exception as e:
                print(f"   ❌ Error reading config: {e}")
            
            print()
    
    def _print_config(self, config):
        """Pretty print LoRA config"""
        if isinstance(config, dict):
            for key, value in config.items():
                if key != 'target_modules':  # target_modules can be long
                    print(f"   ├─ {key}: {value}")
                else:
                    print(f"   ├─ {key}: {value[:2]}..." if isinstance(value, list) else f"   ├─ {key}: {value}")
    
    def export_config_summary(self, output_file='adapter_configs_summary.csv'):
        """Export summary of all adapter configs to CSV"""
        if not self.adapters:
            print("❌ No adapters found")
            return
        
        summary_data = []
        
        for adapter in self.adapters:
            try:
                config = self.get_adapter_config(adapter['name'])
                summary_data.append({
                    'adapter_name': adapter['name'],
                    'r': config.get('r'),
                    'lora_alpha': config.get('lora_alpha'),
                    'lora_dropout': config.get('lora_dropout'),
                    'target_modules': str(config.get('target_modules')),
                    'task_type': config.get('task_type'),
                    'peft_type': config.get('peft_type'),
                })
            except Exception as e:
                print(f"⚠️ Error reading {adapter['name']}: {e}")
                continue
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        print(f"✅ Exported summary to {output_file}")
        print(f"\nSummary Statistics:")
        print(df[['r', 'lora_alpha', 'lora_dropout']].describe())
        
        return df
    
    def load_adapter_model(self, adapter_name, base_model_id='gpt2'):
        """Load full model with LoRA adapter"""
        adapter = next((a for a in self.adapters if a['name'] == adapter_name), None)
        if not adapter:
            print(f"❌ Adapter not found: {adapter_name}")
            return None
        
        try:
            base_model = GPT2LMHeadModel.from_pretrained(base_model_id)
            peft_model = PeftModel.from_pretrained(base_model, adapter['path'])
            print(f"✅ Loaded {adapter_name} with base model {base_model_id}")
            return peft_model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def compare_adapters(self, adapter_names=None):
        """Compare parameters across multiple adapters"""
        if adapter_names is None:
            adapter_names = [a['name'] for a in self.adapters[:5]]  # First 5
        
        comparison_data = []
        
        for adapter_name in adapter_names:
            config = self.get_adapter_config(adapter_name)
            if config:
                comparison_data.append({
                    'adapter': adapter_name,
                    'r': config.get('r'),
                    'alpha': config.get('lora_alpha'),
                    'dropout': config.get('lora_dropout'),
                })
        
        df = pd.DataFrame(comparison_data)
        print(f"\n{'='*70}")
        print("Adapter Parameter Comparison:")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        return df
    
    def get_model_size_info(self):
        """Get size of each adapter"""
        print(f"\n{'='*70}")
        print("Adapter Folder Sizes:")
        print(f"{'='*70}\n")
        
        size_data = []
        total_size = 0
        
        for adapter in self.adapters:
            adapter_size = self._get_dir_size(adapter['path'])
            total_size += adapter_size
            size_data.append({
                'adapter_name': adapter['name'],
                'size_mb': adapter_size / (1024 * 1024),
                'size_gb': adapter_size / (1024 * 1024 * 1024)
            })
        
        df = pd.DataFrame(size_data)
        df = df.sort_values('size_mb', ascending=False)
        
        for _, row in df.iterrows():
            print(f"  {row['adapter_name']}: {row['size_mb']:.2f} MB ({row['size_gb']:.2f} GB)")
        
        print(f"\n  Total: {total_size / (1024 * 1024):.2f} MB ({total_size / (1024 * 1024 * 1024):.2f} GB)")
        
        return df
    
    def _get_dir_size(self, directory):
        """Calculate total size of directory"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total

def main():
    """Example usage"""
    print("🔍 LoRA Adapter Inspector\n")
    
    inspector = LoRAInspector()
    
    # 1. List all adapters and their configs
    inspector.print_all_configs()
    
    # 2. Export summary to CSV
    df_summary = inspector.export_config_summary()
    
    # 3. Compare adapters
    inspector.compare_adapters()
    
    # 4. Check folder sizes
    inspector.get_model_size_info()
    
    # 5. Load and inspect a specific adapter (if exists)
    if inspector.adapters:
        first_adapter = inspector.adapters[0]['name']
        print(f"\n{'='*70}")
        print(f"Loading adapter: {first_adapter}")
        print(f"{'='*70}")
        model = inspector.load_adapter_model(first_adapter)
        if model:
            print(f"\n✅ Model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Model config: {model.config}")

if __name__ == '__main__':
    main()
