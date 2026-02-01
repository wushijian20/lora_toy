"""
Diagnose what files are actually in the adapters folders
"""

import os
from pathlib import Path
import pandas as pd

def explore_adapter_structure(adapters_dir='./adapters'):
    """Explore the actual structure of adapter folders"""
    
    if not os.path.exists(adapters_dir):
        print(f"❌ Adapters directory not found: {adapters_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Exploring Adapter Folder Structure")
    print(f"{'='*70}\n")
    
    adapter_info = []
    
    for adapter_name in sorted(os.listdir(adapters_dir)):
        adapter_path = os.path.join(adapters_dir, adapter_name)
        
        if not os.path.isdir(adapter_path):
            continue
        
        print(f"📁 {adapter_name}/")
        files = os.listdir(adapter_path)
        
        for file in sorted(files):
            file_path = os.path.join(adapter_path, file)
            if os.path.isfile(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                print(f"   ├─ {file:<40} ({size_kb:>10.2f} KB)")
            else:
                print(f"   ├─ {file}/ (folder)")
        
        # Record info
        adapter_info.append({
            'adapter_name': adapter_name,
            'num_files': len(files),
            'has_adapter_config': 'adapter_config.json' in files,
            'has_adapter_model_bin': 'adapter_model.bin' in files,
            'has_pytorch_model_bin': 'pytorch_model.bin' in files,
            'has_safetensors': any(f.endswith('.safetensors') for f in files),
            'files': ', '.join(files)
        })
        
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(adapter_info)
    
    print(f"Total adapters: {len(df)}")
    print(f"\nFile presence:")
    print(f"  adapter_config.json: {df['has_adapter_config'].sum()}/{len(df)}")
    print(f"  adapter_model.bin: {df['has_adapter_model_bin'].sum()}/{len(df)}")
    print(f"  pytorch_model.bin: {df['has_pytorch_model_bin'].sum()}/{len(df)}")
    print(f"  .safetensors: {df['has_safetensors'].sum()}/{len(df)}")
    
    return df

if __name__ == '__main__':
    explore_adapter_structure()
