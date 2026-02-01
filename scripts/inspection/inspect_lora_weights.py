"""
Inspect LoRA A and LoRA B weight matrices from adapters
These are the actual learned low-rank decomposition matrices
"""

import os
import torch
import json
from pathlib import Path
from peft import PeftConfig, PeftModel
from transformers import GPT2LMHeadModel
import pandas as pd

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠️ safetensors not installed. Install with: pip install safetensors")

class LoRAWeightInspector:
    """Inspect LoRA A and LoRA B weight matrices"""
    
    def __init__(self, adapters_dir='./adapters'):
        self.adapters_dir = adapters_dir
        self.adapters = self._find_adapters()
    
    def _find_adapters(self):
        """Find all LoRA adapters"""
        adapters = []
        if not os.path.exists(self.adapters_dir):
            return adapters
        
        for adapter_name in os.listdir(self.adapters_dir):
            adapter_path = os.path.join(self.adapters_dir, adapter_name)
            if os.path.isdir(adapter_path):
                adapters.append({
                    'name': adapter_name,
                    'path': adapter_path
                })
        
        return adapters
    
    def get_lora_weights(self, adapter_name):
        """
        Extract LoRA A and LoRA B weights from adapter
        
        LoRA decomposition: Δw = B @ A
        - A: (r, d_in)  - down-projection matrix
        - B: (d_out, r) - up-projection matrix
        """
        adapter = next((a for a in self.adapters if a['name'] == adapter_name), None)
        if not adapter:
            print(f"❌ Adapter not found: {adapter_name}")
            return None
        
        adapter_path = adapter['path']
        
        # Try safetensors first, then .bin format
        weights_file = os.path.join(adapter_path, 'adapter_model.safetensors')
        is_safetensors = True
        
        if not os.path.exists(weights_file):
            weights_file = os.path.join(adapter_path, 'adapter_model.bin')
            is_safetensors = False
        
        if not os.path.exists(weights_file):
            print(f"❌ Weights file not found: {weights_file}")
            return None
        
        try:
            # Load weights
            if is_safetensors:
                if not HAS_SAFETENSORS:
                    print("❌ safetensors library not available. Install with: pip install safetensors")
                    return None
                state_dict = load_safetensors(weights_file)
            else:
                state_dict = torch.load(weights_file, map_location='cpu')
            
            print(f"\n{'='*70}")
            print(f"LoRA Weights for: {adapter_name}")
            print(f"{'='*70}")
            print(f"\nTotal parameters: {sum(p.numel() for p in state_dict.values()):,}")
            print(f"\n{'Module':<40} {'Type':<10} {'Shape':<20} {'Size'}")
            print(f"{'-'*70}")
            
            lora_weights = {}
            
            for key, tensor in state_dict.items():
                print(f"{key:<40} {str(tensor.dtype):<10} {str(tuple(tensor.shape)):<20} {tensor.numel():,}")
                lora_weights[key] = tensor
            
            return lora_weights
            
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return None
    
    def analyze_lora_pair(self, adapter_name, layer_name='c_attn'):
        """
        Analyze a LoRA A-B pair for a specific layer
        
        Example: c_attn.lora_A.weight and c_attn.lora_B.weight
        """
        weights = self.get_lora_weights(adapter_name)
        if not weights:
            return None
        
        print(f"\n{'='*70}")
        print(f"Detailed Analysis: {adapter_name} - {layer_name}")
        print(f"{'='*70}\n")
        
        # Find A and B matrices for this layer
        lora_a_key = f"base_model.model.{layer_name}.lora_A.weight"
        lora_b_key = f"base_model.model.{layer_name}.lora_B.weight"
        
        if lora_a_key in weights and lora_b_key in weights:
            lora_a = weights[lora_a_key]
            lora_b = weights[lora_b_key]
            
            print(f"LoRA A (down-projection, input projection):")
            print(f"  Shape: {tuple(lora_a.shape)}  (r × d_in)")
            print(f"  Rank (r): {lora_a.shape[0]}")
            print(f"  Input dim: {lora_a.shape[1]}")
            print(f"  Parameters: {lora_a.numel():,}")
            print(f"  Memory: {lora_a.numel() * 4 / 1024:.2f} KB (float32)")
            print(f"  Min: {lora_a.min().item():.4f}, Max: {lora_a.max().item():.4f}")
            print(f"  Mean: {lora_a.mean().item():.4f}, Std: {lora_a.std().item():.4f}")
            
            print(f"\nLoRA B (up-projection, output projection):")
            print(f"  Shape: {tuple(lora_b.shape)}  (d_out × r)")
            print(f"  Output dim: {lora_b.shape[0]}")
            print(f"  Rank (r): {lora_b.shape[1]}")
            print(f"  Parameters: {lora_b.numel():,}")
            print(f"  Memory: {lora_b.numel() * 4 / 1024:.2f} KB (float32)")
            print(f"  Min: {lora_b.min().item():.4f}, Max: {lora_b.max().item():.4f}")
            print(f"  Mean: {lora_b.mean().item():.4f}, Std: {lora_b.std().item():.4f}")
            
            # Compute final weight matrix
            delta_w = torch.matmul(lora_b, lora_a)
            print(f"\nΔW = B @ A (final weight update):")
            print(f"  Shape: {tuple(delta_w.shape)}  (d_out × d_in)")
            print(f"  Parameters: {delta_w.numel():,}")
            print(f"  Memory: {delta_w.numel() * 4 / 1024:.2f} KB (float32)")
            
            # Compute compression ratio
            total_full_rank = delta_w.shape[0] * delta_w.shape[1]
            total_lora = lora_a.numel() + lora_b.numel()
            compression = total_full_rank / total_lora
            print(f"\n  Compression ratio: {compression:.2f}x")
            print(f"  Storage saved: {(1 - total_lora/total_full_rank)*100:.1f}%")
            
            return {
                'lora_a': lora_a,
                'lora_b': lora_b,
                'delta_w': delta_w,
                'compression_ratio': compression
            }
        else:
            print(f"⚠️ Could not find {layer_name} weights")
            print(f"Available keys: {[k for k in weights.keys()]}")
            return None
    
    def print_all_weights_summary(self):
        """Print summary of all adapters and their weights"""
        if not self.adapters:
            print("❌ No adapters found")
            return
        
        print(f"\n{'='*70}")
        print(f"LoRA Weights Summary for All Adapters")
        print(f"{'='*70}\n")
        
        summary_data = []
        
        for adapter in self.adapters:
            weights = self.get_lora_weights(adapter['name'])
            if weights:
                # Count LoRA A and B matrices
                lora_a_count = sum(1 for k in weights.keys() if 'lora_A' in k)
                lora_b_count = sum(1 for k in weights.keys() if 'lora_B' in k)
                total_params = sum(p.numel() for p in weights.values())
                
                summary_data.append({
                    'adapter': adapter['name'],
                    'lora_A_matrices': lora_a_count,
                    'lora_B_matrices': lora_b_count,
                    'total_parameters': total_params,
                    'size_kb': (total_params * 4 / 1024)
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))
            print(f"\nTotal adapters: {len(df)}")
            print(f"Avg parameters per adapter: {df['total_parameters'].mean():,.0f}")
            print(f"Avg size per adapter: {df['size_kb'].mean():.2f} KB")
    
    def visualize_lora_matrix(self, adapter_name, layer_name='c_attn', matrix_type='A'):
        """
        Show statistics about LoRA matrix values
        matrix_type: 'A' or 'B'
        """
        weights = self.get_lora_weights(adapter_name)
        if not weights:
            return
        
        key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
        
        if key in weights:
            matrix = weights[key]
            print(f"\n{'='*70}")
            print(f"Matrix Statistics: {key}")
            print(f"{'='*70}")
            print(f"Shape: {tuple(matrix.shape)}")
            print(f"Type: {matrix.dtype}")
            print(f"Min: {matrix.min().item():.6f}")
            print(f"Max: {matrix.max().item():.6f}")
            print(f"Mean: {matrix.mean().item():.6f}")
            print(f"Std: {matrix.std().item():.6f}")
            print(f"Median: {matrix.median().item():.6f}")
            
            # Histogram
            print(f"\nValue distribution (histogram):")
            hist, bin_edges = torch.histogram(matrix.flatten(), bins=10)
            for i in range(len(hist)):
                bar = '█' * int(hist[i] / hist.max() * 30)
                print(f"  {bin_edges[i]:.3f} to {bin_edges[i+1]:.3f}: {bar} ({hist[i].item():.0f})")
        else:
            print(f"❌ Matrix not found: {key}")

def main():
    """Example usage"""
    print("🔍 LoRA Weight Inspector\n")
    
    inspector = LoRAWeightInspector()
    
    if not inspector.adapters:
        print("❌ No adapters found in ./adapters/")
        return
    
    # 1. Print summary of all adapters
    inspector.print_all_weights_summary()
    
    # 2. Detailed analysis of first adapter
    first_adapter = inspector.adapters[0]['name']
    print(f"\n\nDetailed analysis of first adapter: {first_adapter}")
    result = inspector.analyze_lora_pair(first_adapter, layer_name='c_attn')
    
    # 3. Visualize LoRA A matrix
    if result:
        inspector.visualize_lora_matrix(first_adapter, layer_name='c_attn', matrix_type='A')
        inspector.visualize_lora_matrix(first_adapter, layer_name='c_attn', matrix_type='B')

if __name__ == '__main__':
    main()
