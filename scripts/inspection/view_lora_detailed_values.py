"""
Discover the actual key format in LoRA weights and display values
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

class LoRAKeyDiscovery:
    """View detailed numerical values in LoRA matrices"""
    
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
    
    def load_weights(self, adapter_name):
        """Load LoRA weights from adapter"""
        adapter = next((a for a in self.adapters if a['name'] == adapter_name), None)
        if not adapter:
            print(f"❌ Adapter not found: {adapter_name}")
            return None
        
        adapter_path = adapter['path']
        
        # Try safetensors first, then .bin
        weights_file = os.path.join(adapter_path, 'adapter_model.safetensors')
        is_safetensors = True
        
        if not os.path.exists(weights_file):
            weights_file = os.path.join(adapter_path, 'adapter_model.bin')
            is_safetensors = False
        
        if not os.path.exists(weights_file):
            print(f"❌ Weights file not found")
            return None
        
        try:
            if is_safetensors:
                if not HAS_SAFETENSORS:
                    print("❌ safetensors not installed. pip install safetensors")
                    return None
                state_dict = load_safetensors(weights_file)
            else:
                state_dict = torch.load(weights_file, map_location='cpu')
            
            return state_dict
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return None
    
    def show_all_keys(self, adapter_name):
        """Display all weight keys in the adapter"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        print(f"\n{'='*80}")
        print(f"All Keys in {adapter_name}")
        print(f"{'='*80}\n")
        
        for i, key in enumerate(sorted(weights.keys()), 1):
            shape = weights[key].shape
            numel = weights[key].numel()
            print(f"{i:2d}. {key:<60} Shape: {str(shape):<20} Params: {numel:>10,}")
        
        print(f"\n{'-'*80}")
        print(f"Total keys: {len(weights)}")
    
    def find_lora_matrices(self, adapter_name):
        """Find all LoRA A and B matrices"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return None
        
        print(f"\n{'='*80}")
        print(f"LoRA A and B Matrices in {adapter_name}")
        print(f"{'='*80}\n")
        
        lora_a_keys = [k for k in weights.keys() if 'lora_A' in k]
        lora_b_keys = [k for k in weights.keys() if 'lora_B' in k]
        
        print(f"Found {len(lora_a_keys)} LoRA A matrices:")
        for key in sorted(lora_a_keys):
            shape = weights[key].shape
            print(f"  ├─ {key:<60} Shape: {shape}")
        
        print(f"\nFound {len(lora_b_keys)} LoRA B matrices:")
        for key in sorted(lora_b_keys):
            shape = weights[key].shape
            print(f"  ├─ {key:<60} Shape: {shape}")
        
        print(f"\n{'-'*80}")
        
        return {
            'lora_a_keys': lora_a_keys,
            'lora_b_keys': lora_b_keys,
            'weights': weights
        }
    
    def display_matrix(self, adapter_name, key):
        """Display a specific matrix by key"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            self.show_all_keys(adapter_name)
            return
        
        matrix = weights[key].cpu().numpy()
        
        print(f"\n{'='*80}")
        print(f"Matrix Values: {key}")
        print(f"{'='*80}")
        print(f"Shape: {matrix.shape}")
        print(f"Data type: {matrix.dtype}")
        print(f"\n{'-'*80}\n")
        
        # For large matrices, show sample
        if matrix.shape[0] > 20 or matrix.shape[1] > 20:
            print("(Showing first 10 rows, first 15 columns):\n")
            display_matrix = matrix[:10, :15]
        else:
            display_matrix = matrix
        
        df = pd.DataFrame(display_matrix)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print(df.to_string())
        
        print(f"\n{'-'*80}")
        print(f"Statistics:")
        print(f"  Min: {matrix.min():.8f}")
        print(f"  Max: {matrix.max():.8f}")
        print(f"  Mean: {matrix.mean():.8f}")
        print(f"  Std: {matrix.std():.8f}")
        print(f"  Sum: {matrix.sum():.8f}")
    
    def show_slice(self, adapter_name, key, row_range=(0, 5), col_range=(0, 10)):
        """Show a slice of a specific matrix"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return
        
        matrix = weights[key].cpu().numpy()
        
        r1, r2 = row_range
        c1, c2 = col_range
        
        # Validate ranges
        r2 = min(r2, matrix.shape[0])
        c2 = min(c2, matrix.shape[1])
        
        slice_matrix = matrix[r1:r2, c1:c2]
        
        print(f"\n{'='*80}")
        print(f"Matrix Slice [{r1}:{r2}, {c1}:{c2}]")
        print(f"Key: {key}")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(slice_matrix)
        pd.set_option('display.float_format', '{:.8f}'.format)
        
        print(df.to_string())
        
        print(f"\n{'-'*80}")
    
    def export_to_csv(self, adapter_name, key):
        """Export a matrix to CSV"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return
        
        matrix = weights[key].cpu().numpy()
        
        # Create CSV filename
        safe_key = key.replace('/', '_').replace('.', '_')
        csv_filename = f"{adapter_name}_{safe_key}.csv"
        
        df = pd.DataFrame(matrix)
        df.to_csv(csv_filename, index=True)
        
        print(f"\n✅ Exported to: {csv_filename}")
        print(f"   Shape: {matrix.shape}")
        print(f"   Rows: {matrix.shape[0]}, Columns: {matrix.shape[1]}")
    
    def compare_adapters(self, adapter_names, lora_type='A'):
        """Compare LoRA matrices across adapters"""
        print(f"\n{'='*80}")
        print(f"Comparing LoRA {lora_type} Matrices Across Adapters")
        print(f"{'='*80}\n")
        
        comparison_data = []
        
        for adapter_name in adapter_names:
            weights = self.load_weights(adapter_name)
            if not weights:
                continue
            
            # Find first LoRA A or B matrix
            target_keys = [k for k in weights.keys() if f'lora_{lora_type}' in k]
            
            if not target_keys:
                print(f"⚠️ {adapter_name}: No LoRA {lora_type} matrices found")
                continue
            
            key = target_keys[0]
            matrix = weights[key].cpu().numpy()
            
            comparison_data.append({
                'Adapter': adapter_name,
                'Key': key,
                'Shape': str(matrix.shape),
                'Params': matrix.size,
                'Min': f"{matrix.min():.6f}",
                'Max': f"{matrix.max():.6f}",
                'Mean': f"{matrix.mean():.6f}",
                'Std': f"{matrix.std():.6f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print(f"\n{'-'*80}")
    
    def export_matrix_to_csv(self, adapter_name, layer_name='c_attn', matrix_type='A'):
        """Export LoRA matrix values to CSV file"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return
        
        matrix = weights[key].cpu().numpy()
        
        # Create CSV filename
        csv_filename = f"{adapter_name}_{layer_name}_lora_{matrix_type}.csv"
        
        df = pd.DataFrame(matrix)
        df.to_csv(csv_filename, index=True)
        
        print(f"✅ Exported to: {csv_filename}")
        print(f"   Shape: {matrix.shape}")
        print(f"   Rows: {matrix.shape[0]}, Columns: {matrix.shape[1]}")
    
    def show_slice(self, adapter_name, layer_name='c_attn', matrix_type='A', 
                   row_range=(0, 5), col_range=(0, 10)):
        """
        Show a slice of the matrix
        
        Args:
            row_range: tuple (start_row, end_row)
            col_range: tuple (start_col, end_col)
        """
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return
        
        matrix = weights[key].cpu().numpy()
        
        r1, r2 = row_range
        c1, c2 = col_range
        
        slice_matrix = matrix[r1:r2, c1:c2]
        
        print(f"\n{'='*80}")
        print(f"LoRA {matrix_type} Matrix Slice [{r1}:{r2}, {c1}:{c2}]")
        print(f"Adapter: {adapter_name} / Layer: {layer_name}")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(slice_matrix)
        print(df.to_string())
        print(f"\n{'-'*80}")
    
    def compare_matrices(self, adapter_names, layer_name='c_attn', matrix_type='A'):
        """Compare the same matrix across different adapters"""
        print(f"\n{'='*80}")
        print(f"Comparing LoRA {matrix_type} Matrix across Adapters")
        print(f"{'='*80}\n")
        
        all_stats = []
        
        for adapter_name in adapter_names:
            weights = self.load_weights(adapter_name)
            if not weights:
                continue
            
            key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
            
            if key not in weights:
                print(f"⚠️ {adapter_name}: Key not found")
                continue
            
            matrix = weights[key].cpu().numpy()
            
            all_stats.append({
                'Adapter': adapter_name,
                'Shape': str(matrix.shape),
                'Min': f"{matrix.min():.6f}",
                'Max': f"{matrix.max():.6f}",
                'Mean': f"{matrix.mean():.6f}",
                'Std': f"{matrix.std():.6f}",
                'Sum': f"{matrix.sum():.6f}",
                'Parameters': matrix.size
            })
        
        df = pd.DataFrame(all_stats)
        print(df.to_string(index=False))
        print(f"\n{'-'*80}")
    
    def get_specific_values(self, adapter_name, layer_name='c_attn', 
                           matrix_type='A', indices=None):
        """
        Get specific values from the matrix
        
        Args:
            indices: list of (row, col) tuples, e.g., [(0,0), (0,1), (1,0)]
        """
        weights = self.load_weights(adapter_name)
        if not weights:
            return
        
        key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return
        
        matrix = weights[key].cpu().numpy()
        
        if indices is None:
            # Show first 10 values
            indices = [(i, j) for i in range(min(3, matrix.shape[0])) 
                              for j in range(min(4, matrix.shape[1]))]
        
        print(f"\n{'='*80}")
        print(f"Specific Values from LoRA {matrix_type} Matrix")
        print(f"Adapter: {adapter_name} / Layer: {layer_name}")
        print(f"{'='*80}\n")
        
        print(f"{'Row':<6} {'Col':<6} {'Value':<20}")
        print(f"{'-'*32}")
        
        for row, col in indices:
            if row < matrix.shape[0] and col < matrix.shape[1]:
                value = matrix[row, col]
                print(f"{row:<6} {col:<6} {value:<20.10f}")
            else:
                print(f"{row:<6} {col:<6} {'OUT OF BOUNDS':<20}")
        
        print(f"\n{'-'*80}")
    
    def create_heatmap_data(self, adapter_name, layer_name='c_attn', matrix_type='A'):
        """Create data for heatmap visualization"""
        weights = self.load_weights(adapter_name)
        if not weights:
            return None
        
        key = f"base_model.model.{layer_name}.lora_{matrix_type}.weight"
        
        if key not in weights:
            print(f"❌ Key not found: {key}")
            return None
        
        matrix = weights[key].cpu().numpy()
        
        # For large matrices, show a downsampled version
        if matrix.shape[0] > 20 or matrix.shape[1] > 20:
            # Show summary statistics per row/col block
            row_size = max(1, matrix.shape[0] // 10)
            col_size = max(1, matrix.shape[1] // 10)
            
            heatmap_data = np.zeros((10, 10))
            for i in range(10):
                for j in range(10):
                    r_start = i * row_size
                    r_end = min((i+1) * row_size, matrix.shape[0])
                    c_start = j * col_size
                    c_end = min((j+1) * col_size, matrix.shape[1])
                    
                    heatmap_data[i, j] = matrix[r_start:r_end, c_start:c_end].mean()
        else:
            heatmap_data = matrix
        
        return heatmap_data
    
    def print_heatmap_ascii(self, adapter_name, layer_name='c_attn', matrix_type='A'):
        """Print ASCII heatmap of matrix values"""
        heatmap_data = self.create_heatmap_data(adapter_name, layer_name, matrix_type)
        if heatmap_data is None:
            return
        
        print(f"\n{'='*80}")
        print(f"ASCII Heatmap - LoRA {matrix_type} Matrix")
        print(f"Adapter: {adapter_name} / Layer: {layer_name}")
        print(f"{'='*80}\n")
        
        # Normalize for visualization
        data_min = heatmap_data.min()
        data_max = heatmap_data.max()
        normalized = (heatmap_data - data_min) / (data_max - data_min + 1e-8)
        
        # ASCII characters for intensity
        chars = " .:-=+*#%@"
        
        for row in normalized:
            line = ""
            for val in row:
                char_idx = int(val * (len(chars) - 1))
                line += chars[char_idx]
            print(line)
        
        print(f"\n{'-'*80}")

def main():
    """Example usage"""
    print("� LoRA Weight Key Discovery & Viewer\n")
    
    viewer = LoRAKeyDiscovery()
    
    if not viewer.adapters:
        print("❌ No adapters found")
        return
    
    # Pick first adapter
    adapter_name = viewer.adapters[0]['name']
    print(f"Using adapter: {adapter_name}\n")
    
    # 1. Show all keys
    print("1️⃣ All keys in adapter:")
    viewer.show_all_keys(adapter_name)
    
    # 2. Find LoRA A and B
    print("\n2️⃣ Finding LoRA A and B matrices:")
    result = viewer.find_lora_matrices(adapter_name)
    
    if result and result['lora_a_keys']:
        # 3. Display first LoRA A matrix
        first_a_key = result['lora_a_keys'][0]
        print(f"\n3️⃣ Displaying first LoRA A matrix:")
        viewer.display_matrix(adapter_name, first_a_key)
        
        # 4. Show slice
        print(f"\n4️⃣ Showing slice (first 5 rows, first 10 cols):")
        viewer.show_slice(adapter_name, first_a_key, row_range=(0, 5), col_range=(0, 10))
        
        # 5. Export to CSV
        print(f"\n5️⃣ Exporting to CSV:")
        viewer.export_to_csv(adapter_name, first_a_key)
    
    # 6. Compare across adapters
    if len(viewer.adapters) > 1:
        print(f"\n6️⃣ Comparing LoRA A matrices across first 3 adapters:")
        adapter_names = [a['name'] for a in viewer.adapters[:3]]
        viewer.compare_adapters(adapter_names, lora_type='A')

if __name__ == '__main__':
    main()
