from inspect_adapters import LoRAInspector

inspector = LoRAInspector()

# View all configs
inspector.print_all_configs()

# Get specific adapter config
config = inspector.get_adapter_config("shakespearelora")
print(config)

# Compare multiple adapters
inspector.compare_adapters(['adapter1_lora', 'adapter2_lora'])

# Check storage usage
inspector.get_model_size_info()