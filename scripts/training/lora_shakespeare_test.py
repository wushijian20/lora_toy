# Access a specific layer's LoRA weights
# Layer 0 is: transformer.h[0].attn.c_attn
lora_a_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_A['default'].weight
lora_b_weight = model.base_model.model.transformer.h[0].attn.c_attn.lora_B['default'].weight

print("LoRA A Matrix Shape:", lora_a_weight.shape) # Should be [rank, input_dim]
print("LoRA A Sample Values:\n", lora_a_weight[:2, :2]) 

print("\nLoRA B Matrix Shape:", lora_b_weight.shape) # Should be [output_dim, rank]
print("LoRA B Sample Values:\n", lora_b_weight[:2, :2])