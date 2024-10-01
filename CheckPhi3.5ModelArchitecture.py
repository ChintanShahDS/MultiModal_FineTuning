import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Phi-3.5 vision instruct model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, _attn_implementation='eager')

# Print the model architecture
print(model)

# Print the different layers of the model one by one with the layer number
layer_number = 0
for name, module in model.named_modules():
    if len(name) > 0:  # Skip the root module
        print(f"Layer {layer_number}: {name}")
        print(f"  Type: {type(module).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in module.parameters()):,}")
        print()
        layer_number += 1


# Print the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_params:,}")

# Print the model's config
print("\nModel Config:")
print(model.config)
