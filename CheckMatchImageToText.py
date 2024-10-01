import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import h5py

# Load the Phi model
model_name = "microsoft/phi-2"  # or whichever Phi model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
phi_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# # Define the projection layer
# class CLIPProjection(nn.Module):
#     def __init__(self, clip_hidden_size, phi_hidden_size):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(clip_hidden_size, phi_hidden_size),
#             nn.ReLU(),
#             nn.Linear(phi_hidden_size, phi_hidden_size)
#         )
    
#     def forward(self, clip_embeddings):
#         return self.projection(clip_embeddings)

# # Initialize the projection layer
# clip_hidden_size = 768  # This should match the size of your CLIP embeddings
# phi_hidden_size = phi_model.config.hidden_size
# clip_projection = CLIPProjection(clip_hidden_size, phi_hidden_size)

# # Function to load CLIP embeddings from h5 file
# def load_clip_embeddings(h5_file_path):
#     with h5py.File(h5_file_path, 'r') as f:
#         embeddings = torch.tensor(f['embeddings'][:])
#     return embeddings

# # Example usage
# h5_file_path = "./data/train_embeddings/example_image.h5"  # Replace with your actual file path
# clip_embeddings = load_clip_embeddings(h5_file_path)

# # Project CLIP embeddings
# projected_embeddings = clip_projection(clip_embeddings)

# # Now you can use these projected embeddings as input to your Phi model
# # For example:
# # phi_outputs = phi_model(inputs_embeds=projected_embeddings)

# print(f"Original CLIP embedding shape: {clip_embeddings.shape}")
# print(f"Projected embedding shape: {projected_embeddings.shape}")
print(f"Phi model input size: {phi_model.config.hidden_size}")
# Function to load CLIP embeddings from h5 file
def load_clip_embeddings(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        embeddings = torch.tensor(f['embeddings'][:])
    return embeddings

# Function to convert embeddings to tokens
def embeddings_to_tokens(embeddings, phi_model, tokenizer, max_length=512):
    # Flatten the embeddings
    flattened = embeddings.view(-1)
    
    # Normalize the embeddings
    normalized = (flattened - flattened.mean()) / flattened.std()
    
    # Scale to the range of token IDs
    scaled = (normalized * (len(tokenizer) / 2)).long() + len(tokenizer) // 2
    
    # Clip values to be within the valid token ID range
    clipped = torch.clamp(scaled, 0, len(tokenizer) - 1)
    
    # Truncate or pad to max_length
    if len(clipped) > max_length:
        clipped = clipped[:max_length]
    else:
        clipped = torch.nn.functional.pad(clipped, (0, max_length - len(clipped)))
    
    return clipped

# Example usage
h5_file_path = "./data/train_embeddings/COCO_train2014_000000000009.h5"  # Replace with your actual file path
clip_embeddings = load_clip_embeddings(h5_file_path)

# Convert embeddings to tokens
tokens = embeddings_to_tokens(clip_embeddings, phi_model, tokenizer)

print(f"Original CLIP embedding shape: {clip_embeddings.shape}")
print(f"Converted tokens shape: {tokens.shape}")

# Now you can use these tokens as input to your Phi model
# For example:
# phi_outputs = phi_model(input_ids=tokens.unsqueeze(0))

# Optionally, you can decode the tokens back to text to see what they represent
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text from tokens: {decoded_text}")
