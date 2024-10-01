import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import h5py

# Load the Phi-3.5 model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the Phi-3.5 model
phi_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Custom dataset
class ImageTextDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        image_name = row['image']
        # human_text = row['human']
        gpt_text = row['gpt']

        # Remove extension from image_name
        image_name_without_ext = os.path.splitext(image_name)[0]
        
        # Find the corresponding image embedding file
        # print([f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')])
        image_embeddings_files = [f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')]
        if not image_embeddings_files:
            raise FileNotFoundError(f"No image embedding file found for {image_name}")
        image_embedding_path = os.path.join(self.image_dir, image_embeddings_files[0])
        
        # Load the image embedding
        with h5py.File(image_embedding_path, 'r') as hf:
            image_features = torch.tensor(hf['embeddings'][:])

        return {
            "image": image_name,
            "gpt_text": gpt_text,
            "image_features": image_features.squeeze()
        }

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two 3D tensors.
    Computes similarity along 2nd and 3rd dimensions, then averages across the first dimension (batch).
    
    Args:
    vec1, vec2: Tensors of shape (batch_size, seq_len, embedding_dim)
    
    Returns:
    Tensor of shape (batch_size,) containing average cosine similarities
    """
    # Compute cosine similarity along last two dimensions
    # dot_product = torch.sum(q1_embs * q2_embs, axis=1)
    # print("vec1 shape", vec1.shape, "vec2 shape", vec2.shape)
    dot_product = torch.sum(vec1 * vec2, dim=-1)  # Shape: (batch_size, seq_len)
    # print("dot_product shape", dot_product.shape)
    vec1_norm = torch.norm(vec1, dim=-1)  # Shape: (batch_size, seq_len)
    # print("vec1_norm shape", vec1_norm.shape)
    vec2_norm = torch.norm(vec2, dim=-1)  # Shape: (batch_size, seq_len)
    # print("vec2_norm shape", vec2_norm.shape)
    
    # Avoid division by zero
    similarity = dot_product / (vec1_norm * vec2_norm + 1e-8)  # Shape: (batch_size, seq_len)
    # print("similarity shape", similarity.shape)
    # Average across sequence length (2nd dimension)
    avg_similarity = torch.mean(similarity, dim=1)  # Shape: (batch_size,)
    # print("avg_similarity shape", avg_similarity.shape)
    avg_similarity = torch.mean(avg_similarity)  # Shape: (batch_size,)
    # print("avg_similarity shape", avg_similarity.shape)

    return avg_similarity

# class SimpleResBlock(nn.Module):
#     def __init__(self, embed_size, orig_embed_length=50, embed_length=512):
#         super().__init__()
#         self.pre_norm = nn.LayerNorm(embed_size)
#         self.proj = nn.Sequential(
#             nn.Linear(embed_size, embed_size),
#             nn.GELU(),
#             nn.Linear(embed_size, embed_size)
#         )
#         self.increase_dim = nn.Linear(orig_embed_length, embed_length)
#     def forward(self, x):
        
#         # Return the residual connection
#         return x

# Projection model for image embeddings to text embeddings
class ProjectionModel(nn.Module):
    def __init__(self, input_dim_CLIP=768, input_dim_phi2=3072, orig_embed_length=50, embed_length=512):
        super().__init__()

        self.image_projection = nn.Linear(input_dim_CLIP, input_dim_phi2, bias=False)

        self.pre_norm = nn.LayerNorm(input_dim_phi2)
        self.proj = nn.Sequential(
            nn.Linear(input_dim_phi2, input_dim_phi2),
            nn.GELU(),
            nn.Linear(input_dim_phi2, input_dim_phi2)
        )
        self.increase_dim = nn.Linear(orig_embed_length, embed_length)


    def forward(self, image_features):
        # print("forward start")
        image_projections = image_features
        image_projections = self.image_projection(image_projections)

        # Apply layer normalization
        x = self.pre_norm(image_projections)
        
        # Apply projection
        xnew = self.proj(x)
        # print("xnew shape after projection:", xnew.shape)
        
        x = x + xnew
        # print("x shape after addition:", x.shape)

        # Switch between 2nd and 3rd dimension
        x = x.transpose(1, 2)
        # print("x shape after transpose:", x.shape)
        
        # Increase the size of the 3rd dimension (now the 2nd after transpose)
        x = self.increase_dim(x)
        # print("x shape after dimension operations:", x.shape)
        
        # Switch back to original dimension order
        image_projections = x.transpose(1, 2)
        # print("x shape after transpose:", image_projections.shape)

        return image_projections

def embeddings_to_text(embeddings):
    # Convert embeddings to text using the Phi-3.5 model
    with torch.no_grad():
        # Ensure the model is in evaluation mode
        phi_model.eval()
        
        # Use the language model head to get logits
        logits = phi_model.lm_head(embeddings)
        predicted_token_ids = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        return predicted_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the multimodal model
projection_model = ProjectionModel()

batch_size = 10
# Prepare the dataset and dataloader
dataset = ImageTextDataset("./conversations.csv", "./data/train_embeddings")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
projection_model.to(device)
optimizer = torch.optim.AdamW(projection_model.parameters(), lr=5e-5)
print("projection_model", projection_model)
print("project model number of total parameters", sum(p.numel() for p in projection_model.parameters()))

# multimodal_model.eval()
iteration = 0

total_loss = 0
for batch in tqdm(dataloader, desc=f"Dataloader testing"):
    # human_text = batch["human_text"]
    gpt_text = batch["gpt_text"]
    image_features = batch["image_features"].to(device)
    image = batch["image"]

    image_projections = projection_model(image_features)

    loss = 0
    # word_output_pred_tokens = None
    # output_pred_tokens = out_phi[0].argmax(dim=-1)
    gpt_tokens = tokenizer(gpt_text, return_tensors="pt", return_attention_mask=False, padding=True, truncation=True)
    # print("output_pred_tokens shape", output_pred_tokens.shape, "gpt_tokens shape", gpt_tokens.input_ids.shape)
    # Ensure output_pred_tokens and gpt_tokens have the same sequence length
    max_length = max(image_projections.shape[1], gpt_tokens.input_ids.shape[1])
    
    # if output_pred_tokens.shape[1] < max_length:
    #     pad_length = max_length - output_pred_tokens.shape[1]
    #     output_pred_tokens = F.pad(output_pred_tokens, (0, pad_length), value=self.tokenizer.pad_token_id)
    
    if gpt_tokens.input_ids.shape[1] < max_length:
        pad_length = max_length - gpt_tokens.input_ids.shape[1]
        gpt_tokens.input_ids = F.pad(gpt_tokens.input_ids, (0, pad_length), value=tokenizer.pad_token_id)

    phi_model = phi_model.to(device)

    # print("output_pred_tokens shape", output_pred_tokens.shape, "gpt_tokens shape", gpt_tokens.input_ids.shape)

    # output_pred_embeds = self.phi_model.get_input_embeddings()(output_pred_tokens)
    gpt_embeds = phi_model.get_input_embeddings()(gpt_tokens.input_ids.to(device))

    # print("type(output_pred_tokens)", type(output_pred_tokens), "gpt_tokens type", type(gpt_tokens))
    # lossFunction = nn.CosineEmbeddingLoss()
    # target = torch.ones(batch_size)
    cosine_similarity_loss = cosine_similarity(image_projections, gpt_embeds)
    # print("cosine_similarity_loss", cosine_similarity_loss)
    # loss = lossFunction(image_projections.squeeze(), gpt_embeds.squeeze(), torch.tensor([1.0]).to(device))
    loss = 1.0-cosine_similarity_loss
    # loss.requires_grad = True
    # print("loss", loss)

    # if iteration > 20:
    #     break

    loss.backward()
    # print("loss backward done")
    optimizer.step()
    # print("optimizer step done")

    total_loss += loss.item()

    if (iteration % 20) == 0: 
        print("Iteration:", iteration, " Loss:", loss.item())
        print("image", image, "gpt_text", gpt_text)
        print("Predictions:", embeddings_to_text(image_projections))

    iteration += 1

avg_loss = total_loss / len(dataloader)
print(f"Average Loss: {avg_loss:.4f}")

# Save the projection model
# torch.save(projection_model.state_dict(), "./models/projection_model.pth")

# print("Projection model saved successfully.")
