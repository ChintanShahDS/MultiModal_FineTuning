import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Load the Phi-3.5 model
model_name = "microsoft/Phi-3.5-mini-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

phi_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
    # _attn_implementation='eager'
)
phi_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Custom dataset
class ImageTextDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import h5py
        
        row = self.data.iloc[idx]
        image_name = row['image']
        human_text = row['human']
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

        # # Prepare text input
        # text_input = f"Human: {human_text}\nAssistant: {gpt_text}"
        # encoding = self.tokenizer(text_input, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        return {
            "image": image_name,
            "human_text": human_text,
            "gpt_text": gpt_text,
            "image_features": image_features.squeeze()
        }

class ImageTextDatasetForCausalLM(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import h5py
        
        row = self.data.iloc[idx]
        image_name = row['image']
        human_text = row['human']
        gpt_text = row['gpt']

        # Remove extension from image_name
        image_name_without_ext = os.path.splitext(image_name)[0]
        
        # Find the corresponding image embedding file
        image_embeddings_files = [f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')]
        if not image_embeddings_files:
            raise FileNotFoundError(f"No image embedding file found for {image_name}")
        image_embedding_path = os.path.join(self.image_dir, image_embeddings_files[0])
        
        # Load the image embedding
        with h5py.File(image_embedding_path, 'r') as hf:
            image_features = torch.tensor(hf['embeddings'][:])

        # Prepare text input for causal language modeling
        full_text = f"Context: [IMAGE]\nQuestion: {human_text}\nAnswer: {gpt_text}"

        # print("full_text", full_text)
        
        return {
            "image_features": image_features.squeeze(),
            "full_text": full_text
        }

def collate_fn(batch):
    # print("batch", batch)
    image_features = torch.stack([item['image_features'] for item in batch])
    full_texts = [item['full_text'] for item in batch]
    
    # Tokenize the full texts
    encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Create labels for causal language modeling (shift input_ids right)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Set the last token's label to -100 (ignored in loss calculation)
    
    # Set labels to -100 for all tokens before "Answer:" to ignore them in loss calculation
    answer_start = (input_ids == tokenizer.encode("Answer:", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
    for i, start in enumerate(answer_start):
        labels[i, :start] = -100
    # except KeyError as e:
    #     # print(f"KeyError in collate_fn: {e}")
    #     print("Batch contents:")
    #     for i, item in enumerate(batch):
    #         print(f"Item {i}: {item.keys()}")
    #     raise

    return {
        "image_features": image_features,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Usage example:
dataset = ImageTextDatasetForCausalLM("/content/drive/MyDrive/Workspace/llava_150k/conversations.csv", "/content/drive/MyDrive/Workspace/llava_150k/train_embeddings/train_embeddings", tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

class ProjectionBlock(nn.Module):
    def __init__(self, input_dim_CLIP, input_dim_phi2):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim_CLIP)
        self.proj = nn.Sequential(
            nn.Linear(input_dim_CLIP, input_dim_phi2),
            nn.GELU(),
            nn.Linear(input_dim_phi2, input_dim_phi2)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)

# Modify the MultimodalPhiModel class to work with HuggingFace Trainer
class MultimodalPhiModel(nn.Module):
    def __init__(self, phi_model, tokenizer, device, input_dim_CLIP=768, input_dim_phi2=3072):
        super().__init__()
        self.phi_model = phi_model
        # self.image_projection = nn.Linear(input_dim_CLIP, input_dim_phi2, bias=False)
        self.image_projection = ProjectionBlock(input_dim_CLIP, input_dim_phi2)
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        # image_projections = self.resblock(image_projections)
        return image_projections

    def forward(self, input_ids, attention_mask, labels, image_features):
        batch_size = input_ids.shape[0]
        
        # Encode image features
        image_embeddings = self.encode(image_features)
        
        # Get the position of the [IMAGE] token
        image_token_pos = (input_ids == self.tokenizer.encode("[IMAGE]", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
        
        # Replace [IMAGE] token embeddings with encoded image features
        inputs_embeds = self.phi_model.get_input_embeddings()(input_ids)
        for i in range(batch_size):
            inputs_embeds[i, image_token_pos[i]] = image_embeddings[i, 0]  # Assuming image_embeddings has shape (batch_size, 1, embed_dim)
        
        # Forward pass through the language model
        outputs = self.phi_model(inputs_embeds=inputs_embeds, 
                                 attention_mask=attention_mask, 
                                #  labels=labels, 
                                 return_dict=True)
        
        return outputs

def getInputs(question):

    # Prepare text input for causal language modeling
    full_text = f"Context: [IMAGE]\nQuestion: {question}\nAnswer: "
    full_texts = [full_text]
    
    # Tokenize the full texts
    encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

device = torch.device("cpu")
peft_model_location = "/content/drive/MyDrive/Workspace/llava_150k/results/checkpoint-500"

# Usage example:
model = MultimodalPhiModel(phi_model, tokenizer, device)

model.load_adapter(peft_model_location)

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
from PIL import Image

vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
vision_model = CLIPVisionModel.from_pretrained(vision_model_name)

def get_clip_embeddings(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")
    # print("Inputs keys: ", inputs.keys())
    
    # Get the embeddings
    with torch.no_grad():
        outputs = vision_model(**inputs, output_hidden_states=True)
        # print("Outputs: ", outputs)
        # image_feature = feature_select(outputs)
        # print("Image feature shape: ", image_feature.shape)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        # print("Embeddings shape: ", embeddings.shape)
    
    return embeddings.cpu().numpy()  # Move back to CPU for storage

image_path = './data/train2014/COCO_train2014_000000000064.jpg'
embeddings = get_clip_embeddings(image_path)

input_ids, attention_mask = getInputs("Please summarize the information in the data?")

output = model()
print (output)
