import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Load the model and processor
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

max_length = 512

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
    # _attn_implementation='eager'
)
phi_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Ensure you have downloaded the stopwords and punkt packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    newtext = ' '.join(filtered_text.split())
    return newtext

def remove_punctuation(text):
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

def preprocess_text(text):
    text_no_punct = remove_punctuation(text)
    text_no_stopwords = remove_stopwords(text_no_punct)
    return text_no_stopwords

class ImageTextDatasetForCausalLM(Dataset):
    def __init__(self, clip_processor, csv_file, image_dir, tokenizer, max_length=max_length):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        row = self.data.iloc[idx]
        image_name = row['image']
        human_text = row['human']
        gpt_text = row['gpt']

        gpt_text_st = preprocess_text(gpt_text)
        # print("Before gpt_text:", gpt_text, "After gpt_text_st:", gpt_text_st)

        # Remove extension from image_name
        image_name_without_ext = os.path.splitext(image_name)[0]
        
        # Find the corresponding image embedding file
        # print([f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')])
        image_files = [os.path.join(self.image_dir,f) for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.jpg')]
        if not image_files:
            raise FileNotFoundError(f"No image file found for {image_name}")

        image = self.clip_processor(images=Image.open(image_files[0]), return_tensors="pt")

        # Generate the embedding
        image_features = clipmodel.get_image_features(**image)

        # Start text before putting image embedding
        start_text = f"<|system|>\nYou are a helpful assistant good at answering questions based on the given context.<|end|>\n<|user|>\n"

        # Prepare text input for causal language modeling
        end_text = f"\n{human_text}<|end|>\n<|assistant|>\n{gpt_text}"
        # print("start_text:", start_text, "end_text:", end_text)

        # print("full_text", full_text)
        
        return {
            "image_features": image_features,
            "start_text": start_text,
            "end_text": end_text
        }
    
    def shuffle(self, seed=None):
        self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        return

def collate_fn(batch):
    # print("batch", batch)
    image_features = torch.stack([item['image_features'] for item in batch])
    start_texts = [item['start_text'] for item in batch]
    end_texts = [item['end_text'] for item in batch]

    batch_size = image_features.shape[0]
    num_image_tokens = image_features.shape[1]
    # print("batch_size:", batch_size)
    # print("num_image_tokens:", num_image_tokens)
    
    # print("image features shape:", image_features.shape)
    # Encode image features
    image_tokens = torch.full((batch_size, num_image_tokens), -100, dtype=torch.long)

    # Tokenize the full texts
    start_tokens = tokenizer(start_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    end_tokens = tokenizer(end_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # print(f"start_encodings shape: {start_encodings['input_ids'].shape}, end_encodings shape: {end_encodings['input_ids'].shape}")
    
    start_input_ids = start_tokens['input_ids']
    start_attention_mask = start_tokens['attention_mask']
    end_input_ids = end_tokens['input_ids']
    end_attention_mask = end_tokens['attention_mask']
    # print(f"start attention mask: {start_attention_mask}, end attention mask: {end_attention_mask}")

    # print("start_input_ids type:", type(start_input_ids), "image_tokens type:", type(image_tokens))
    # print(f"start_input_ids shape: {start_input_ids.shape}, image_tokens shape: {image_tokens.shape}, end_input_ids shape: {end_input_ids.shape}")
    input_ids = torch.cat([start_input_ids,image_tokens,end_input_ids], dim=1)
    attention_mask = torch.cat([start_attention_mask, torch.ones((batch_size, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)

    # Create labels for causal language modeling (shift input_ids right)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Set the last token's label to -100 (ignored in loss calculation)
    
    # Set labels to -100 for all tokens before "Answer:" to ignore them in loss calculation
    answer_start = (input_ids == tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
    for i, start in enumerate(answer_start):
        labels[i, :start] = -100

    return {
        "start_input_ids": start_input_ids,
        "end_input_ids": end_input_ids,
        "image_features": image_features,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Usage example:
dataset = ImageTextDatasetForCausalLM(clipprocessor, "./conversations.csv", "./data/train2014", tokenizer)
# print("Dataset length:", len(dataset))
# print("Dataset[0]:", dataset[0])
dataset.shuffle(seed=42)
# print("After Dataset length:", len(dataset))
# print("Dataset[0]:", dataset[0])
train_set, val_set = torch.utils.data.random_split(dataset, [0.9,0.1])

class ProjectionBlock(nn.Module):
    def __init__(self, input_dim_CLIP, input_dim_phi):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim_CLIP)
        self.proj = nn.Sequential(
            nn.Linear(input_dim_CLIP, input_dim_phi),
            nn.GELU(),
            nn.Linear(input_dim_phi, input_dim_phi)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)

# Modify the MultimodalPhiModel class to work with HuggingFace Trainer
class MultimodalPhiModel(PreTrainedModel):

    def gradient_checkpointing_enable(self, **kwargs):
        self.phi_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi_model.gradient_checkpointing_disable()

    def __init__(self, phi_model, tokenizer, projection):
        super().__init__(phi_model.config)
        self.phi_model = phi_model
        self.image_projection = projection
        self.tokenizer = tokenizer
        # self.device = device

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, *model_args, debug=False, **kwargs):

        model_name = "microsoft/Phi-3.5-mini-instruct"
        base_phi_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Save the base model
        model = PeftModel.from_pretrained(base_phi_model, pretrained_model_name_or_path)
        phi_model = model.merge_and_unload()

        input_dim = 512
        output_dim = 3072

        # Load the projector weights
        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi_model.device)

            projector = ProjectionBlock(input_dim, output_dim)

            # Try to load the state dict, ignoring mismatched keys
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            output_dim = phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        # Create and return the Phi3WithProjector instance
        model = self(phi_model, projector, debug=debug)
        return model

    def save_pretrained(self, save_directory):
        # Load the Phi-3.5 model
        self.phi_model.save_pretrained(save_directory)

        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.image_projection.state_dict(), projector_path)

        # Save the config
        self.config.save_pretrained(save_directory)

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_input_ids, end_input_ids, image_features, attention_mask, labels):
        # print("tokenizer bos_token_id", self.tokenizer.bos_token_id, "tokenizer eos_token", self.tokenizer.eos_token,
        #       "tokenizer pad_token_id", self.tokenizer.pad_token_id, "tokenizer sep_token_id", self.tokenizer.sep_token_id,
        #       "tokenizer cls_token_id", self.tokenizer.cls_token_id, "tokenizer mask_token_id", self.tokenizer.mask_token_id,
        #       "tokenizer unk_token_id", self.tokenizer.unk_token_id)
        # device = next(self.parameters()).device

        # Encode image features
        image_embeddings = self.encode(image_features)

        start_embeds = self.phi_model.get_input_embeddings()(start_input_ids)
        end_embeds = self.phi_model.get_input_embeddings()(end_input_ids)
        # print("start_embeds shape:", start_embeds.shape, "image_embeddings shape:", image_embeddings.shape, "end_embeds shape:", end_embeds.shape)
        input_embeds = torch.cat([start_embeds, image_embeddings, end_embeds], dim=1)
        # print("Input Embeds shape:", input_embeds.shape, "attention_mask shape:", attention_mask.shape, "labels shape:", labels.shape)

        # Forward pass through the language model
        outputs = self.phi_model(inputs_embeds=input_embeds, 
                                 attention_mask=attention_mask, 
                                 labels=labels, 
                                 return_dict=True)
        
        return outputs

# Define a custom Trainer to handle the multimodal input

class MultimodalTrainer(Trainer):

    def printOutput(self, outputs):
        tokens = outputs.logits.argmax(dim=-1)
        # print("Tokens type:", type(tokens))
        # print("Tokens len:", len(tokens))
        output = self.tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
            )
        # print("Output:", output)

    def printLabels(self, labels):
        # print("Labels type:", type(labels))
        # print("Labels len:", len(labels))
        output = self.tokenizer.decode(
            labels[0],
            skip_special_tokens=True
            )
        # print("Labels:", output)

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(" In compute loss method")
        # print("inputs:", inputs)
        # print("Input tokens:", outputs.logits.argmax(dim=-1))

        # if "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        outputs = model(**inputs)
        # print("Output tokens:", outputs.logits.argmax(dim=-1))
        # print("Label tokens:", labels)
        # self.printOutput(outputs)
        # if labels is not None:
        #     self.printLabels(labels)
        # else:
        #     print("Labels are None:", labels)

        # outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        # print("Loss:", loss)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Save the projection layer separately
        projection_layer_path = os.path.join(output_dir, "projection_layer")
        os.makedirs(projection_layer_path, exist_ok=True)
        torch.save(self.model.image_projection.state_dict(), os.path.join(projection_layer_path, "pytorch_model.bin"))

        # Save the Phi-3.5 QLoRA weights separately
        phi_model_path = os.path.join(output_dir, "phi_model")
        os.makedirs(phi_model_path, exist_ok=True)
        self.model.phi_model.save_pretrained(phi_model_path)

        # Save the tokenizer
        self.model.tokenizer.save_pretrained(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained_model_path = "./Projmodel"
input_dim = 512
output_dim = 3072
projector = ProjectionBlock(input_dim, output_dim)
# projector_path = os.path.join(pretrained_model_path, "image_projector.pth")
# if os.path.exists(projector_path):
#     projector_state_dict = torch.load(projector_path, map_location=phi_model.device)

#     projector = ProjectionBlock(input_dim, output_dim)

#     # Try to load the state dict, ignoring mismatched keys
#     projector.load_state_dict(projector_state_dict, strict=False)
#     print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")

# Usage example:
model = MultimodalPhiModel(phi_model, tokenizer, projector)

# for param in model.parameters():
#     param.requires_grad = False

# for param in model.image_projection.parameters():
#     param.requires_grad = True

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "o_proj",
        "qkv_proj",
        "gate_up_proj",
        "down_proj",
        # "fc1",
        # "fc2",
    ]
)

# # Apply LoRA to the Phi model part of the multimodal model
model.phi_model = get_peft_model(model.phi_model, peft_config)
# Enable gradient checkpointing for the model
model.gradient_checkpointing_enable()

# Print the names of the layers whose parameters are trainable
# trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
# print("Trainable layers:")
# for layer in trainable_layers:
#     print(layer)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / all_params

print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")
print(f"Percentage of trainable parameters: {trainable_percent:.2f}%")

trainer = MultimodalTrainer(
    model=model,
    args=TrainingArguments(output_dir="./MM_FT_C1_results",
                           do_train=True,
                           do_eval=True,
                           num_train_epochs=1, 
                           per_device_train_batch_size=2,
                           remove_unused_columns=False,
                           max_steps = 6000,
                           save_steps = 0.4,
                           logging_steps = 0.1,
                           eval_steps = 0.1,
                           save_total_limit = 2,
                           bf16=True),
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
model.save_pretrained("./MM_FT_C1")
# tokenizer.save_pretrained("./models")
