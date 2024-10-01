import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Optional


# Load the model and processor
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

# # Custom dataset
# class ImageTextDataset(Dataset):
#     def __init__(self, csv_file, image_dir, tokenizer, device, max_length=512):
#         self.data = pd.read_csv(csv_file)
#         self.image_dir = image_dir
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.device = device

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image_name = row['image']
#         human_text = row['human']
#         gpt_text = row['gpt']

#         # Remove extension from image_name
#         image_name_without_ext = os.path.splitext(image_name)[0]
        
#         # Find the corresponding image embedding file
#         # print([f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')])
#         image_files = [os.path.join(self.image_dir,f) for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.jpg')]
#         if not image_files:
#             raise FileNotFoundError(f"No image file found for {image_name}")
#         # image_path = os.path.join(self.image_dir, image_files[0])

#         # Load and preprocess the image
#         # There are 2 options 1) To get from get_image_features or to get from the hidden states
#         # Even in hidden state which state to get it from -1 or -2 is to be tried
#         image = clipprocessor(images=Image.open(image_files[0]), return_tensors="pt")
#         image_forward_out = clipmodel(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#         image_features = image_forward_out.hidden_states[-2]
#         print("image features shape:", image_features.shape)
#         # Generate the embedding
#         # image_features = clipmodel.get_image_features(**image)
        
#         # # Load the image embedding
#         # with h5py.File(image_embedding_path, 'r') as hf:
#         #     image_features = torch.tensor(hf['embeddings'][:])

#         # # Prepare text input
#         # text_input = f"Human: {human_text}\nAssistant: {gpt_text}"
#         # encoding = self.tokenizer(text_input, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

#         return {
#             "image": image_name,
#             "human_text": human_text,
#             "gpt_text": gpt_text,
#             "image_features": image_features
#         }

class ImageTextDatasetForCausalLM(Dataset):
    def __init__(self, clip_processor, csv_file, image_dir, tokenizer, max_length=512):
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

        # Remove extension from image_name
        image_name_without_ext = os.path.splitext(image_name)[0]
        
        # Find the corresponding image embedding file
        # print([f for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.h5')])
        image_files = [os.path.join(self.image_dir,f) for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.jpg')]
        if not image_files:
            raise FileNotFoundError(f"No image file found for {image_name}")
        # image_path = os.path.join(self.image_dir, image_files[0])

        # Load and preprocess the image
        image = self.clip_processor(images=Image.open(image_files[0]), return_tensors="pt")

        # Generate the embedding
        image_features = clipmodel.get_image_features(**image)

        # Start text before putting image embedding
        start_text = f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n"

        # Prepare text input for causal language modeling
        end_text = f"\n{human_text}<|end|>\n<|assistant|>\n{gpt_text}"

        # print("full_text", full_text)
        
        return {
            "image_features": image_features,
            "start_text": start_text,
            "end_text": end_text
        }

def collate_fn(batch):
    # print("batch", batch)
    image_features = torch.stack([item['image_features'] for item in batch])
    start_texts = [item['start_text'] for item in batch]
    end_texts = [item['end_text'] for item in batch]
    
    
    # except KeyError as e:
    #     # print(f"KeyError in collate_fn: {e}")
    #     print("Batch contents:")
    #     for i, item in enumerate(batch):
    #         print(f"Item {i}: {item.keys()}")
    #     raise

    return {
        "start_texts": start_texts,
        "end_texts": end_texts,
        "image_features": image_features,
        # "input_ids": input_ids,
        # "attention_mask": attention_mask,
        # "labels": labels
    }

# Usage example:
dataset = ImageTextDatasetForCausalLM(clipprocessor, "./conversations.csv", "./data/train2014", tokenizer)
train_set, val_set = torch.utils.data.random_split(dataset, [0.9,0.1])
# print(dataset[0])
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #, collate_fn=collate_fn

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
    def __init__(self, phi_model, tokenizer, projection):
        super().__init__(phi_model.config)
        self.phi_model = phi_model
        self.image_projection = projection
        self.tokenizer = tokenizer

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_texts, end_texts, image_features):
        # print("tokenizer bos_token_id", self.tokenizer.bos_token_id, "tokenizer eos_token", self.tokenizer.eos_token,
        #       "tokenizer pad_token_id", self.tokenizer.pad_token_id, "tokenizer sep_token_id", self.tokenizer.sep_token_id,
        #       "tokenizer cls_token_id", self.tokenizer.cls_token_id, "tokenizer mask_token_id", self.tokenizer.mask_token_id,
        #       "tokenizer unk_token_id", self.tokenizer.unk_token_id)
        batch_size = image_features.shape[0]
        num_image_tokens = image_features.shape[1]
        # print("batch_size:", batch_size)
        # print("num_image_tokens:", num_image_tokens)
        
        # print("image features shape:", image_features.shape)
        # Encode image features
        image_embeddings = self.encode(image_features).to(self.device)
        image_tokens = torch.full((batch_size, num_image_tokens), -100, dtype=torch.long).to(self.device)

        # Tokenize the full texts
        start_tokens = self.tokenizer(start_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        end_tokens = self.tokenizer(end_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # print(f"start_encodings shape: {start_encodings['input_ids'].shape}, end_encodings shape: {end_encodings['input_ids'].shape}")
        
        start_input_ids = start_tokens['input_ids'].to(self.device)
        start_attention_mask = start_tokens['attention_mask'].to(self.device)
        end_input_ids = end_tokens['input_ids'].to(self.device)
        end_attention_mask = end_tokens['attention_mask'].to(self.device)

        start_encodings = self.phi_model.get_input_embeddings()(start_input_ids)
        # print("start_encodings shape:", start_encodings.shape)
        # print("start_input_ids type:", type(start_input_ids), "image_tokens type:", type(image_tokens))
        # print(f"start_input_ids shape: {start_input_ids.shape}, image_tokens shape: {image_tokens.shape}, end_input_ids shape: {end_input_ids.shape}")
        input_ids = torch.cat([start_input_ids,image_tokens,end_input_ids], dim=1)
        attention_mask = torch.cat([start_attention_mask, torch.ones((batch_size, num_image_tokens), dtype=torch.long).to(self.device), end_attention_mask], dim=1)

        # Replace [IMAGE] token embeddings with encoded image features
        # for i in range(batch_size):
        #     inputs_embeds[i, image_token_pos[i]] = image_embeddings[i, 0]  # Assuming image_embeddings has shape (batch_size, 1, embed_dim)

        # Create labels for causal language modeling (shift input_ids right)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Set the last token's label to -100 (ignored in loss calculation)
        
        # Set labels to -100 for all tokens before "Answer:" to ignore them in loss calculation
        answer_start = (input_ids == tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
        answer_start_labels = (labels == tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]).nonzero(as_tuple=True)[1]
        for i, start in enumerate(answer_start):
            labels[i, :start] = -100
        # print("answer_start", answer_start, "answer_start_labels", answer_start_labels)
        # print("assistant token:", tokenizer.encode("<|assistant|>", add_special_tokens=False)[0])
        # print("input_ids:", input_ids)
        # print("labels:", labels)

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

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, *model_args, debug=False, **kwargs):

        model_name = "microsoft/Phi-3.5-mini-instruct"
        base_phi_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # phi_path = os.path.join(pretrained_model_name_or_path, "phi_model")
        phi_path = pretrained_model_name_or_path

        # Save the base model
        model = PeftModel.from_pretrained(base_phi_model, phi_path)
        phi_model = model.merge_and_unload()

        # # Load the base Phi-3 model
        # phi_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        input_dim = 512
        output_dim = 3072

        # Load the projector weights
        # projector_path = os.path.join(pretrained_model_name_or_path, "projection_layer", "image_projector.pth")
        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi_model.device)

            projector = ProjectionBlock(input_dim, output_dim)

            # Try to load the state dict, ignoring mismatched keys
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            # input_dim = 512  # Default CLIP embedding size
            output_dim = phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        # Create and return the Phi3WithProjector instance
        model = self(phi_model, tokenizer, projector)
        return model

    def save_pretrained(self, save_directory):
        # Load the Phi-3.5 model
        self.phi_model.save_pretrained(save_directory)
        # model_name = "microsoft/Phi-3.5-mini-instruct"
        # base_phi_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        # )
        # # Save the base model
        # model = PeftModel.from_pretrained(base_phi_model, self.phi_model)
        # model = model.merge_and_unload()
        # model.save_pretrained(save_directory)

        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.image_projection.state_dict(), projector_path)

        # Save the config
        self.config.save_pretrained(save_directory)

# Define a custom Trainer to handle the multimodal input

class MultimodalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        image_features = inputs.pop("image_features")
        outputs = model(**inputs, image_features=image_features)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    #     run_dir = self._get_output_dir(trial=trial)
    #     output_dir = os.path.join(run_dir, checkpoint_folder)

    #     # Save the projection layer separately
    #     projection_layer_path = os.path.join(output_dir, "projection_layer")
    #     os.makedirs(projection_layer_path, exist_ok=True)
    #     self.model.image_projection._save_checkpoint()
    #     torch.save(self.model.image_projection.state_dict(), os.path.join(projection_layer_path, "pytorch_model.bin"))

    #     # Save the Phi-3.5 QLoRA weights separately
    #     phi_model_path = os.path.join(output_dir, "phi_model")
    #     os.makedirs(phi_model_path, exist_ok=True)
    #     self.model.phi_model.save_pretrained(phi_model_path)

    #     # Save the tokenizer
    #     self.model.tokenizer.save_pretrained(output_dir)

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

# Usage example:
model = MultimodalPhiModel(phi_model, tokenizer, device)

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

# # Initialize the multimodal model
# multimodal_model = MultimodalPhiModel(phi_model, tokenizer, device)

# for param in model.parameters():
#     param.requires_grad = False

# for param in model.image_projection.parameters():
#     param.requires_grad = True

# for param in model.resblock.parameters():
#     param.requires_grad = True

# model.gradient_checkpointing_enable()
# print(model)

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

# model.gradient_checkpointing_enable()

trainer = MultimodalTrainer(
    model=model,
    args=TrainingArguments(output_dir="./results",
                           do_train=True,
                           do_eval=True,
                           num_train_epochs=1, 
                           per_device_train_batch_size=2, 
                           remove_unused_columns=False,
                           max_steps = 1,
                           save_steps = 250,
                           logging_steps = 250,
                           eval_steps = 250,
                           save_total_limit = 2,
                           bf16=True),
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,
)

# Start training
trainer.train()
model.save_pretrained("./models")
# tokenizer.save_pretrained("./models")

# # Get trainable and total parameters
# trainable_params = sum(p.numel() for p in multimodal_model.phi_model.parameters() if p.requires_grad)
# all_params = sum(p.numel() for p in multimodal_model.phi_model.parameters())
# trainable_percent = 100 * trainable_params / all_params

# print(f"Model Statistics: Trainable parameters: {trainable_params:,}, All parameters: {all_params:,}, Percentage of trainable parameters: {trainable_percent:.2f}%")

# # Prepare the dataset and dataloader
# dataset = ImageTextDataset("./conversations.csv", "./data/train_embeddings", tokenizer)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

   
# # Training loop
# multimodal_model.to(device)
# optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=5e-5)

# # multimodal_model.eval()
# iteration = 0

# print("multimodal_model", multimodal_model)



# def get_cosing_embeddings(q1_embs, q2_embs):
#     return torch.sum(q1_embs * q2_embs, axis=1)

# text1 = "What is the capital of France?"
# text2 = "Germany is a country in Europe."
# text3 = "Paris is the capital of France."
# text4 = "The weather is great today."
# text5 = "AI is the future."
# texts = [text1, text2, text3, text4, text5]

# text_tokens = multimodal_model.tokenizer(texts, return_tensors="pt", max_length=10, return_attention_mask=False, padding=True, add_special_tokens = True)
# text_embs = multimodal_model.phi_model.get_input_embeddings()(text_tokens.input_ids.to(device))
# print("text_embs shape", text_embs.shape)

# text1_tokens = multimodal_model.tokenizer(text1, return_tensors="pt", max_length=10, return_attention_mask=False, padding=True, add_special_tokens = True)
# text2_embs = multimodal_model.phi_model.get_input_embeddings()(text2_tokens.input_ids.to(device))
# text3_tokens = multimodal_model.tokenizer(text3, return_tensors="pt", max_length=10, return_attention_mask=False, padding=True, add_special_tokens = True)
# text3_embs = multimodal_model.phi_model.get_input_embeddings()(text3_tokens.input_ids.to(device))
# text4_tokens = multimodal_model.tokenizer(text4, return_tensors="pt", max_length=10, return_attention_mask=False, padding=True, add_special_tokens = True)
# text4_embs = multimodal_model.phi_model.get_input_embeddings()(text4_tokens.input_ids.to(device))
# text5_tokens = multimodal_model.tokenizer(text5, return_tensors="pt", max_length=10, return_attention_mask=False, padding=True, add_special_tokens = True)
# text5_embs = multimodal_model.phi_model.get_input_embeddings()(text5_tokens.input_ids.to(device))

# print("text_embs shape", text_embs.shape)
# # print("text2_embs shape", text2_embs.shape)
# # print("text3_embs shape", text3_embs.shape)
# # print("text4_embs shape", text4_embs.shape)
# # print("text5_embs shape", text5_embs.shape)

# text1_text2_cosine_sim = get_cosing_embeddings(text_embs[0], text_embs[1])
# print("text1_text2_cosine_sim", torch.mean(text1_text2_cosine_sim))
# text1_text3_cosine_sim = get_cosing_embeddings(text_embs[0], text_embs[2])
# print("text1_text3_cosine_sim", torch.mean(text1_text3_cosine_sim)) 
# text1_text4_cosine_sim = get_cosing_embeddings(text_embs[0], text_embs[3])
# print("text1_text4_cosine_sim", torch.mean(text1_text4_cosine_sim)) 
# text1_text5_cosine_sim = get_cosing_embeddings(text_embs[0], text_embs[4])
# print("text1_text5_cosine_sim", torch.mean(text1_text5_cosine_sim)) 

# loss = nn.CosineEmbeddingLoss()

# print("text1_text2_loss", loss(text_embs[0], text_embs[1], torch.tensor([1.0]).to(device)))
# print("text1_text3_loss", loss(text_embs[0], text_embs[2], torch.tensor([1.0]).to(device)))
# print("text1_text4_loss", loss(text_embs[0], text_embs[3], torch.tensor([1.0]).to(device)))
# print("text1_text5_loss", loss(text_embs[0], text_embs[4], torch.tensor([1.0]).to(device)))


# total_loss = 0
# for batch in tqdm(dataloader, desc=f"Dataloader testing"):
#     iteration += 1
#     human_text = batch["human_text"]
#     gpt_text = batch["gpt_text"]
#     image_features = batch["image_features"].to(device)
#     image = batch["image"]
#     # # Ensure inputs are on the correct device
#     # human_text = [text.to(device) for text in human_text]
#     # gpt_text = [text.to(device) for text in gpt_text]

#     out_phi, output_pred_tokens, loss = multimodal_model(human_text, gpt_text, image_features)
#     # print("out_phi", out_phi)
#     # print("out_phi[0].shape", out_phi[0].shape)
#     text = tokenizer.batch_decode(output_pred_tokens)
#     # if (iteration % 1000) == 0: 
#     #     print("iteration", iteration)
#     #     print("image", image, "human_text", human_text, "gpt_text", gpt_text)
#     #     print("text:", text)

#     # if iteration > 2:
#     #     break

#     loss.backward()
#     # print("loss backward done")
#     optimizer.step()
#     # print("optimizer step done")

#     total_loss += loss.item()

#     if (iteration % 1000) == 0: 
#         print("Iteration:", iteration, " Loss:", loss.item())
#         print("image", image, "human_text", human_text, "gpt_text", gpt_text)
#         print("Predictions:", tokenizer.batch_decode(output_pred_tokens))

# avg_loss = total_loss / len(dataloader)
# print(f"Average Loss: {avg_loss:.4f}")

# # Save the fine-tuned model
# # multimodal_model.save_pretrained("./models")
# # tokenizer.save_pretrained("./models")

# print("Testing completed.")


