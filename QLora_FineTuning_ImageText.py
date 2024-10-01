import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Load the Phi-3.5 model
model_name = "microsoft/phi-2"
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
    _attn_implementation='eager'
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
            "human_text": human_text,
            "gpt_text": gpt_text,
            "image_features": image_features.squeeze()
        }

class SimpleResBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_size)
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)    

# Multimodal model
class MultimodalPhiModel(nn.Module):
    def __init__(self, phi_model, tokenizer, device, input_dim_CLIP=768, input_dim_phi2=2560):
        super().__init__()
        self.phi_model = phi_model
        print("phi_model.config.hidden_size", self.phi_model.config.hidden_size)
        self.image_projection = nn.Linear(input_dim_CLIP, input_dim_phi2, bias=False)
        self.resblock = SimpleResBlock(input_dim_phi2)
        self.tokenizer = tokenizer
        context = self.tokenizer("Context: ", return_tensors="pt", return_attention_mask=False)
        question = self.tokenizer("\nQuestion: ", return_tensors="pt", return_attention_mask=False)
        answer = self.tokenizer("\nAnswer: ", return_tensors="pt", return_attention_mask=False)
        self.device = device

        self.context_embeddings = self.phi_model.get_input_embeddings()(context.input_ids.to(self.device)).squeeze(0)
        self.question_embeddings = self.phi_model.get_input_embeddings()(question.input_ids.to(self.device)).squeeze(0)
        self.answer_embeddings = self.phi_model.get_input_embeddings()(answer.input_ids.to(self.device)).squeeze(0)

    def forward(self, human_text, gpt_text, image_features):
        print("forward start")
        image_projections = image_features.to(self.device)
        image_projections = self.image_projection(image_projections)
        print("image projections")
        image_projections = self.resblock(image_projections)
        print("resblock done")
        # Uncomment the following line if you need to use the dimension converter
        # image_projections = self.dimension_converter(image_projections)
        batch_size = image_projections.shape[0]

        human_tokens = self.tokenizer(human_text, return_tensors="pt", return_attention_mask=False, padding=True, truncation=True)
        human_embeds = self.phi_model.get_input_embeddings()(human_tokens.input_ids.to(self.device))
        gpt_tokens = self.tokenizer(gpt_text, return_tensors="pt", return_attention_mask=False, padding=True, truncation=True)
        gpt_embeds = self.phi_model.get_input_embeddings()(gpt_tokens.input_ids.to(self.device))
        print("tokens done")
        
        # print("image_projections.shape", image_projections.shape, "gpt_embeds.shape", gpt_embeds.shape, "human_embeds.shape", human_embeds.shape)
        # print("self.context_embeddings.shape", self.context_embeddings.repeat(batch_size,1,1).shape)
        inputs_embeds = torch.cat([self.context_embeddings.repeat(batch_size,1,1), image_projections,
                                    self.question_embeddings.repeat(batch_size,1,1), human_embeds, self.answer_embeddings.repeat(batch_size,1,1)], dim=1)
        
        # # Adjust attention mask for the added image tokens
        # num_image_tokens = image_projections.shape[1]
        # extended_attention_mask = torch.cat([torch.ones((attention_mask.shape[0], num_image_tokens), device=attention_mask.device), attention_mask], dim=1)
        # # Ensure extended_attention_mask matches the sequence length of inputs_embeds
        # if extended_attention_mask.shape[1] != inputs_embeds.shape[1]:
        #     print(f"Adjusting attention mask from {extended_attention_mask.shape[1]} to {inputs_embeds.shape[1]}")
        #     if extended_attention_mask.shape[1] < inputs_embeds.shape[1]:
        #         # Pad the attention mask if it's shorter
        #         padding = torch.ones((attention_mask.shape[0], inputs_embeds.shape[1] - extended_attention_mask.shape[1]), device=attention_mask.device)
        #         extended_attention_mask = torch.cat([extended_attention_mask, padding], dim=1)
        #     else:
        #         # Truncate the attention mask if it's longer
        #         extended_attention_mask = extended_attention_mask[:, :inputs_embeds.shape[1]]
        # print("extended_attention_mask.shape", extended_attention_mask.shape)
        # print("inputs_embeds.shape", inputs_embeds.shape)
        # assert extended_attention_mask.shape[1] == inputs_embeds.shape[1], "Attention mask and input embeddings must have the same sequence length"
        loss = 0
        word_output_pred_tokens = None
        print("gpt_embeds.shape", gpt_embeds.shape)
        for idx in range(gpt_embeds.shape[1]): 
            if (idx % 10) == 0: 
                print("forward loop:", idx)
            out_phi = self.phi_model.base_model.model.model.layers[0](inputs_embeds.to(torch.bfloat16))
            for layer_idx in range(1, 32): 
                out_phi = self.phi_model.base_model.model.model.layers[layer_idx](out_phi[0])
            out_phi = self.phi_model.base_model.model.model.final_layernorm(out_phi[0])
            out_phi = self.phi_model.base_model.model.lm_head(out_phi) ## torch.Size([batch, 55, 50297])
            print("out_phi", out_phi[:,-1,:])
            next_word = torch.argmax(out_phi[:, -1, :], dim=-1) ## [batch]
            print("next_word", next_word)
            
            caption_word_token = gpt_embeds[:,idx].squeeze()
            caption_word_embedding = self.phi_model.get_input_embeddings()(next_word).unsqueeze(1)
            # print("caption_word_embedding done")
            ## instead of append like instruct image output words.. instruct image w1 out, instruct image w2 output ..
            inputs_embeds = torch.cat((inputs_embeds, caption_word_embedding), dim=1)
            caption_word_embedding = caption_word_embedding.squeeze()
            # print("caption_word_token.shape", caption_word_token.shape)
            # print("caption_word_embedding.shape", caption_word_embedding.shape)
            loss_val = F.cross_entropy(caption_word_embedding, caption_word_token)
            # print("loss_val done")
            if loss == 0:
                loss = loss_val
            else:
                loss += loss_val
            print("loss_val", loss_val, "loss", loss)
            
            if word_output_pred_tokens is None: 
                word_output_pred_tokens = next_word.unsqueeze(1) 
            else:
                word_output_pred_tokens = torch.cat((word_output_pred_tokens, next_word.unsqueeze(1)), dim=1)
                
        loss_result = loss/idx
        loss_result.requires_grad = True
        return loss_result, word_output_pred_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the multimodal model
multimodal_model = MultimodalPhiModel(phi_model, tokenizer, device)

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2",
    ]
)

# Apply LoRA to the Phi model part of the multimodal model
multimodal_model.phi_model = get_peft_model(multimodal_model.phi_model, peft_config)
# Get trainable and total parameters
trainable_params = sum(p.numel() for p in multimodal_model.phi_model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in multimodal_model.phi_model.parameters())
trainable_percent = 100 * trainable_params / all_params

print(f"Model Statistics: Trainable parameters: {trainable_params:,}, All parameters: {all_params:,}, Percentage of trainable parameters: {trainable_percent:.2f}%")

# Prepare the dataset and dataloader
dataset = ImageTextDataset("./conversations.csv", "./data/train_embeddings", tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
multimodal_model.to(device)
optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    multimodal_model.train()
    total_loss = 0
    iteration = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        print("batch:", iteration)
        human_text = batch["human_text"]
        gpt_text = batch["gpt_text"]
        image_features = batch["image_features"].to(device)

        # # Ensure inputs are on the correct device
        # human_text = [text.to(device) for text in human_text]
        # gpt_text = [text.to(device) for text in gpt_text]

        # Wrap the forward pass in torch.autograd.set_detect_anomaly(True) for debugging
        with torch.autograd.set_detect_anomaly(True):
            loss, output_pred_tokens = multimodal_model(human_text, gpt_text, image_features)
            print("loss forward done")

            if loss.requires_grad:
                loss.backward()
                print("loss backward done")
                optimizer.step()
                print("optimizer step done")
            else:
                print("Warning: Loss does not require gradient. Check your model architecture.")

        total_loss += loss.item()

        if (iteration % 1000) == 0: 
            print("Iteration:", iteration, " Loss:", loss.item())
            print("Question:", human_text)
            print("Predictions:", tokenizer.batch_decode(output_pred_tokens))
            print("Gt answer:", gpt_text)

        iteration += 1

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
multimodal_model.save_pretrained("./models")
tokenizer.save_pretrained("./models")

print("Fine-tuning completed and model saved.")
