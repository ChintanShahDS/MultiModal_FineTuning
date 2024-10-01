
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from LLaVA.llava.model.multimodal_projector.builder import build_vision_projector

# Load the Phi model
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
    trust_remote_code=True
)
phi_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load Whisper model
# whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

class MultimodalModel(nn.Module):
    def __init__(self, phi_model, clip_hidden_size, phi_hidden_size):
        super().__init__()
        self.phi_model = phi_model
        self.clip_projector = build_vision_projector(
            config=type('Config', (), {'mm_hidden_size': clip_hidden_size, 'hidden_size': phi_hidden_size, 'mm_projector_type': 'mlp2x_gelu'})()
        )
        # self.whisper_projector = nn.Linear(whisper_hidden_size, phi_hidden_size)

    def forward(self, input_ids, attention_mask, clip_embeddings=None):
        embeddings_list = []
        
        if clip_embeddings is not None:
            projected_clip = self.clip_projector(clip_embeddings)
            embeddings_list.append(projected_clip)
        
        text_embeddings = self.phi_model.get_input_embeddings()(input_ids)
        embeddings_list.append(text_embeddings)
        
        combined_embeddings = torch.cat(embeddings_list, dim=1)
        outputs = self.phi_model(inputs_embeds=combined_embeddings, attention_mask=attention_mask)
        return outputs

# Initialize the multimodal model
clip_hidden_size = 768  # Adjust this based on your CLIP model
phi_hidden_size = phi_model.config.hidden_size
multimodal_model = MultimodalModel(phi_model, clip_hidden_size, phi_hidden_size)

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
    ]
)

# Apply LoRA to the Phi model part of the multimodal model
multimodal_model.phi_model = get_peft_model(multimodal_model.phi_model, peft_config)

# Freeze the CLIP and Whisper projectors
for param in multimodal_model.clip_projector.parameters():
    param.requires_grad = False

# # Pipeline to process audio, image, and text inputs
# def multimodal_pipeline(audio_input=None, image_input=None, text_input=""):
#     inputs = []
    
#     # # Process audio input with Whisper
#     # if audio_input is not None:
#     #     audio_features = whisper_processor(audio_input, return_tensors="pt").input_features
#     #     whisper_output = whisper_model.generate(audio_features)
#     #     whisper_embeddings = whisper_model.get_encoder()(audio_features).last_hidden_state
#     #     inputs.append(("whisper", whisper_embeddings))
    
#     # Process image input with CLIP (assuming you have a function to get CLIP embeddings)
#     if image_input is not None:
#         clip_embeddings = get_clip_embeddings(image_input)  # You need to implement this function
#         inputs.append(("clip", clip_embeddings))
    
#     # Process text input
#     prompt = f"Based on the provided image and audio, please answer the following question or provide information: {text_input}"
#     encoded_text = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#     input_ids = encoded_text.input_ids
#     attention_mask = encoded_text.attention_mask
    
#     # Prepare inputs for the multimodal model
#     model_inputs = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask
#     }
#     for input_type, embeddings in inputs:
#         if input_type == "whisper":
#             model_inputs["whisper_embeddings"] = embeddings
#         elif input_type == "clip":
#             model_inputs["clip_embeddings"] = embeddings
    
#     # Generate output from the multimodal model
#     with torch.no_grad():
#         output = multimodal_model(**model_inputs)
    
#     generated_text = tokenizer.decode(output.logits.argmax(dim=-1)[0], skip_special_tokens=True)
#     return generated_text

# Example usage of the pipeline
# audio_input = load_audio("path/to/audio.wav")  # You need to implement this function
# image_input = load_image("path/to/image.jpg")  # You need to implement this function
# text_input = "What is the main object in the image and what is being said in the audio?"
# result = multimodal_pipeline(audio_input, image_input, text_input)
# print(result)


# Training loop for image-to-text data using CLIP embeddings

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

class ImageTextDataset(Dataset):
    def __init__(self, clip_embeddings_file, tokenizer, max_length=512):
        with open(clip_embeddings_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_embedding = torch.tensor(item['embedding'])
        text = item['caption']  # Assuming there's a 'caption' field in your JSON
        
        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, 
                                      max_length=self.max_length, return_tensors='pt')
        
        return {
            'clip_embeddings': image_embedding,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze()
        }

# Hyperparameters
batch_size = 8
learning_rate = 2e-5
num_epochs = 3

# Create dataset and dataloader
dataset = ImageTextDataset('./data/clip_embeddings.json', tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(multimodal_model.parameters(), lr=learning_rate)

# Training loop
multimodal_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'clip_embeddings': batch['clip_embeddings'].to(device)
        }
        
        outputs = multimodal_model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the trained model
multimodal_model.save_pretrained("./trained_multimodal_model")
tokenizer.save_pretrained("./trained_multimodal_model")

print("Training completed and model saved.")
