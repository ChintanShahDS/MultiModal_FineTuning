import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import os
from transformers import CLIPProcessor, CLIPModel
from typing import Optional

from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, phi_model, tokenizer, device, input_dim_CLIP=512, input_dim_phi2=3072):
        super().__init__()
        self.phi_model = phi_model
        # if proj_model:
        #     self.image_projection = proj_model
        # else:
        self.image_projection = ProjectionBlock(input_dim_CLIP, input_dim_phi2)
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_text, end_text, attention_mask, labels, image_features):
        # batch_size = input_ids.shape[0]
        
        # Encode image features
        image_embeddings = self.encode(image_features).unsqueeze(0).bfloat16().to(self.device)

        # Tokenize the full texts
        tokens = tokenizer(start_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        start_tokens = tokens['input_ids']
        tokens = tokenizer(end_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        end_tokens = tokens['input_ids']

        # attention_mask = encodings['attention_mask']

        start_embeds = self.phi_model.get_input_embeddings()(start_tokens)
        # print("shape start_embeds:", start_embeds.shape)
        end_embeds = self.phi_model.get_input_embeddings()(end_tokens)
        # print("shape end_embeds:", end_embeds.shape)
        # print("image token value:", input_ids[0,image_token_pos[0]])
        # print("type start_embeds:", start_embeds.dtype)
        # print("type end embeds:", end_embeds.dtype)

        # for i in range(batch_size):
        #     print("Inside batchsize:", i)
        # print(image_embeddings.shape)
        # print(start_embeds.shape)
        # print(end_embeds.shape)
        input_embeds = torch.concat([start_embeds,image_embeddings,end_embeds],dim=1)

        # print(input_embeds.shape)
        # for i in range(batch_size):
        # inputs_embeds[0,image_token_pos] = image_embeddings[0]  # Assuming image_embeddings has shape (batch_size, 1, embed_dim)
        
        # Forward pass through the language model
        outputs = self.phi_model(inputs_embeds=input_embeds, 
                                #  attention_mask=attention_mask, 
                                #  labels=labels, 
                                 return_dict=True)
        
        return outputs

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

def getInputs(question, answer=""):

    # Prepare text input for causal language modeling
    starting_text = "<|system|> You are a helpful assistant.<|end|> <|user|> Context:"
    end_text = f"Question: {question} Answer: {answer}"
    # full_text = f"<|system|> You are a helpful assistant.<|end|> <|user|> Context: <|image_1|> Question: {question} Answer: {answer}"
    # print("full_text:", full_text)
    # full_texts = full_text

    # # Tokenize the full texts
    # encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # input_ids = encodings['input_ids']
    # attention_mask = encodings['attention_mask']

    return starting_text, end_text

# def getInputs(question):

#     # Prepare text input for causal language modeling
#     full_text = f"Context: [IMAGE]\nQuestion: {question}\nAnswer: "
#     full_texts = [full_text]
    
#     # Tokenize the full texts
#     encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
#     input_ids = encodings['input_ids']
#     attention_mask = encodings['attention_mask']
    
#     return input_ids, attention_mask

from accelerate import load_checkpoint_and_dispatch

peft_model_location = "./results/checkpoint-3000/phi_model"
proj_model_location = "./results/checkpoint-3000/projection_layer/pytorch_model.bin"


# model = load_checkpoint_and_dispatch(
#     model, checkpoint=model_location, device_map="auto"
# )
# proj_model = ProjectionBlock(input_dim_CLIP=512, input_dim_phi2=3072).to(device)

# Usage example:
model = MultimodalPhiModel(phi_model, tokenizer, device).to(device)
model.phi_model.load_adapter(peft_model_location)
model.image_projection.load_state_dict(torch.load(proj_model_location, weights_only=True))
model = model.to(device)

vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
vision_model = CLIPVisionModel.from_pretrained(vision_model_name)

def get_clip_embeddings(image_path):
    # Load and preprocess the image
    image = clipprocessor(images=Image.open(image_path), return_tensors="pt")

    # Generate the embedding
    image_features = clipmodel.get_image_features(**image)
    
    return image_features  # Move back to CPU for storage

import re

def getStringAfterAnswer(output):
    if "Answer:" in output:
        answer = output.split("Answer:")[1]
    elif "Answer" in output:
        answer = output.split("Answer")[1]
    else:
        answer = output

    answer = re.sub('\\s+', ' ', answer)
    return answer


def generateOutput(imageEmbeddings, questionText, max_length=30):
    answerPart = ""
    for i in range(max_length):
        start_text, end_text = getInputs(questionText, answer=answerPart)
        output = model(start_text=start_text, end_text=end_text, attention_mask=None, labels=None, image_features=torch.tensor(imageEmbeddings).to(device))
        tokens = output.logits.argmax(dim=-1)
        output = tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
            )
        answerPart = getStringAfterAnswer(output)
        print("Answerpart:", answerPart)

    return answerPart

image_path = './data/train2014/COCO_train2014_000000000034.jpg'
embeddings = get_clip_embeddings(image_path)

questionText = "Please describe the data given in Context in your own words?"
print(generateOutput(embeddings, questionText))

# input_ids, attention_mask = getInputs("Please summarize the information in the data?")
# # print("input_ids:", input_ids)
# # print("IMAGE encoding:", tokenizer.encode("[IMAGE]", add_special_tokens=False))
# output = model(input_ids.to(device), attention_mask.to(device), None, embeddings.to(device))

# tokens = output.logits.argmax(dim=-1)
# output = tokenizer.decode(
#     tokens[0],
#     skip_special_tokens=True
#     )
# print("Output:", output)
