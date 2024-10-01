import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
from transformers import PreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the model and processor
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

    # @classmethod
    # def from_pretrained(self, pretrained_model_name_or_path, *model_args, debug=False, **kwargs):
    #     # Load the base Phi-3 model
    #     # Load the adaptor model for Phi3.5 from directory
    #     # Load the Phi-3.5 model
    #     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    #     model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_4bit=True, device_map="auto")
    #     print("type model:", type(model))
    #     # model_name = "microsoft/Phi-3.5-mini-instruct"
    #     # bnb_config = BitsAndBytesConfig(
    #     #     load_in_4bit=True,
    #     #     bnb_4bit_quant_type="nf4",
    #     #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     # )

    #     # phi_model = AutoModelForCausalLM.from_pretrained(
    #     #     model_name,
    #     #     torch_dtype=torch.bfloat16,
    #     #     quantization_config=bnb_config,
    #     #     trust_remote_code=True,
    #     #     # _attn_implementation='eager'
    #     # )
    #     # phi_model.config.use_cache = False

    #     # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #     # tokenizer.pad_token = tokenizer.eos_token

    #     # adaptor_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
    #     # adaptor_model = torch.load(pretrained_model_name_or_path)

    #     # # Apply the adaptor model to the phi_model
    #     # phi_model.load_state_dict(adaptor_model, strict=False)

    #     # phi_model = phi_model.from_pretrained(pretrained_model_name_or_path)

    #     input_dim = 512
    #     output_dim = 3072

    #     # Load the projector weights
    #     projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
    #     if os.path.exists(projector_path):
    #         projector_state_dict = torch.load(projector_path, map_location=model.device)
    #         projector = ProjectionBlock(input_dim, output_dim)
    #         # Try to load the state dict, ignoring mismatched keys
    #         projector.load_state_dict(projector_state_dict, strict=False)
    #         print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
    #     else:
    #         print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
    #         projector = ProjectionBlock(input_dim, output_dim)

    #     # Create and return the Phi3WithProjector instance
    #     model = self(model, tokenizer, projector, debug=debug)
    #     return model

    # def save_pretrained(self, save_directory):
    #     # Save the base model
    #     self.phi_model.save_pretrained(save_directory)

    #     # Save the projector weights
    #     projector_path = os.path.join(save_directory, "image_projector.pth")
    #     torch.save(self.image_projection.state_dict(), projector_path)

    #     # Save the config
    #     self.config.save_pretrained(save_directory)

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_input_ids, end_input_ids, image_features, attention_mask, labels):
        # print("tokenizer bos_token_id", self.tokenizer.bos_token_id, "tokenizer eos_token", self.tokenizer.eos_token,
        #       "tokenizer pad_token_id", self.tokenizer.pad_token_id, "tokenizer sep_token_id", self.tokenizer.sep_token_id,
        #       "tokenizer cls_token_id", self.tokenizer.cls_token_id, "tokenizer mask_token_id", self.tokenizer.mask_token_id,
        #       "tokenizer unk_token_id", self.tokenizer.unk_token_id)
        device = next(self.parameters()).device

        # Encode image features
        image_embeddings = self.encode(image_features.to(device)).bfloat16()

        start_embeds = self.phi_model.get_input_embeddings()(start_input_ids.to(device))
        end_embeds = self.phi_model.get_input_embeddings()(end_input_ids.to(device))
        # print("start_embeds shape:", start_embeds.shape, "image_embeddings shape:", image_embeddings.shape, "end_embeds shape:", end_embeds.shape)
        # print("start_embeds dtype:", start_embeds.dtype, "image_embeddings dtype:", image_embeddings.dtype, "end_embeds dtype:", end_embeds.dtype)
        input_embeds = torch.cat([start_embeds, image_embeddings, end_embeds], dim=1)
        # print("Input Embeds shape:", input_embeds.shape, "attention_mask shape:", attention_mask.shape, "labels shape:", labels.shape)

        # print("input_embeds dtype:", input_embeds.dtype, "attention_mask dtype:", attention_mask.dtype)
        # Forward pass through the language model
        outputs = self.phi_model(inputs_embeds=input_embeds.to(device), 
                                 attention_mask=attention_mask.to(device), 
                                 labels=labels, 
                                 return_dict=True)
        
        return outputs


def get_clip_embeddings(image_path):
    # Load and preprocess the image
    image = clipprocessor(images=Image.open(image_path), return_tensors="pt")

    # Generate the embedding
    image_features = clipmodel.get_image_features(**image)
    
    return image_features  # Move back to CPU for storage

def getInputs(image_path, question, answer=""):

    # Generate the embedding
    image_features = get_clip_embeddings(image_path)

    # Start text before putting image embedding
    start_text = f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n"

    # Prepare text input for causal language modeling
    end_text = f"\n{question}<|end|>\n<|assistant|>\n{answer}"

    image_features = torch.stack([image_features])

    num_image_tokens = image_features.shape[1]
    # print("batch_size:", batch_size)
    # print("num_image_tokens:", num_image_tokens)
    
    # print("image features shape:", image_features.shape)
    # Encode image features
    # image_tokens = torch.full((1, num_image_tokens), -100, dtype=torch.long)

    # Tokenize the full texts
    start_tokens = tokenizer(start_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    end_tokens = tokenizer(end_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # print(f"start_encodings shape: {start_encodings['input_ids'].shape}, end_encodings shape: {end_encodings['input_ids'].shape}")
    
    start_input_ids = start_tokens['input_ids']
    start_attention_mask = start_tokens['attention_mask']
    end_input_ids = end_tokens['input_ids']
    end_attention_mask = end_tokens['attention_mask']

    # print("start_input_ids type:", type(start_input_ids), "image_tokens type:", type(image_tokens))
    # print(f"start_input_ids shape: {start_input_ids.shape}, image_tokens shape: {image_tokens.shape}, end_input_ids shape: {end_input_ids.shape}")
    # input_ids = torch.cat([start_input_ids,image_tokens,end_input_ids], dim=1)
    attention_mask = torch.cat([start_attention_mask, torch.ones((1, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)

    return start_input_ids, end_input_ids, image_features, attention_mask

model_location = "./Phi3_CV_results/models"

# model = load_checkpoint_and_dispatch(
#     model, checkpoint=model_location, device_map="auto"
# )
# proj_model = ProjectionBlock(input_dim_CLIP=512, input_dim_phi2=3072).to(device)

# Usage example:
# projector = ProjectionBlock(512, 3072)
# Usage example:
model = MultimodalPhiModel.from_pretrained(model_location).to(device)
print(model)
# # Configure LoRA
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "o_proj",
#         "qkv_proj",
#         "gate_up_proj",
#         "down_proj",
#         # "fc1",
#         # "fc2",
#     ]
# )

# # # Apply LoRA to the Phi model part of the multimodal model
# model.phi_model = get_peft_model(model.phi_model, peft_config)
# # Enable gradient checkpointing for the model
# model.image_projection.load_state_dict(torch.load(proj_model_location, weights_only=True))
# model = model.to(device)

# Load the Phi-3.5 adaptor model from the directory
# model_path = "./models"

# # Apply the adaptor model to the phi_model
# model.phi_model.load_state_dict(adaptor_model, strict=False)

model_name = "microsoft/Phi-3.5-mini-instruct"
base_phi_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)


# vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
# image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
# vision_model = CLIPVisionModel.from_pretrained(vision_model_name)

import re

def getStringAfterAnswer(output):
    if "<|assistant|>" in output:
        answer = output.split("<|assistant|>")[1]
    else:
        answer = output

    answer = re.sub('\\s+', ' ', answer)
    return answer


def generateOutput(image_path, questionText, max_length=30):
    answerPart = ""
    for i in range(max_length):
        start_tokens, end_tokens, image_features, attention_mask = getInputs(image_path, questionText, answer=answerPart)
        # print("image_features dtype:", image_features.dtype)
        output = model(start_tokens, end_tokens, image_features, attention_mask, labels=None)
        tokens = output.logits.argmax(dim=-1)
        output = tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
            )
        answerPart = getStringAfterAnswer(output)
        # print("Answerpart:", answerPart)

    # print("Answerpart:", answerPart)
    input_text = (
        "<|system|>\nPlease understand the context "
        "and provide a sentence to describe the objects and their relationships in it"
        f"<|end|>\n<|user|>\n<|context|>{answerPart}"
        "<|end|>\n<|assistant|>\n"
    )
    # print("input_text:", input_text)
    start_tokens = tokenizer(input_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")['input_ids'].to(device)
    # base_phi_model.generate(start_tokens, max_length=2, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    output_text = tokenizer.decode(
        base_phi_model.generate(start_tokens, max_length=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)[0],
        skip_special_tokens=True
    )

    return output_text

image_path = None
i = 0
for image in os.listdir('./data/train2014/'):
    image_path = f'./data/train2014/{image}'
    questionText = "Summarize in simple english the important words?"
    print(f"Image: {image_path} is {generateOutput(image_path, questionText)}\n")
    i += 1
    if i > 20:
        break

# image_path = './data/train2014/COCO_train2014_000000000030.jpg'
# embeddings = get_clip_embeddings(image_path)


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
