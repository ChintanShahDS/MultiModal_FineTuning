import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
from transformers import PreTrainedModel
import gradio as gr

import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the model and processor
# clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clipvisionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clipimageprocessor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # Load Whisper model and processor
# whisper_model_name = "openai/whisper-small"
# whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
# whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# def transcribe_speech(audiopath):
#     # Load and preprocess the audio
#     speech, rate = librosa.load(audiopath, sr=16000)
#     audio_input = whisper_processor(speech, return_tensors="pt", sampling_rate=16000)
#     print("audio_input:", audio_input)
    
#     # Generate transcription
#     with torch.no_grad():
#         generated_ids = whisper_model.generate(audio_input["input_features"])
    
#     # Decode the transcription
#     transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     return transcription

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
        # phi_path = pretrained_model_name_or_path

        # # Save the base model
        # model = PeftModel.from_pretrained(base_phi_model, phi_path)
        # phi_model = model.merge_and_unload()

        # # Load the base Phi-3 model
        # phi_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        input_dim = 768
        output_dim = 3072

        # Load the projector weights
        # projector_path = os.path.join(pretrained_model_name_or_path, "projection_layer", "pytorch_model.bin")
        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=base_phi_model.device)

            projector = ProjectionBlock(input_dim, output_dim)

            # Try to load the state dict, ignoring mismatched keys
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            input_dim = 768  # Default CLIP embedding size
            output_dim = base_phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        # Create and return the Phi3WithProjector instance
        model = self(base_phi_model, tokenizer, projector)
        return model

    def save_pretrained(self, save_directory):
        # Load the Phi-3.5 model
        # self.phi_model.save_pretrained(save_directory)
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

    def encode(self, image_features):
        image_projections = self.image_projection(image_features)
        return image_projections

    def forward(self, start_input_ids, end_input_ids, image_features, attention_mask, labels):
        # print("tokenizer bos_token_id", self.tokenizer.bos_token_id, "tokenizer eos_token", self.tokenizer.eos_token,
        #       "tokenizer pad_token_id", self.tokenizer.pad_token_id, "tokenizer sep_token_id", self.tokenizer.sep_token_id,
        #       "tokenizer cls_token_id", self.tokenizer.cls_token_id, "tokenizer mask_token_id", self.tokenizer.mask_token_id,
        #       "tokenizer unk_token_id", self.tokenizer.unk_token_id)
        device = next(self.parameters()).device

        start_embeds = self.phi_model.get_input_embeddings()(start_input_ids.to(device))
        end_embeds = self.phi_model.get_input_embeddings()(end_input_ids.to(device))
        # print("start_embeds shape:", start_embeds.shape, "image_embeddings shape:", image_embeddings.shape, "end_embeds shape:", end_embeds.shape)
        # print("start_embeds dtype:", start_embeds.dtype, "image_embeddings dtype:", image_embeddings.dtype, "end_embeds dtype:", end_embeds.dtype)
        if image_features is not None:
            # Encode image features
            image_embeddings = self.encode(image_features.to(device)).bfloat16()
            input_embeds = torch.cat([start_embeds, image_embeddings, end_embeds], dim=1)
        else:
            input_embeds = torch.cat([start_embeds, end_embeds], dim=1)
        # print("Input Embeds shape:", input_embeds.shape, "attention_mask shape:", attention_mask.shape, "labels shape:", labels.shape)

        # print("input_embeds dtype:", input_embeds.dtype, "attention_mask dtype:", attention_mask.dtype)
        # Forward pass through the language model
        outputs = self.phi_model(inputs_embeds=input_embeds.to(device), 
                                 attention_mask=attention_mask.to(device), 
                                 labels=labels, 
                                 return_dict=True)
        
        return outputs


# def get_clip_embeddings(image):
#     # Load and preprocess the image
#     image = clipprocessor(images=image, return_tensors="pt")

#     # Generate the embedding
#     image_features = clipmodel.get_image_features(**image)
    
#     return image_features  # Move back to CPU for storage

def getImageArray(image_path):
    image = Image.open(image_path)
    return image

def getAudioArray(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    return speech

def getInputs(image_path, speech, textcontext, question, answer=""):

    image_features = None
    speech_text = ""
    if speech is not None:
        speech_text = transcribe_speech(speech)  

    if image_path is not None:
        # print("type of image:", type(image_path))
        # print("image path:", image_path)	
        image = Image.open(image_path).convert('RGB')
        image = clipimageprocessor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_forward_out = clipvisionmodel(image.unsqueeze(0), output_hidden_states=True)
        # print("clipvisionmodel:", clipvisionmodel)
        image_features = image_forward_out.hidden_states[-1].squeeze(0)

        # Generate the embedding
        # image_features = get_clip_embeddings(image)
        image_features = torch.stack([image_features])

    # Start text before putting image embedding
    # start_text = (
    #     f"<|system|>\nYou are a helpful assistant good at answering questions"
    #     f" based on the given context.<|end|>\n<|user|>\n{speech_text}\n{textcontext}\n"
    # )
    start_text = (
        "<|system|>\nPlease understand the context "
        "and provide a sentence to describe the objects and their relationships in it"
        f"<|end|>\n<|user|>\n<|context|>\n"
    )

    # Prepare text input for causal language modeling
    # end_text = f"\n{question}<|end|>\n<|assistant|>\n{answer}"
    end_text = f"<|end|>\n<|assistant|>\n"

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
    if image is not None:
        attention_mask = torch.cat([start_attention_mask, torch.ones((1, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)
    else:
        attention_mask = torch.cat([start_attention_mask, end_attention_mask], dim=1)

    return start_input_ids, end_input_ids, image_features, attention_mask

model_location = "./Projmodel"

model = MultimodalPhiModel.from_pretrained(model_location).to(device)

# vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
# image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
# vision_model = CLIPVisionModel.from_pretrained(vision_model_name)

model_name = "microsoft/Phi-3.5-mini-instruct"
base_phi_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)

import re

def getStringAfterAnswer(output):
    if "<|assistant|>" in output:
        answer = output.split("<|assistant|>")[1]
    else:
        answer = output

    answer = re.sub('\\s+', ' ', answer)
    return answer

def generateOutput(image_path, speech_text, context_text, question, max_length=1):
    answerPart = ""
    for i in range(max_length):
        start_tokens, end_tokens, image_features, attention_mask = getInputs(image_path, speech_text, context_text, question, answer=answerPart)
        # print("image_features dtype:", image_features.dtype)
        output = model(start_tokens, end_tokens, image_features, attention_mask, labels=None)
        tokens = output.logits.argmax(dim=-1)
        output = tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
            )
        answerPart = getStringAfterAnswer(output)
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

# def getResponse(image, audio, text, question):
#     image_path = f'./data/train2014/{image}'
#     audio_path = f'./data/train2014/{audio}'
#     questionText = f"{text} {question}"
#     print(f"Image: {image_path} is {generateOutput(image_path, questionText)}\n")
#     print(f"Audio: {audio_path} is {transcribe_audio(audio_path)}\n")

image_path = None
i = 0
folderpath = "./data/test/"
# folderpath = "./data/train2014/"
for image in os.listdir(folderpath):
    if i > 0:
        image_path = f'{folderpath}{image}'
        questionText = "Specify the objects that are present in the context."
        print(f"Image: {image_path} is {generateOutput(image_path, None, "", questionText)}\n")
    i += 1
    if i > 10:
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

# import gradio as gr

# demo = gr.Blocks()

# def process_inputs(image, audio_source, audio_file, audio_mic, context_text, question):
#     if audio_source == "Microphone":
#         speech = audio_mic
#         # speech_text = transcribe_speech(audio_mic)
#     elif audio_source == "Audio File":
#         speech = audio_file
#         # speech_text = transcribe_speech(audio_file)
#     else:
#         speech = None

#     # image_features = get_clip_embeddings(image) if image else None
#     answer = generateOutput(image, speech, context_text, question)
    
#     # output = model(start_input_ids, end_input_ids, image_features, attention_mask, labels=None)
#     # tokens = output.logits.argmax(dim=-1)
#     # answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
#     return answer

# with demo:
#     with gr.Row():
#         audio_source = gr.Radio(choices=["Microphone", "Audio File"], label="Select Audio Source")
#         audio_file = gr.Audio(sources="upload", type="filepath", visible=False)
#         audio_mic = gr.Audio(sources="microphone", type="filepath", visible=False)
#         image_input = gr.Image(type="filepath", label="Upload Image")
#         context_text = gr.Textbox(label="Context Text")
#         question = gr.Textbox(label="Question")
#         output_text = gr.Textbox(label="Output")

#     def update_audio_input(source):
#         if source == "Microphone":
#             return gr.update(visible=True), gr.update(visible=False)
#         elif source == "Audio File":
#             return gr.update(visible=False), gr.update(visible=True)
#         else:
#             return gr.update(visible=False), gr.update(visible=False)

#     audio_source.change(fn=update_audio_input, inputs=audio_source, outputs=[audio_mic, audio_file])
#     submit_button = gr.Button("Submit")
#     submit_button.click(fn=process_inputs, inputs=[image_input, audio_source, audio_file, audio_mic, context_text, question], outputs=output_text)

# examples = [
#     ["example_image.jpg", "Microphone", None, "example_audio_mic.wav", "This is a context text.", "What is the question?"],
#     ["example_image.jpg", "Audio File", "example_audio_file.wav", None, "This is another context text.", "What is the answer?"],
#     [None, "Microphone", None, "example_audio_mic.wav", "Context without image.", "What is the result?"],
#     ["example_image.jpg", "Audio File", "example_audio_file.wav", None, "Context with image.", "What is the output?"]
# ]

# demo.launch(debug=True)
