import torch
import torch.nn as nn
import os
from peft import PeftModel
from PIL import Image
import gradio as gr
import librosa
import nltk

from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the model and processor
clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

nltk.download('punkt')
nltk.download('punkt_tab')

def remove_punctuation(text):
    newtext = ''.join([char for char in text if char.isalnum() or char.isspace()])
    newtext = ' '.join(newtext.split())
    return newtext

def preprocess_text(text):
    text_no_punct = remove_punctuation(text)
    return text_no_punct

# Load Whisper model and processor
whisper_model_name = "openai/whisper-small"
whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

def transcribe_speech(audiopath):
    # Load and preprocess the audio
    speech, rate = librosa.load(audiopath, sr=16000)
    audio_input = whisper_processor(speech, return_tensors="pt", sampling_rate=16000)
    # print("audio_input:", audio_input)
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = whisper_model.generate(audio_input["input_features"])
    
    # Decode the transcription
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

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
        self.base_phi_model = None

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
        # projector_path = os.path.join(pretrained_model_name_or_path, "projection_layer", "pytorch_model.bin")
        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi_model.device)

            projector = ProjectionBlock(input_dim, output_dim)

            # Try to load the state dict, ignoring mismatched keys
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            input_dim = 512  # Default CLIP embedding size
            output_dim = phi_model.config.hidden_size
            projector = ProjectionBlock(input_dim, output_dim)

        # Create and return the Phi3WithProjector instance
        model = self(phi_model, tokenizer, projector)
        model.base_phi_model = base_phi_model
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

def getImageArray(image_path):
    image = Image.open(image_path)
    return image

def getAudioArray(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    return speech

def getInputs(image_path, question, answer=""):

    image_features = None
    speech_text = ""
    num_image_tokens = 0

    if image_path is not None:
        # print("type of image:", type(image_path))
        # print("image path:", image_path)	
        image = clipprocessor(images=Image.open(image_path), return_tensors="pt")

        # Generate the embedding
        image_features = clipmodel.get_image_features(**image)

        # Generate the embedding
        # image_features = get_clip_embeddings(image)
        image_features = torch.stack([image_features])
        num_image_tokens = image_features.shape[1]

    # Start text before putting image embedding
    start_text = "<|system|> You are an assistant good at understanding the context. <|end|> \n <|user|>"

    # Prepare text input for causal language modeling
    end_text = f".\n  Describe the objects and their relationship in the given context. <|end|> \n <|assistant|>\n{answer}"

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
    if image_path is not None:
        attention_mask = torch.cat([start_attention_mask, torch.ones((1, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)
    else:
        attention_mask = torch.cat([start_attention_mask, end_attention_mask], dim=1)

    return start_input_ids, end_input_ids, image_features, attention_mask

model_location = "./MM_FT_C1_V2"
# print("Model location:", model_location)

model = MultimodalPhiModel.from_pretrained(model_location).to(device)

# Start text before putting image embedding
start_text = "<|system|> \n You are an assistant good at understanding the context.<|end|> \n <|user|> \n"
# Prepare text input for causal language modeling
end_text = "\n Describe the objects and their relationship in the given context.<|end|> \n <|assistant|> \n"

words = nltk.word_tokenize(start_text) + nltk.word_tokenize(end_text)
input_words = list(set(words))
print("Input words:",input_words)

import re

def getStringAfter(output, start_str):
    if start_str in output:
        answer = output.split(start_str)[1]
    else:
        answer = output

    answer = preprocess_text(answer)
    return answer

def getAnswerPart(output):
    output_words = nltk.word_tokenize(output)
    filtered_words = [word for word in output_words if word.lower() not in [w.lower() for w in input_words]]
    return ' '.join(filtered_words)

# def getStringAfterAnswer(output):
#     if "<|assistant|>" in output:
#         answer = output.split("<|assistant|>")[1]
#     else:
#         answer = output

#     answer = preprocess_text(answer)
#     return answer

def generateOutput(image_path, audio_path, context_text, question, max_length=3):
    answerPart = ""
    speech_text = ""
    if image_path is not None:
        for i in range(max_length):
            start_tokens, end_tokens, image_features, attention_mask = getInputs(image_path, question, answer=answerPart)
            # print("image_features dtype:", image_features.dtype)
            output = model(start_tokens, end_tokens, image_features, attention_mask, labels=None)
            tokens = output.logits.argmax(dim=-1)
            output = tokenizer.decode(
                tokens[0],
                skip_special_tokens=True
                )
            answerPart = getAnswerPart(output)
        print("Output:", output)
        print("Answerpart:", answerPart)

    if audio_path is not None:
        speech_text = transcribe_speech(audio_path)
        print("Speech Text:", speech_text)

    if (question is None) or (question == ""):
        question = "Provide only in 1 sentence to describe the objects and their relationships in it."

    input_text = (
        "<|system|>\nPlease understand the context "
        "and answer the question based on the context in 1 or 2 summarized sentences.\n"
        f"<|end|>\n<|user|>\n<|context|>{answerPart}\n{speech_text}\n{context_text}"
        f"\n<|question|>: {question}\n<|end|>\n<|assistant|>\n"
    )
    print("input_text:", input_text)
    tokens = tokenizer(input_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    start_tokens = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    # base_phi_model.generate(start_tokens, max_length=2, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    output_text = tokenizer.decode(
        model.base_phi_model.generate(start_tokens, attention_mask=attention_mask,max_length=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id)[0],
        skip_special_tokens=True
    )

    output_text = getStringAfter(output_text, question).strip()
    return output_text

demo = gr.Blocks()

title = "Created Fine Tuned MultiModal model"
description = "Test the fine tuned multimodal model created using clip, phi3.5 mini instruct, whisper models"

def process_inputs(image, audio_source, audio_file, audio_mic, context_text, question):
    if audio_source == "Microphone":
        speech = audio_mic
        # speech_text = transcribe_speech(audio_mic)
    elif audio_source == "Audio File":
        speech = audio_file
        # speech_text = transcribe_speech(audio_file)
    else:
        speech = None

    # image_features = get_clip_embeddings(image) if image else None
    answer = generateOutput(image, speech, context_text, question)
    
    # output = model(start_input_ids, end_input_ids, image_features, attention_mask, labels=None)
    # tokens = output.logits.argmax(dim=-1)
    # answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    return answer

with demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(f" {description}")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(type="filepath", label="Upload Image")
        with gr.Column(scale=2, min_width=300):
            question = gr.Textbox(label="Question")
            with gr.Row():
                audio_source = gr.Radio(choices=["Microphone", "Audio File"], label="Select Audio Source")
                audio_file = gr.Audio(sources="upload", type="filepath", visible=False)
                audio_mic = gr.Audio(sources="microphone", type="filepath", visible=False)
            context_text = gr.Textbox(label="Context Text")
            output_text = gr.Textbox(label="Output")
    # with gr.Row():

    def update_audio_input(source):
        if source == "Microphone":
            return gr.update(visible=True), gr.update(visible=False)
        elif source == "Audio File":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    audio_source.change(fn=update_audio_input, inputs=audio_source, outputs=[audio_mic, audio_file])
    submit_button = gr.Button("Submit")
    submit_button.click(fn=process_inputs, inputs=[image_input, audio_source, audio_file, audio_mic, context_text, question], outputs=output_text)

    # examples = gr.Examples(
    #     examples=[
    #         ["./data/images/COCO_train2014_000000581181.jpg", None, None, None, None, "Describe what is happening in this image."],
    #         [None, "Audio File", "./data/audio/03-01-01-01-01-01-01.wav", None, None, "Describe what is the person trying to tell in this audio."],
    #     ],
    #     inputs=[image_input, audio_source, audio_file, audio_mic, context_text, question],
    # ) 


demo.launch(debug=True)
