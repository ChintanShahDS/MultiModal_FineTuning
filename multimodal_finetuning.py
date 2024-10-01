
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from PIL import Image
import torchaudio
from transformers import CLIPVisionModel, CLIPImageProcessor, WhisperProcessor, WhisperForConditionalGeneration

# Load and configure the Phi model
model_name = "microsoft/phi-2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load CLIP vision model and processor
clip_model_name = 'openai/clip-vit-base-patch32'
clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
clip_model = CLIPVisionModel.from_pretrained(clip_model_name).to("cuda")

# Load Whisper model and processor
whisper_model_name = "openai/whisper-base"
whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")

# Function to get CLIP embeddings
def get_clip_embeddings(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = clip_model(**inputs)
    return outputs.last_hidden_state.squeeze(0)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    audio, _ = torchaudio.load(audio_path)
    input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Prepare dataset (you'll need to implement this based on your data)
def prepare_dataset():
    # This is a placeholder. You should implement this function to load and preprocess your dataset.
    # The dataset should include image paths, audio paths (if applicable), and corresponding queries/text.
    return load_dataset("your_dataset_name_or_path")

# Configure LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["query_key_value"]  # Adjust based on your model architecture
)

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=3,
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
)

# Prepare the model for training
model = get_peft_model(model, peft_config)

# Custom data collator
def collate_fn(batch):
    input_ids = []
    attention_mask = []
    for item in batch:
        # Process image if present
        if 'image_path' in item:
            image_embeds = get_clip_embeddings(item['image_path'])
            image_text = f"[IMAGE] {' '.join(map(str, image_embeds.flatten().tolist()))}"
        else:
            image_text = ""
        
        # Process audio if present
        if 'audio_path' in item:
            audio_text = f"[AUDIO] {transcribe_audio(item['audio_path'])}"
        else:
            audio_text = ""
        
        # Combine all inputs
        full_text = f"{image_text} {audio_text} [QUERY] {item['query']} [RESPONSE] {item['response']}"
        
        encoded = tokenizer(full_text, truncation=True, padding='max_length', max_length=512)
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
    }

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=prepare_dataset(),
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_phi_multimodal")

print("Fine-tuning completed and model saved.")
