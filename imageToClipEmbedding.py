
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
from PIL import Image
import os
import json
import sys
import h5py

# vision_model_name = 'openai/clip-vit-large-patch14-336' 
vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
_ = vision_model.requires_grad_(False)

vision_model = vision_model.to("cuda")

# def feature_select(image_forward_outs):
#     image_features = image_forward_outs.hidden_states[-1] # last layer
#     # print(image_features.shape) # 1, 50, 768
#     image_features = image_features[:, 1:, :]
#     return image_features # 1, 49, 768


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

def process_images_in_folder(folder_path, output_location):
    embeddings_data = []
    
    count = 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(image_path, folder_path)
                embeddings = get_clip_embeddings(image_path)
                
                # embeddings_data.append({
                #     "image_path": relative_path,
                #     "image_name": filename,
                #     "embedding": embeddings.tolist()  # Convert to list for JSON serialization
                # })
                with h5py.File(f'{output_location}/{os.path.basename(image_path).split(".")[0]}.h5', 'w') as f:
                    f.create_dataset('embeddings', data=embeddings)

            if (count % 1000 == 0):
                print("Processed: ", count)

            # if (count > 2):
            #     print("Processed: ", count)
            #     break
    
# Example usage
image_folder = "./data/train2014"
# output_file = "./data/clip_embeddings.json"
output_location = "./data/train_embeddings"
process_images_in_folder(image_folder, output_location)

print(f"CLIP embeddings have been extracted and saved to {output_location}")
# print("The file contains a list of dictionaries with 'image_path', 'image_name', and 'embedding' for each image.")
