import torch
import pandas as pd
from transformers import CLIPModel, CLIPProcessor

import torch.nn as nn
import torch.optim as optim

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

def train_projection_layer(projection_layer, clip_embeddings, phi3_5_embeddings, epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(projection_layer.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        projection_layer.train()
        optimizer.zero_grad()

        outputs = projection_layer(clip_embeddings)
        loss = criterion(outputs, phi3_5_embeddings)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Load the CLIP model and processor from Hugging Face
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    # Load the phi3.5 model from Hugging Face
    phi3_5_model_id = "path/to/phi3.5-model"
    phi3_5_model = torch.hub.load('huggingface/pytorch-transformers', 'model', phi3_5_model_id)

    # Example dimensions, adjust as necessary
    clip_embedding_dim = 512
    phi3_5_embedding_dim = 768

    # Example data, replace with actual embeddings
    clip_embeddings = torch.randn((10, clip_embedding_dim))
    phi3_5_embeddings = torch.randn((10, phi3_5_embedding_dim))

    projection_layer = ProjectionLayer(clip_embedding_dim, phi3_5_embedding_dim)
    train_projection_layer(projection_layer, clip_embeddings, phi3_5_embeddings)
    def load_embeddings(file_path):
        return torch.tensor(torch.load(file_path))

    def load_qa_data(csv_path):
        df = pd.read_csv(csv_path)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        return questions, answers

    # Paths to the embeddings and QA data
    clip_embeddings_path = 'path/to/clip_embeddings.pt'
    phi3_5_embeddings_path = 'path/to/phi3_5_embeddings.pt'
    qa_data_path = 'path/to/qa_data.csv'

    # Load embeddings and QA data
    clip_embeddings = load_embeddings(clip_embeddings_path)
    phi3_5_embeddings = load_embeddings(phi3_5_embeddings_path)
    questions, answers = load_qa_data(qa_data_path)

    # Initialize and train the projection layer
    projection_layer = ProjectionLayer(clip_embeddings.size(1), phi3_5_embeddings.size(1))
    train_projection_layer(projection_layer, clip_embeddings, phi3_5_embeddings)


