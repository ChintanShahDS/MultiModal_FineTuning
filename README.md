# MultiModal FineTuning

## Overall approach
- Model based on the Llava model approach
- Dataset used is Instruct 150k dataset with the COCO original images

### Design
- Phi 3.5 mini instruct model used as an LLM
- QLora used for fine tuning the projection weights
- Created a projection layer with 2 linear layers connected by Gelu before the Phi 3.5
- Clip embeddings from Clipmodel are passed through the projection layer to create embeddings in dimension of Phi 3.5
- They are then passed through the Phi 3.5 along with the question and answer from 150k dataset
- The loss calculated is then backward propogated to finetune the Phi3.5 Lora weights
- Along with that the projection layer is fine tuned to change it to the embeddings of Phi 3.5

### USP
- Remove stopwords and punctuation from the answers to keep focus on specific words
- Keep the Start part of text and end part separate and insert the image tokens in between rather than using a placeholder
  - This was done so that we can use the outputs from different layers of ClipModel for trials and any number of channels can be accomodated
- Projection layer is 2 layered with Gelu in between
- The model is fine tuned multiple times on smaller set of inputs due to the GPU availability limitation
- On the output of this fine tuned model base phi model is run during inference to get the answer
  - This is done since the output of the fine tuned model was garbled with some words and repeated aspects and spelling mistakes
- Speech is converted to text and direct text inputs are passed directly to the base model and not to the fine tuned adaptor
- During inference the output is formatted to remove punctuations, extra spaces and repeated text from question
- Also during inference since image embeddings are passed the model inference is called multiple times to generate a longer response
  - This is due to unability to use generate with embeddings
- Used Huggingface trainer
- Enabled Gradient checkpointing
- Saved the projection layer and the Lora Phi 3.5 separately

### Trials done and learnings
- Trained only the projection layer by freezing the Phi 3.5 - Did not work
- Trained without removing stop words - The and of kind of words repeated rather than any other important word during inference
- Trained without removing punctuations - Too many punctuations in the output text
- Trained using the -1 layer of the clipvisionmodel with 50 channels and 768 embedding dimension - Results were ok
  - The normal clipmodel.get_image_features gave 1 x 512 output and better or similar results
  - The thought process to use the 50 x 768 was to get the patches information as well but seems issues maybe due to smaller images with big objects
- Defined self loss function and trainer module - Faced issues with the loss
- Data curation was not done and used as is - Probably data curation could have helped in this case
- Looked at Llava implementation, Phi3.5 vision and many other implementations
  - They were quite complex and needed larger infrastructure so settled for own version
- Tried training the projection layer directly by comparing the outputs with the text embedding to calcualate loss - Did not work
- Tried doing cosine similarity of the sentence for loss calculations - Did not work

