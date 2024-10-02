# MultiModal FineTuning

## High level details
- Model based on the Llava model approach
- Dataset used is Instruct 150k dataset with the COCO original images

### Huggingface app for trial (Running on CPU to avoid paying money so very slow)
https://huggingface.co/spaces/Chintan-Shah/MyMultiModalExperiment

### File details
- imageToClipEmbedding.py: Initial try when I dumped clip embeddings for each image into a file. Dumped the hidden layer outputs in this case. But used direct Clip embeddings in the later models.
- MultiModalFull_Inference_C1.py - Main Inference file where Projection layer and Phi 3.5 QLora trained with get_image_features (1 x 512). Also used the output of the fine tuned model here and passed it to the base Phi3.5 model to get the language correct.
- MultiModalFull_Inference.py - Inference file where Projection layer and Phi 3.5 trained with last layer embeddings from CLIP model (50 x 768).
- QLora_FineTuning_Phi3.py - Initial fine tuning file where training was done using the last layer (50 x 768). These has patches and might be good for images with many things. Though this file was used for multiple trainings and trials.
- QLora_FineTuning_Phi3_C1.py - Main fine tuning file where Projection layer and Phi 3.5 QLora trained with get_image_features (1 x 512). This is what is deployed on huggingface
- QLora_FineTuning_Phi3_C1_FT2.py - Secondary fine tuning on top of tuned model. This is for continuous fine tuning of the models that I keep doing to improve the accuracy based on time and GPU availability. Will deploy the new models on hugging face
- TrainProjectionModel.py - This was the trial done where I trained only the ProjectionModel first and then used this to train the full Projection layer and Phi3.5 QLora again. But did not get good results. Will have to try more on this later.
- GetPhi2Vocab.py - Tried to get the Vocabulary of the Phi model to help choose the right tokens as well as an overall understanding
- Other files: The other files are the multiple trials done during this capstone that might help understand the different aspects

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

### Things that could be done further
- Train with a more deep projections layer
- Add image patches along with the full image by resizing and getting embeddings to get better results
- Check the text generation aspect of Phi 3.5 and the words not forming fully in the outputs
- Based on outputs do select the right images and the text to fine tune and fix those
- Curate the dataset to describe the scene in brief for the model to learn better

### Logs
#### Training round 1
```
D:\Chintan\OneDrive\Workspace\ERAv2\MultiModal_FineTuning>python QLora_FineTuning_Phi3_C1.py
C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
`flash-attention` package not found, consider installing for better performance: No module named 'flash_attn'.
Current `flash-attention` does not support `window_size`. Either upgrade or use `attn_implementation='eager'`.
`low_cpu_mem_usage` was None, now set to True since model is quantized.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:18<00:00,  9.13s/it]
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\Trial\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\Trial\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package punkt_tab to
[nltk_data]     C:\Users\Trial\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
Trainable parameters: 111,680,512
All parameters: 2,120,820,736
Percentage of trainable parameters: 5.27%
max_steps is given, it will override any value given in num_train_epochs
  0%|                                                                                                                             | 0/6000 [00:00<?, ?it/s]You are not running the flash-attention implementation, expect numerical differences.
C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
Could not estimate the number of tokens of the input, floating-point operations will not be computed
{'loss': 4.2947, 'grad_norm': 1.6839556694030762, 'learning_rate': 4.5e-05, 'epoch': 0.01}
{'loss': 3.2241, 'grad_norm': 1.8457916975021362, 'learning_rate': 4e-05, 'epoch': 0.03}
{'loss': 3.0872, 'grad_norm': 1.8399261236190796, 'learning_rate': 3.5e-05, 'epoch': 0.04}
{'loss': 2.9852, 'grad_norm': 1.998960256576538, 'learning_rate': 3e-05, 'epoch': 0.06}
 40%|█████████████████████████████████████████████▌                                                                    | 2400/6000 [41:55<55:32,  1.08it/s]C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 2.9303, 'grad_norm': 1.5595556497573853, 'learning_rate': 2.5e-05, 'epoch': 0.07}
{'loss': 2.8839, 'grad_norm': 2.708310604095459, 'learning_rate': 2e-05, 'epoch': 0.09}
{'loss': 2.8794, 'grad_norm': 2.0087153911590576, 'learning_rate': 1.5e-05, 'epoch': 0.1}
{'loss': 2.8551, 'grad_norm': 1.93650221824646, 'learning_rate': 1e-05, 'epoch': 0.12}
 80%|█████████████████████████████████████████████████████████████████████████████████████████▌                      | 4800/6000 [1:18:20<17:53,  1.12it/s]C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 2.8151, 'grad_norm': 2.461733818054199, 'learning_rate': 5e-06, 'epoch': 0.13}
{'loss': 2.8189, 'grad_norm': 2.3669679164886475, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': 5801.282, 'train_samples_per_second': 2.069, 'train_steps_per_second': 1.034, 'train_loss': 3.0773943888346356, 'epoch': 0.15}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6000/6000 [1:36:41<00:00,  1.03it/s]
```

#### Training Round 2

```
  projector_state_dict = torch.load(projector_path, map_location=phi_model.device)
Loaded projector with input_dim=512, output_dim=3072
Trainable parameters: 111,680,512
All parameters: 3,932,760,064
Percentage of trainable parameters: 2.84%
max_steps is given, it will override any value given in num_train_epochs
  0%|                                                                                                                             | 0/6000 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
You are not running the flash-attention implementation, expect numerical differences.
C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
Could not estimate the number of tokens of the input, floating-point operations will not be computed
{'loss': 2.8158, 'grad_norm': 0.5202670693397522, 'learning_rate': 4.5e-05, 'epoch': 0.01}
{'loss': 2.7849, 'grad_norm': 0.9820236563682556, 'learning_rate': 4e-05, 'epoch': 0.03}
{'loss': 2.7731, 'grad_norm': 0.4473443627357483, 'learning_rate': 3.5e-05, 'epoch': 0.04}
{'loss': 2.7528, 'grad_norm': 0.39497828483581543, 'learning_rate': 3e-05, 'epoch': 0.06}
{'loss': 2.7371, 'grad_norm': 0.5294825434684753, 'learning_rate': 2.5e-05, 'epoch': 0.07}
 50%|████████████████████████████████████████████████████████                                                        | 3000/6000 [2:28:35<47:34,  1.05it/s]C:\Users\Trial\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 2.7332, 'grad_norm': 0.7306874990463257, 'learning_rate': 2e-05, 'epoch': 0.09}
{'loss': 2.7198, 'grad_norm': 0.5060272812843323, 'learning_rate': 1.5e-05, 'epoch': 0.1}
{'loss': 2.7074, 'grad_norm': 0.6618576049804688, 'learning_rate': 1e-05, 'epoch': 0.12}
{'loss': 2.7094, 'grad_norm': 0.6062239408493042, 'learning_rate': 5e-06, 'epoch': 0.13}
{'loss': 2.7186, 'grad_norm': 1.026847243309021, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': 11545.8529, 'train_samples_per_second': 1.039, 'train_steps_per_second': 0.52, 'train_loss': 2.745202392578125, 'epoch': 0.15}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6000/6000 [3:12:25<00:00,  1.92s/it]
```

### Outputs (Initial - Updated model later for further improvement)

![image](https://github.com/user-attachments/assets/10eb42a3-f363-4fdc-9276-9c58dcff119a)

![image](https://github.com/user-attachments/assets/0e8ed752-eaa9-483d-ad89-e9b7d4e002f7)

![image](https://github.com/user-attachments/assets/fbbc8077-12fc-4696-a633-ec195c5f1316)
