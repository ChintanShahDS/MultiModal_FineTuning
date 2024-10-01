
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Phi model
model_name = "microsoft/Phi-3-mini-4k-instruct"  # or whichever Phi model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
phi_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print("phi_model details:")
print(phi_model)

# print("children of phi_model:")
# main_phi_children = []
# for i, child in enumerate(phi_model.children()):
#     print(f"Child {i}: {child}", "type: ", type(child))
#     main_phi_children.append(child)

# for i, child in enumerate(main_phi_children):
#     print(f"{i}: type: {type(child)} child: {child}")
#     if i == 0:
#         print("child without first layer: ", child[1:])

# class ModifiedPhi2Model(nn.Module):
#     def __init__(self, original_model):
#         super(ModifiedPhi2Model, self).__init__()
#         self.features = nn.Sequential(*main_phi_children)  # Skip the first layer

#     def forward(self, x):
#         x = self.features(x)
#         return x

# modified_phi2 = ModifiedPhi2Model(phi_model)

# # print("modified_phi2 details:")
# print(modified_phi2)
