from transformers import AutoTokenizer

# Load the Phi-2 tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Get the vocabulary
vocab = tokenizer.get_vocab()

# Print some information about the vocabulary
print(f"Vocabulary size: {len(vocab)}")
print("\nFirst 10 tokens:")
for token, index in list(vocab.items())[:10]:
    print(f"{token}: {index}")

print("\nLast 10 tokens:")
for token, index in list(vocab.items())[-10:]:
    print(f"{token}: {index}")

# Create a DataFrame from the vocabulary
import pandas as pd
df = pd.DataFrame(list(vocab.items()), columns=['Token', 'Index'])

# Save the DataFrame to a CSV file
df.to_csv('phi2_vocabulary.csv', index=False)

print("\nFull vocabulary has been saved to 'phi2_vocabulary.csv'")

# # Optionally, save the vocabulary to a file
# with open('phi2_vocabulary.txt', 'w', encoding='utf-8') as f:
#     for token, index in vocab.items():
#         f.write(f"{token}: {index}\n")

# print("\nFull vocabulary has been saved to 'phi2_vocabulary.txt'")
