import json
import pandas as pd
import os

def read_conversations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    conversations = {}
    for item in data:
        id = item['id']
        image = item['image']
        conv = []
        for dialogue in item['conversations']:
            if dialogue['from'] == 'human':
                conv.append(('human', dialogue['value'].replace('<image>', '').replace('\n', ' ').strip()))
            elif dialogue['from'] == 'gpt':
                conv.append(('gpt', dialogue['value'].replace('\n', ' ').strip()))
        conversations[id] = {'image': image, 'conversation': conv}
    
    return conversations

def conversations_to_dataframe(conversations):
    rows = []
    for id, data in conversations.items():
        image = data['image']
        turn = 1
        for i in range(0, len(data['conversation']), 2):
            human_message = data['conversation'][i][1] if i < len(data['conversation']) else ""
            gpt_message = data['conversation'][i+1][1] if i+1 < len(data['conversation']) else ""
            rows.append({
                'id': id,
                'image': image,
                'turn': turn,
                'human': human_message,
                'gpt': gpt_message
            })
            turn += 1
    return pd.DataFrame(rows)

# Quick test

file_path = './data/llava_instruct_150k.json'
# Run the function with the test file
test_conversations = read_conversations(file_path)

# Convert conversations to DataFrame
df = conversations_to_dataframe(test_conversations)

# Write DataFrame to CSV
csv_file_path = 'conversations.csv'
df.to_csv(csv_file_path, index=False)

print(f"Conversations have been written to {csv_file_path}")

# Display the first few rows of the DataFrame
print(df.head())
