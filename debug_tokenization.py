#!/usr/bin/env python3
"""Debug tokenization issue"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

# Load one sample
dataset = load_dataset('json', data_files='data/rl_training_data_full/train_data.jsonl', split='train')
print(f"Dataset size: {len(dataset)}")
print(f"Features: {dataset.features}")

# Get first sample
sample = dataset[0]
print(f"\nFirst sample keys: {sample.keys()}")
print(f"messages type: {type(sample['messages'])}")
print(f"messages (first 200 chars): {sample['messages'][:200]}")

# Try to parse messages
try:
    messages = json.loads(sample['messages'])
    print(f"\nParsed messages type: {type(messages)}")
    print(f"Parsed messages length: {len(messages)}")
    print(f"First message: {messages[0]}")

    # Try chat template
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"\nChat template applied successfully!")
        print(f"Result length: {len(text)}")
        print(f"Result (first 500 chars): {text[:500]}")

        # Try tokenization
        tokens = tokenizer(text, truncation=True, max_length=6144)
        print(f"\nTokenization successful!")
        print(f"Input IDs length: {len(tokens['input_ids'])}")

    except Exception as e:
        print(f"\nChat template ERROR: {e}")

except json.JSONDecodeError as e:
    print(f"\nJSON parsing ERROR: {e}")

# Now test batched processing
print("\n" + "="*80)
print("TESTING BATCHED PROCESSING")
print("="*80)

# Get first 5 samples as a batch
batch = dataset[:5]
print(f"Batch messages type: {type(batch['messages'])}")
print(f"Batch messages length: {len(batch['messages'])}")

text_list = []
for i, messages_str in enumerate(batch['messages']):
    print(f"\nSample {i}:")
    print(f"  Type: {type(messages_str)}")
    print(f"  Length: {len(messages_str)}")

    if isinstance(messages_str, str):
        try:
            messages_value = json.loads(messages_str)
            print(f"  Parsed OK: {len(messages_value)} messages")

            # Apply chat template
            text = tokenizer.apply_chat_template(messages_value, tokenize=False, add_generation_prompt=False)
            print(f"  Template OK: {len(text)} chars")
            text_list.append(text)

        except json.JSONDecodeError as e:
            print(f"  JSON ERROR: {e}")
            text_list.append("")
    else:
        print(f"  Not a string!")
        text_list.append("")

print(f"\nFinal text_list length: {len(text_list)}")
print(f"Non-empty texts: {sum(1 for t in text_list if t)}")

if text_list and any(text_list):
    encodings = tokenizer(text_list, truncation=True, max_length=6144, padding=False)
    print(f"\nBatch tokenization successful!")
    print(f"Encodings keys: {encodings.keys()}")
    print(f"input_ids type: {type(encodings['input_ids'])}")
    print(f"input_ids length: {len(encodings['input_ids'])}")
    for i, ids in enumerate(encodings['input_ids'][:3]):
        print(f"  Sample {i}: {len(ids)} tokens")
else:
    print("\nERROR: All texts are empty!")
