#!/usr/bin/env python3
"""Debug ROLL's get_chat_template function with Qwen3-8B"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer

# Import ROLL's chat template function
import sys
sys.path.insert(0, '/home/claude-user/ROLL')
from roll.datasets.chat_template import get_chat_template

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# Load one sample
dataset = load_dataset('json', data_files='data/rl_training_data_full/train_data.jsonl', split='train')
print(f"\nDataset size: {len(dataset)}")

# Get first sample
sample = dataset[0]
messages_str = sample['messages']
messages = json.loads(messages_str)

print(f"\nMessages: {len(messages)} messages")
print(f"First message role: {messages[0]['role']}")

# Test with qwen2_5 template
print("\n" + "="*80)
print("TESTING: qwen2_5 template")
print("="*80)
try:
    chat_template_qwen2_5 = get_chat_template("qwen2_5", tokenizer)
    text = chat_template_qwen2_5(messages)
    print(f"✓ SUCCESS: Generated {len(text)} characters")
    print(f"First 300 chars: {text[:300]}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with qwen3 template
print("\n" + "="*80)
print("TESTING: qwen3 template")
print("="*80)
try:
    chat_template_qwen3 = get_chat_template("qwen3", tokenizer)
    text = chat_template_qwen3(messages)
    print(f"✓ SUCCESS: Generated {len(text)} characters")
    print(f"First 300 chars: {text[:300]}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test batched processing (like ROLL does)
print("\n" + "="*80)
print("TESTING: Batched processing with qwen3")
print("="*80)
batch = dataset[:5]
print(f"Batch size: {len(batch['messages'])}")

text_list = []
for i, messages_str in enumerate(batch['messages']):
    try:
        messages_value = json.loads(messages_str)
        text = chat_template_qwen3(messages_value)
        text_list.append(text)
        print(f"  Sample {i}: {len(text)} chars")
    except Exception as e:
        print(f"  Sample {i}: FAILED - {e}")
        text_list.append("")

print(f"\nTotal successful: {sum(1 for t in text_list if t)}/{len(text_list)}")
