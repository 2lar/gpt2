"""
Quick script to prepare tiny Shakespeare dataset for testing.
Downloads ~1MB of text and tokenizes it into train/val shards.
"""
import os
import numpy as np
import tiktoken
import requests

# Download Shakespeare
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data = requests.get(url).text

# Tokenize
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
tokens_np = np.array(tokens, dtype=np.int32)

# Split into train/val (90/10)
split_idx = int(len(tokens_np) * 0.9)
train_tokens = tokens_np[:split_idx]
val_tokens = tokens_np[split_idx:]

# Create directory
os.makedirs("shakespeare_data", exist_ok=True)

# Save shards
np.save("shakespeare_data/train_00000.npy", train_tokens)
np.save("shakespeare_data/val_00000.npy", val_tokens)

print(f"Saved {len(train_tokens):,} train tokens and {len(val_tokens):,} val tokens")
print("Update train.py: data_root='shakespeare_data'")
