import torch
from config import GPTConfig
from model import GPT # Assumes model.py contains the GPT class

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use default GPT-2 small config settings
config = GPTConfig() 
# You can override defaults like: config = GPTConfig(n_layer=6, n_head=6, n_embd=384)

# --- Model Initialization ---
model = GPT(config)
model.eval() # Set model to evaluation mode (disables dropout)
model.to(device)
print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters()):,} parameters.")

# --- Dummy Input Data ---
B = 4  # Batch size
T_in = 64 # Input sequence length (must be <= config.block_size)
assert T_in <= config.block_size

# Create random token IDs (long integers)
dummy_input_ids = torch.randint(0, config.vocab_size, (B, T_in), device=device)

# --- Perform Forward Pass ---
print(f"\nPerforming forward pass with input shape: {dummy_input_ids.shape}")
with torch.no_grad(): # Disable gradient calculation for inference/validation
    # Using torch.amp.autocast for potential mixed precision benefits even without training
    # On CPU, dtype defaults to float32. On CUDA, might use bfloat16 if supported.
    # Adjust dtype if needed (e.g., dtype=torch.float32 for consistency)
    with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32):
        logits, loss = model(dummy_input_ids, targets=None) # No targets needed for forward pass check

# --- Validate Output Shape ---
expected_shape = (B, T_in, config.vocab_size)
print(f"Output logits shape: {logits.shape}")
assert logits.shape == expected_shape, f"Expected shape {expected_shape}, but got {logits.shape}"
print("Forward pass successful and output shape is correct!")

# Optional: Print loss if you provided targets
# if loss is not None:
#    print(f"Calculated Loss (dummy targets): {loss.item():.4f}")
