import torch
import torch.optim as optim
import tiktoken
from config import GPTConfig
from model import GPT 

# --- Setup ---
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use the default GPT-2 small config settings
config = GPTConfig() 
model = GPT(config)
model.to(device)

# Initialize the model with random weights (we need to train from scratch here)
print(f"Model initialized with random weights: {sum(p.numel() for p in model.parameters()):,} parameters.")

# --- 1. Define Dummy Training Data ---
tokenizer = tiktoken.get_encoding("gpt2")
# A simple, short sequence of text that the model should easily memorize
text = "I am training a small GPT model on a small dataset." * 10 
print(f"\nTraining on a repetitive sequence of {len(text)} characters.")

# Encode the text into token IDs
data = tokenizer.encode(text)
data = torch.tensor(data, dtype=torch.long, device=device)

# --- FIX: Correctly calculate the sequence length (T) ---
# We must ensure there is at least one token left over for the target (Y) sequence.
max_len_with_target = len(data) - 1

# T is the length of our input and target sequences, capped by the model's block size
T = min(config.block_size, max_len_with_target) 
B = 1 # Batch size is 1

# Create the single batch (B=1, T=max_len)
X = data[:T].unsqueeze(0) # Input tokens (1, T) -> indices [0, T-1]
Y = data[1:T+1].unsqueeze(0) # Target tokens (1, T) -> indices [1, T]

print(f"Batch X shape: {X.shape}, Batch Y shape: {Y.shape}. Starting loss calculation...")

# --- 2. Define Optimizer ---
# We use the standard AdamW optimizer used in most transformer projects
optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Fixed learning rate for simplicity

# --- 3. Training Loop (Overfitting) ---
num_iterations = 50 
model.train() # Set model to training mode

print(f"Starting overfit test for {num_iterations} iterations...")

initial_loss = 0.0

for step in range(num_iterations):
    
    # 1. Forward Pass
    # Since we pass targets (Y), the model returns logits and calculates the loss.
    logits, loss = model(X, targets=Y) 
    
    if step == 0:
        initial_loss = loss.item()
        
    # 2. Backward Pass
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients for all parameters
    
    # 3. Update Weights
    optimizer.step()      # Update model weights based on gradients and learning rate

    # 4. Logging
    if step % 5 == 0 or step == num_iterations - 1:
        # Convert loss to a standard Python number for printing
        current_loss = loss.item()
        print(f"Iteration {step:02d}/{num_iterations}: Loss = {current_loss:.6f}")

print("\n--- Overfit Test Complete ---")
print(f"Initial Random Loss: {initial_loss:.4f}")
print(f"Final Trained Loss: {loss.item():.4f}")
print("If the final loss is near 0.0, the training components are correct! Now run the corrected file.")
