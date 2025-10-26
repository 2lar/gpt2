import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Setup: Load Official Hugging Face Components ---
MODEL_NAME = 'gpt2'
# Load the official model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
hf_model.eval()

# --- Custom Model Placeholder ---
# In a real scenario, this would be your manually loaded PyTorch implementation.
# We load a second official model instance here to ensure a true apples-to-apples comparison.
# If your weights were loaded correctly, the two models (custom_model and hf_model)
# should produce identical outputs.
custom_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
custom_model.eval()


def compare_full_text_generation(model_custom, model_hf, tokenizer, input_text, max_length=50):
    """
    Generates a full text sequence using both models and prints the results side-by-side.
    This is the ultimate functional test.
    """
    print(f"\n{'='*70}")
    print(f"## FUNCTIONAL TEXT GENERATION COMPARISON (Max {max_length} tokens)")
    print(f"Prompt: '{input_text}'")
    print(f"{'='*70}\n")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # --- Generate Text using Custom Model ---
    with torch.no_grad():
        # Using a fixed seed for generation ensures reproducibility, so the two models
        # MUST produce the same output if their logits are identical.
        torch.manual_seed(42)
        generated_ids_custom = model_custom.generate(
            input_ids, 
            max_length=input_ids.shape[1] + max_length,
            do_sample=True,             # <-- Turn on sampling!
            top_k=50,                   # <-- Only consider the top 50 tokens
            top_p=0.95,                 # <-- And sample from the top 95% of the probability mass
            pad_token_id=tokenizer.eos_token_id 
        )
        ## THE GEREATE IDS CUSTOM BELOW will ensure that the sampling is like from the huggingface model
        ## THis is for greedy decoding (disable sampling) = old
        # generated_ids_custom = model_custom.generate(
        #     input_ids, 
        #     max_length=input_ids.shape[1] + max_length, # max_length includes prompt tokens
        #     do_sample=False, # Use deterministic greedy decoding
        #     pad_token_id=tokenizer.eos_token_id 
        # )
        output_text_custom = tokenizer.decode(generated_ids_custom.squeeze(), skip_special_tokens=True)

    # --- Generate Text using Hugging Face Model ---
    with torch.no_grad():
        torch.manual_seed(42) # Reset seed to ensure the exact same generation conditions
        generated_ids_hf = model_hf.generate(
            input_ids, 
            max_length=input_ids.shape[1] + max_length, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        output_text_hf = tokenizer.decode(generated_ids_hf.squeeze(), skip_special_tokens=True)

    # --- Output Results ---
    print("Custom Model Output:\n" + "="*25)
    print(output_text_custom)
    
    print("\nHF Model Output:\n" + "="*25)
    print(output_text_hf)
    
    print(f"\n{'*'*30}")
    if output_text_custom == output_text_hf:
        print("RESULT: FULL GENERATED TEXT IS IDENTICAL! (Perfect Port)")
    else:
        print("RESULT: FULL GENERATED TEXT DIFFERENCE FOUND. (Check weights)")
    print(f"{'*'*30}\n")


def compare_runs_with_output(model_custom, model_hf, tokenizer, input_text):
    """
    (Original Check) Compares the logit outputs of two models and prints the max difference
    along with the top predicted next token from each model.
    """
    print(f"\n--- Technical Logit Comparison for: '{input_text}' ---\n")

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        output_custom = model_custom(input_ids=input_ids)
        logits_custom = output_custom.logits

        output_hf = model_hf(input_ids=input_ids)
        logits_hf = output_hf.logits

    logits_diff = torch.abs(logits_custom - logits_hf)
    max_diff = torch.max(logits_diff).item()
    
    predicted_token_id_custom = logits_custom[0, -1, :].argmax().item()
    predicted_token_custom = tokenizer.decode(predicted_token_id_custom)
    
    predicted_token_id_hf = logits_hf[0, -1, :].argmax().item()
    predicted_token_hf = tokenizer.decode(predicted_token_id_hf)

    print(f"Max Logit Difference: {max_diff:.8f}")
    print("-" * 40)
    print(f"Custom Model Prediction: '{predicted_token_custom}' (ID: {predicted_token_id_custom})")
    print(f"HF Model Prediction:     '{predicted_token_hf}' (ID: {predicted_token_id_hf})")
    print("-" * 40)
    
    if predicted_token_custom == predicted_token_hf:
        print("RESULT: Next Token Predictions MATCH!")
    else:
        print("RESULT: Next Token Predictions DO NOT MATCH!")


# ----------------------------------------------------------------------
# --- RUN COMPARISONS ---
# ----------------------------------------------------------------------

# Test String 1 (The initial one you used)
test_string_1 = "Hello, my name is Larry and I'm currently building a GPT model from scratch - but loading"
# compare_runs_with_output(custom_model, hf_model, tokenizer, test_string_1)
compare_full_text_generation(custom_model, hf_model, tokenizer, test_string_1, max_length=70)

# Test String 2 (Focus on full generation)
test_string_2 = "Artificial intelligence will fundamentally change how we"
compare_full_text_generation(custom_model, hf_model, tokenizer, test_string_2, max_length=70)
