"""
compare_outputs.py: Compare base model vs LoRA fine-tuned model

This script generates text from both the base model and LoRA fine-tuned model
to visually compare the behavior change after fine-tuning.

Usage:
    python -m scripts.compare_outputs --lora lora_checkpoints/lora_final.pt
"""
import argparse

import torch
import tiktoken

from gpt.model import GPT
from gpt.lora import apply_lora_to_model, load_lora_weights


def generate_text(
    model,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cuda"
):
    """
    Generate text from a model given a prompt.

    Args:
        model: GPT model
        prompt: Input prompt string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (only sample from top k tokens)
        device: Device to run on

    Returns:
        Generated text string
    """
    # Tokenize prompt
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(input_ids)

            # Get logits for last token
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode
    output_tokens = input_ids[0].tolist()
    output_text = enc.decode(output_tokens)

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Compare base vs LoRA fine-tuned outputs")
    parser.add_argument("--model", type=str, default="gpt2", help="Model size")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt", help="Base weights")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Test prompts (you can customize these)
    prompts = [
        "Once upon a time in",
        "The quick brown fox",
        "In the beginning,",
        "A long time ago,",
        "It was a dark and stormy"
    ]

    prompts = prompts[:args.samples]

    # ========================================================================
    # 1. Load base model and generate
    # ========================================================================
    print("="*80)
    print("LOADING BASE MODEL (No fine-tuning)")
    print("="*80)

    base_model = GPT.from_pretrained(args.model, weights_path=args.weights)
    base_model.to(device)
    base_model.eval()

    print("\n" + "="*80)
    print("BASE MODEL OUTPUTS")
    print("="*80 + "\n")

    base_outputs = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[Sample {i}] Prompt: \"{prompt}\"")
        output = generate_text(
            base_model,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        base_outputs.append(output)
        print(f"Output: {output}\n")
        print("-"*80 + "\n")

    # Clean up base model
    del base_model
    torch.cuda.empty_cache()

    # ========================================================================
    # 2. Load LoRA fine-tuned model and generate
    # ========================================================================
    print("="*80)
    print("LOADING LORA FINE-TUNED MODEL")
    print("="*80)

    lora_model = GPT.from_pretrained(args.model, weights_path=args.weights)

    # Apply LoRA structure
    print(f"Applying LoRA structure (rank={args.rank}, alpha={args.alpha})...")
    apply_lora_to_model(lora_model, rank=args.rank, alpha=args.alpha)

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora}...")
    load_lora_weights(lora_model, args.lora)

    lora_model.to(device)
    lora_model.eval()

    print("\n" + "="*80)
    print("LORA FINE-TUNED MODEL OUTPUTS")
    print("="*80 + "\n")

    lora_outputs = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[Sample {i}] Prompt: \"{prompt}\"")
        output = generate_text(
            lora_model,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        lora_outputs.append(output)
        print(f"Output: {output}\n")
        print("-"*80 + "\n")

    # ========================================================================
    # 3. Side-by-side comparison
    # ========================================================================
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[Sample {i}] Prompt: \"{prompt}\"")
        print(f"\nBase Model:")
        print(f"  {base_outputs[i-1]}")
        print(f"\nLoRA Model:")
        print(f"  {lora_outputs[i-1]}")
        print("\n" + "="*80 + "\n")

    print("\nComparison complete!")
    print("Look for changes in:")
    print("  - Vocabulary (does it use words/phrases from your training data?)")
    print("  - Style (does it match the tone of your training data?)")
    print("  - Coherence (is it still generating valid English?)")


if __name__ == "__main__":
    main()
