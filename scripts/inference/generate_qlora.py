"""
generate_qlora.py: Generate text using QLoRA fine-tuned GPT-2

This script loads a base GPT-2 model, quantizes it to 4-bit, applies LoRA adapters,
and generates text.

QLoRA inference is memory-efficient but slightly slower than regular LoRA due to
on-the-fly dequantization.

Usage:
    # Basic usage
    python -m scripts.inference.generate_qlora \
      --qlora qlora_checkpoints/qlora_final.pt \
      --prompt "Once upon a time"

    # With custom weights path
    python -m scripts.inference.generate_qlora \
      --weights data/weights/gpt2_weights.pt \
      --qlora qlora_checkpoints/qlora_final.pt \
      --prompt "Your prompt here" \
      --max_new_tokens 200

    # Greedy decoding
    python -m scripts.inference.generate_qlora \
      --qlora qlora_checkpoints/qlora_final.pt \
      --prompt "To be or not to be" \
      --greedy

    # Custom sampling parameters
    python -m scripts.inference.generate_qlora \
      --qlora qlora_checkpoints/qlora_final.pt \
      --prompt "In the future" \
      --max_new_tokens 150 \
      --temperature 0.8 \
      --top_k 40 \
      --top_p 0.9
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from gpt.model import GPT
from gpt.qlora import apply_qlora_to_model, load_qlora_weights, print_qlora_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with QLoRA fine-tuned GPT-2")

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Base model size")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt",
                        help="Path to base model weights")

    # QLoRA configuration
    parser.add_argument("--qlora", type=str, required=True,
                        help="Path to QLoRA weights checkpoint (REQUIRED)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (must match training, default: 8)")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA alpha (must match training, default: 16.0)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Quantization block size (must match training, default: 64)")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (e.g., 50)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold (0.0-1.0)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (deterministic)")

    return parser.parse_args()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    greedy: bool = False,
    device: str = "cuda",
):
    """
    Generate text from a QLoRA-adapted model.

    Args:
        model: GPT model with QLoRA adapters applied
        tokenizer: GPT2 tokenizer
        prompt_text: Input prompt string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        greedy: Use greedy decoding if True
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()

    # Encode prompt
    x = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    eos_id = tokenizer.eos_token_id

    # Print generation info
    print(f"\n[PROMPT] {prompt_text}")
    print("=" * (len(prompt_text) + 12))

    if greedy:
        print("MODE: Greedy Decoding (Deterministic)")
    elif top_p is not None:
        print(f"MODE: Nucleus (Top-p={top_p}) Sampling")
    elif top_k is not None:
        print(f"MODE: Top-k ({top_k}) Sampling")
    else:
        print(f"MODE: Temperature ({temperature}) Sampling")

    print()

    # Generate tokens
    for _ in range(max_new_tokens):
        # Handle context window
        block_size = model.config.block_size
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        # Forward pass (includes dequantization overhead)
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]  # Get last token logits

        # Greedy decoding
        if greedy:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Temperature scaling
            logits = logits / max(1e-6, temperature)

            # Top-k filtering
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits[logits < thresh] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                remove_indices = sorted_indices[sorted_indices_to_remove]
                logits[0, remove_indices] = -float('inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        x = torch.cat((x, idx_next), dim=1)

        # Stop at EOS token
        if idx_next.item() == eos_id:
            break

    # Decode and return
    decoded = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
    print(decoded)
    print()

    return decoded


def main():
    args = parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("\n⚠️  WARNING: QLoRA inference on CPU will be VERY slow.")
        print("   Each forward pass requires dequantizing weights from 4-bit.")
        print("   Consider using regular LoRA for CPU inference.\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # ========================================================================
    # Load base model
    # ========================================================================
    print(f"\nLoading base {args.model} model from {args.weights}...")
    model = GPT.from_pretrained(args.model, weights_path=args.weights)

    base_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {base_params/1e6:.2f}M")

    # ========================================================================
    # Apply QLoRA structure (quantization + LoRA)
    # ========================================================================
    print(f"\nApplying QLoRA structure...")
    print(f"  - Quantization: NF4 4-bit (block_size={args.block_size})")
    print(f"  - LoRA: rank={args.rank}, alpha={args.alpha}")

    lora_params_count = apply_qlora_to_model(
        model,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=['c_attn'],
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )

    # ========================================================================
    # Load QLoRA weights
    # ========================================================================
    print(f"\nLoading QLoRA weights from {args.qlora}...")
    load_qlora_weights(model, args.qlora)

    model.to(device)
    model.eval()

    # Print memory summary
    print_qlora_summary(model)

    print(f"\n{'='*60}")
    print("QLoRA model ready for generation!")
    print(f"{'='*60}")

    # ========================================================================
    # Generate text
    # ========================================================================
    if args.greedy:
        print("\n--- Greedy Decoding ---")
        generate(
            model,
            tokenizer,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
            device=device,
        )
    else:
        print("\n--- Sampling ---")
        generate(
            model,
            tokenizer,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=False,
            device=device,
        )


if __name__ == "__main__":
    main()
