"""
prepare_custom_text.py: Prepare custom text file for fine-tuning

This script:
1. Loads a raw text file
2. Tokenizes using GPT-2 tokenizer (tiktoken)
3. Splits into train/val (90/10)
4. Saves as .npy shards for DataLoaderLite

Usage:
    python scripts/prepare_custom_text.py --input data/sampleBrainRot.txt --output brainrot_data
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tiktoken


def prepare_text_data(input_path: str, output_dir: str, train_ratio: float = 0.9):
    """
    Prepare custom text data for training.

    Args:
        input_path: Path to input text file
        output_dir: Directory to save train/val shards
        train_ratio: Fraction of data for training (rest is validation)
    """
    # 1. Load text
    print(f"Loading text from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")

    # 2. Tokenize using GPT-2 tokenizer
    print("Tokenizing...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.int32)

    print(f"Tokenized to {len(tokens):,} tokens")

    # 3. Split into train/val
    split_idx = int(len(tokens) * train_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train: {len(train_tokens):,} tokens ({train_ratio*100:.0f}%)")
    print(f"Val: {len(val_tokens):,} tokens ({(1-train_ratio)*100:.0f}%)")

    # 4. Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 5. Save as .npy shards (following DataLoaderLite convention)
    train_file = output_path / "train_00000.npy"
    val_file = output_path / "val_00000.npy"

    np.save(train_file, train_tokens)
    np.save(val_file, val_tokens)

    print(f"\nSaved to {output_dir}/")
    print(f"  - {train_file.name}: {len(train_tokens):,} tokens")
    print(f"  - {val_file.name}: {len(val_tokens):,} tokens")
    print("\nData preparation complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare custom text for fine-tuning")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sampleBrainRot.txt",
        help="Path to input text file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="brainrot_data",
        help="Output directory for train/val shards"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    prepare_text_data(args.input, args.output, args.train_ratio)


if __name__ == "__main__":
    main()
