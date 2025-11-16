"""
create_preference_data.py: Create sample preference data for DPO training

This script helps you create preference data in the correct format for DPO.

Preference data consists of:
- Prompt: The input question/instruction
- Chosen: The preferred (better) response
- Rejected: The less preferred (worse) response

Output format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    ...

Usage:
    # Create sample data
    python -m scripts.data.create_preference_data --output preference_data.jsonl

    # Interactive mode (add your own examples)
    python -m scripts.data.create_preference_data --interactive --output my_prefs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


# Sample preference data for demonstration
SAMPLE_PREFERENCES = [
    {
        "prompt": "Explain what machine learning is.",
        "chosen": "Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed. It involves training algorithms on data to recognize patterns and make decisions. The system learns by analyzing examples and adjusting its internal parameters to minimize errors.",
        "rejected": "Machine learning is when computers learn stuff automatically.",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris, which is located in the north-central part of the country along the Seine River. Paris has been France's capital since the 12th century and is known for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "rejected": "Paris.",
    },
    {
        "prompt": "How do you make a peanut butter sandwich?",
        "chosen": "To make a peanut butter sandwich:\n1. Take two slices of bread\n2. Spread peanut butter on one slice using a knife\n3. Optionally add jelly or jam on the other slice\n4. Put the slices together with the spread sides facing each other\n5. Cut diagonally if desired and enjoy!",
        "rejected": "Put peanut butter on bread.",
    },
    {
        "prompt": "Why is the sky blue?",
        "chosen": "The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules. Blue light has a shorter wavelength than other colors, so it scatters more in all directions. This scattered blue light is what we see when we look up at the sky during the day.",
        "rejected": "Because of light and stuff in the air.",
    },
    {
        "prompt": "What is Python?",
        "chosen": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace and supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, scientific computing, and automation.",
        "rejected": "Python is a programming language.",
    },
    {
        "prompt": "Describe photosynthesis.",
        "chosen": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. During photosynthesis, these organisms use sunlight, water, and carbon dioxide to produce glucose (sugar) and oxygen. This occurs primarily in chloroplasts containing chlorophyll. The simplified equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. This process is essential for life on Earth as it produces oxygen and forms the base of most food chains.",
        "rejected": "Plants use sunlight to make food.",
    },
    {
        "prompt": "What is the difference between AI and machine learning?",
        "chosen": "Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart' or 'intelligent.' Machine Learning (ML) is a subset of AI that focuses specifically on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention. In other words, all machine learning is AI, but not all AI is machine learning. AI can also include rule-based systems, expert systems, and other approaches that don't involve learning from data.",
        "rejected": "AI is smarter than machine learning.",
    },
    {
        "prompt": "How does a computer work?",
        "chosen": "A computer works by processing information using several key components:\n1. CPU (Central Processing Unit): The 'brain' that executes instructions\n2. Memory (RAM): Temporarily stores data being actively used\n3. Storage: Permanently stores programs and files\n4. Input/Output: Devices for user interaction (keyboard, mouse, screen)\n\nWhen you run a program, instructions are loaded from storage into RAM, then the CPU fetches these instructions one by one, performs the requested operations, and stores results back in memory or storage. All of this happens at billions of cycles per second.",
        "rejected": "Computers have a processor that does calculations.",
    },
    {
        "prompt": "What causes seasons on Earth?",
        "chosen": "Seasons are caused by the tilt of Earth's rotational axis relative to its orbital plane around the Sun. Earth is tilted at approximately 23.5 degrees. As Earth orbits the Sun over the course of a year, different parts of the planet receive varying amounts of direct sunlight. When the Northern Hemisphere is tilted toward the Sun, it experiences summer while the Southern Hemisphere experiences winter, and vice versa. The equinoxes occur when neither hemisphere is tilted toward the Sun, resulting in roughly equal day and night lengths.",
        "rejected": "The Earth moves around the Sun and that changes the weather.",
    },
    {
        "prompt": "Explain what a blockchain is.",
        "chosen": "A blockchain is a distributed, decentralized digital ledger that records transactions across multiple computers. Each 'block' contains a list of transactions, a timestamp, and a cryptographic hash of the previous block, forming a chain. This structure makes it extremely difficult to alter past records because changing one block would require changing all subsequent blocks. Blockchains are maintained by a network of nodes that validate new transactions through consensus mechanisms. This technology underpins cryptocurrencies like Bitcoin but has applications in supply chain management, voting systems, and digital identity verification.",
        "rejected": "Blockchain is the technology behind Bitcoin.",
    },
]


def create_sample_data(output_path: str, num_examples: int = None):
    """
    Create sample preference data file.

    Args:
        output_path: Path to output JSONL file
        num_examples: Number of examples to include (None = all)
    """
    examples = SAMPLE_PREFERENCES
    if num_examples is not None:
        examples = examples[:num_examples]

    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Created sample preference data with {len(examples)} examples")
    print(f"Output: {output_path}")
    print(f"\nFormat:")
    print(f"  prompt: Input question/instruction")
    print(f"  chosen: Better response (detailed, helpful)")
    print(f"  rejected: Worse response (short, low-quality)")


def interactive_mode(output_path: str):
    """
    Interactive mode for creating preference data.

    User can input their own examples.
    """
    print("="*70)
    print("INTERACTIVE PREFERENCE DATA CREATION")
    print("="*70)
    print("Enter your own preference pairs.")
    print("For each example, provide:")
    print("  1. Prompt (the question or instruction)")
    print("  2. Chosen response (the better answer)")
    print("  3. Rejected response (the worse answer)")
    print("\nType 'done' for any field to finish.")
    print("="*70 + "\n")

    examples = []

    while True:
        print(f"\n--- Example {len(examples) + 1} ---")

        # Get prompt
        prompt = input("Prompt: ").strip()
        if prompt.lower() == 'done':
            break

        # Get chosen
        print("Chosen response (better, detailed):")
        chosen = input("> ").strip()
        if chosen.lower() == 'done':
            break

        # Get rejected
        print("Rejected response (worse, short):")
        rejected = input("> ").strip()
        if rejected.lower() == 'done':
            break

        # Add example
        examples.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
        })

        print(f"Added example {len(examples)}")

    if not examples:
        print("No examples added. Exiting.")
        return

    # Save to file
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n{'='*70}")
    print(f"Saved {len(examples)} preference pairs to {output_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Create preference data for DPO training")

    parser.add_argument("--output", type=str, default="preference_data.jsonl",
                        help="Output file path")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode (create your own examples)")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of sample examples to create (default: all)")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.output)
    else:
        create_sample_data(args.output, args.num_examples)

    print(f"\nNext steps:")
    print(f"1. Review the data: cat {args.output}")
    print(f"2. Train DPO:")
    print(f"   python -m scripts.training.train_dpo \\")
    print(f"     --data {args.output} \\")
    print(f"     --steps 1000")


if __name__ == "__main__":
    main()
