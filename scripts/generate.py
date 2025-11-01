import os
import torch
import torch.nn.functional as F
import argparse
from transformers import GPT2TokenizerFast
from gpt2.model import GPT
from gpt2.config import GPTConfig

# --- CLI args ---
parser = argparse.ArgumentParser(description="Generate text with custom GPT-2 implementation")
parser.add_argument("--source", choices=["pretrained", "local"], default="pretrained", help="Weight source")
parser.add_argument("--model", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2", help="Model size / config")
parser.add_argument("--ckpt", type=str, default=None, help="Path to local state_dict (.pt) when --source=local")
parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt", help="Path to HF-format weights when --source=pretrained")
parser.add_argument("--prompt", type=str, default="The three laws of robotics are:")
parser.add_argument("--max_new_tokens", type=int, default=50)
parser.add_argument("--temperature", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (overrides sampling flags)")
args, _ = parser.parse_known_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weights_path = None if args.weights in (None, '', 'auto') else args.weights

# --- Build/Load model ---
if args.source == "pretrained":
    if weights_path is None:
        model = GPT.from_pretrained(args.model).to(device).eval()
    else:
        if not os.path.exists(weights_path):
            raise SystemExit(f"Weights file not found: {weights_path}. Use README instructions to download/save GPT-2 weights.")
        model = GPT.from_pretrained(args.model, weights_path=weights_path).to(device).eval()
else:
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024
    model = GPT(GPTConfig(**config_args)).to(device).eval()
    if not args.ckpt:
        raise SystemExit("--ckpt is required when --source=local")
    print(f"Loading local checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)

# --- Tokenizer ---
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
eos_id = tokenizer.eos_token_id

# --- Generation Function ---
@torch.no_grad()
def generate(
    model,
    prompt_text,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    top_p=None,
    deterministic=False,
):
    model.eval()

    x = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    print(f"\n[PROMPT] {prompt_text}")
    print("=" * (len(prompt_text) + 12))

    if deterministic:
        print("MODE: Greedy Decoding (Deterministic)")
    elif top_p is not None:
        print(f"MODE: Nucleus (Top-p={top_p}) Sampling (Random)")
    elif top_k is not None:
        print(f"MODE: Top-k ({top_k}) Sampling (Random)")
    else:
        print(f"MODE: Pure Temperature ({temperature}) Sampling (Random)")

    for _ in range(max_new_tokens):
        block_size = getattr(model, 'config', None).block_size if hasattr(model, 'config') else 1024
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        logits, _ = model(x_cond)
        logits = logits[:, -1, :]

        if deterministic:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(1e-6, temperature)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits[logits < thresh] = -float('inf')

            if top_p is not None and top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                remove_indices = sorted_indices[sorted_indices_to_remove]
                logits[0, remove_indices] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, idx_next), dim=1)

        if idx_next.item() == eos_id:
            break

    decoded = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
    print(decoded)
    return decoded

if __name__ == "__main__":
    if args.greedy:
        print("\n--- Greedy Decoding ---")
        generate(
            model,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            deterministic=True,
        )
    else:
        print("\n--- Sampling ---")
        generate(
            model,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
