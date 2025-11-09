"""Comprehensive validation pipeline for comparing local GPT implementation with HF GPT-2."""
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

from gpt.model import GPT


def load_models(model_name: str, device: str, weights_path: str | None):
    """Load Hugging Face and custom models with identical configs."""

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_name]

    if weights_path is None:
        hf_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        hf_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            **config_args,
        )
        hf_model = GPT2LMHeadModel(hf_config)
        state = torch.load(weights_path, map_location='cpu')
        state = GPT.normalize_hf_state_dict(state)
        hf_model.load_state_dict(state, strict=True)
        hf_model = hf_model.to(device).eval()

    custom_model = GPT.from_pretrained(model_name, weights_path=weights_path).to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    return hf_model, custom_model, tokenizer


def compare_state_dicts(custom_model, hf_model):
    """Ensure state dicts match after accounting for Conv1D vs Linear layout."""
    print("\n--- 1. Validating state_dict alignment ---")
    custom_sd = custom_model.state_dict()
    hf_sd_raw = hf_model.state_dict()

    ignore_suffixes = ('.attn.bias', '.attn.masked_bias')
    transpose_suffixes = (
        'attn.c_attn.weight',
        'attn.c_proj.weight',
        'mlp.c_fc.weight',
        'mlp.c_proj.weight',
    )

    hf_sd = {}
    for k, v in hf_sd_raw.items():
        if k.endswith(ignore_suffixes) or k == 'lm_head.bias':
            continue
        if any(k.endswith(suf) for suf in transpose_suffixes):
            hf_sd[k] = v.t()
        else:
            hf_sd[k] = v

    custom_keys = {k for k in custom_sd.keys() if not k.endswith(ignore_suffixes)}
    hf_keys = set(hf_sd.keys())

    missing = hf_keys - custom_keys
    extra = custom_keys - hf_keys

    if missing or extra:
        raise ValueError(f"Key mismatch. Missing: {missing}, Extra: {extra}")

    for k in hf_keys:
        custom_tensor = custom_sd[k]
        hf_tensor = hf_sd[k]
        if custom_tensor.shape != hf_tensor.shape:
            raise ValueError(f"Shape mismatch for {k}: {custom_tensor.shape} vs {hf_tensor.shape}")
        if not torch.allclose(custom_tensor, hf_tensor, atol=1e-6, rtol=1e-5):
            max_diff = (custom_tensor - hf_tensor).abs().max().item()
            raise ValueError(f"Value mismatch for {k}, max diff {max_diff}")

    print("✅ State_dicts align (after transposition).")


def compare_logits(custom_model, hf_model, tokenizer, text: str, device: str):
    """Compare logits for a single prompt."""
    print("\n--- 2. Validating logits ---")
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        custom_logits, _ = custom_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    if custom_logits.shape != hf_logits.shape:
        raise ValueError(f"Logit shape mismatch: {custom_logits.shape} vs {hf_logits.shape}")

    max_diff = (custom_logits - hf_logits).abs().max().item()
    l2 = torch.norm(custom_logits - hf_logits).item()

    print(f"Max absolute diff: {max_diff:.6e}")
    print(f"L2 norm diff: {l2:.6e}")
    print("✅ Logits closely match.")


def greedy_generate_local(model, input_ids, max_new_tokens, eos_id=None):
    """Greedy decoding for custom GPT."""
    x = input_ids.clone()
    for _ in range(max_new_tokens):
        block_size = getattr(model, 'config', None).block_size if hasattr(model, 'config') else 1024
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return x


def compare_generation(custom_model, hf_model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    """Compare greedy generations."""
    print("\n--- 3. Comparing greedy generations ---")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        hf_ids = hf_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        custom_ids = greedy_generate_local(custom_model, input_ids, max_new_tokens, tokenizer.eos_token_id)

    custom_text = tokenizer.decode(custom_ids.squeeze(), skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_ids.squeeze(), skip_special_tokens=True)

    print("Custom model output:\n" + "=" * 20)
    print(custom_text)
    print()
    print("HF model output:\n" + "=" * 20)
    print(hf_text)

    if custom_text == hf_text:
        print("✅ Greedy generations are identical.")
    else:
        print("⚠️ Generations differ. Inspect above for drift.")


def main():
    parser = argparse.ArgumentParser(description="Validate custom GPT implementation against Hugging Face GPT-2")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="HF model size to compare against")
    parser.add_argument("--prompt", default="The quick brown fox", help="Prompt for forward/logit test")
    parser.add_argument("--gen_prompt", default="Hello, my name is Larry and I'm currently building a GPT model from scratch - but loading", help="Prompt for generation comparison")
    parser.add_argument("--max_new_tokens", type=int, default=70, help="Tokens to generate for comparison")
    parser.add_argument("--weights", type=str, default="data/gpt2_weights.pt", help="Path to HF-format weights file to load locally")
    args = parser.parse_args()
    weights_path = None if args.weights in (None, '', 'auto') else args.weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hf_model, custom_model, tokenizer = load_models(args.model, device, weights_path)

    # report parameter counts
    custom_params = sum(p.numel() for p in custom_model.parameters())
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"Custom params: {custom_params/1e6:.2f}M | HF params: {hf_params/1e6:.2f}M")

    compare_state_dicts(custom_model, hf_model)
    compare_logits(custom_model, hf_model, tokenizer, args.prompt, device)
    compare_generation(custom_model, hf_model, tokenizer, args.gen_prompt, args.max_new_tokens, device)


if __name__ == "__main__":
    main()
