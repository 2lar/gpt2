# GPT-2 Reproduction Workspace

Ensure that you also set up the local virtual environment properly, too.
1. in the workspace run `python3 -m venv env`
2. then to start the virtual environment run `source env/bin/activate`
3. then you can run `pip install -r requirements.txt`

If you keep `data/gpt2_weights.pt` in place, both commands default to it automatically.

## Project Layout
- `src/config.py`, `src/attention.py`, `src/block.py`, `src/model.py`: modular GPT-2 implementation.
- `src/validate_model.py`: end-to-end validation (weights, logits, generation) against Hugging Face.
- `src/generate_text.py`: CLI for text generation using the local model.
- `src/example.py`: Karpathy's original script (kept for reference).
- `data/gpt2_weights.pt`: locally cached Hugging Face GPT-2 state dict (place the downloaded file here). Keys are auto-normalized so either reference the Hugging Face naming scheme or a stripped variant.

## Environment
```bash
pip install -r requirements.txt
```

## Local Weight Management
- Download the GPT-2 checkpoint once (e.g. with `transformers`):
  ```python
  from transformers import GPT2LMHeadModel
  import torch
  model = GPT2LMHeadModel.from_pretrained('gpt2')
  torch.save(model.state_dict(), 'data/gpt2_weights.pt')
  ```
- All commands below accept `--weights data/gpt2_weights.pt` to stay offline. The scripts default to this path, so no flag is needed unless you stash the file elsewhere. Pass `--weights auto` if you prefer to let `transformers` download on demand.

## Validation Pipeline
```bash
# Compare local implementation with Hugging Face on weights, logits, and greedy generation
python src/validate_model.py \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --prompt "The quick brown fox" \
  --gen_prompt "Hello, my name is Larry and I'm currently building a GPT model from scratch - but loading" \
  --max_new_tokens 70
```

Output highlights:
- Parameter counts for both models
- Weight parity check (with Conv1D→Linear transposes applied)
- Logit differences for the chosen prompt
- Side-by-side greedy generations

## Text Generation
```bash
# Using the public GPT-2 weights stored locally
python src/generate_text.py \
  --source pretrained \
  --model gpt2 \
  --weights data/gpt2_weights.pt \
  --prompt "The three laws of robotics are:" \
  --max_new_tokens 50 \
  --greedy            # drop this flag to enable sampling

# Using your own fine-tuned checkpoint (must match the config implied by --model)
python src/generate_text.py \
  --source local \
  --model gpt2 \
  --ckpt path/to/custom_state_dict.pt \
  --prompt "Custom run" \
  --max_new_tokens 60 \
  --temperature 0.8 \
  --top_p 0.95
```

Generation notes:
- Greedy mode prints deterministic completions.
- Sampling mode supports temperature, top-k, and top-p.
- Token IDs come from `GPT2TokenizerFast`, guaranteeing matching vocab with Hugging Face.

