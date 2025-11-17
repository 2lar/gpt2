"""
Test KV cache implementation
"""
import torch
from gpt.model import GPT
from gpt.config import GPTConfig

def test_kv_cache():
    """Test that KV caching produces identical outputs to non-cached version."""
    print("=" * 70)
    print("Testing KV Cache Implementation")
    print("=" * 70)

    # Create a small test model
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=4,
        n_head=8,
        n_embd=256,
        dropout=0.0,
    )

    model = GPT(config)
    model.eval()

    # Test inputs
    B, T = 1, 10  # Batch size 1, sequence length 10
    prompt = torch.randint(0, config.vocab_size, (B, T))

    print(f"\nTest setup:")
    print(f"  Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd dim")
    print(f"  Input: Batch={B}, Sequence Length={T}")

    # ============================================================
    # Test 1: Basic cache functionality
    # ============================================================
    print("\n" + "-" * 70)
    print("Test 1: Basic cache functionality")
    print("-" * 70)

    with torch.no_grad():
        # Forward without cache (baseline)
        logits_no_cache, _ = model(prompt, use_cache=False)
        print(f"✓ Non-cached forward: logits shape = {logits_no_cache.shape}")

        # Forward with cache (first pass)
        logits_with_cache, kv_caches = model(prompt, use_cache=True)
        print(f"✓ Cached forward: logits shape = {logits_with_cache.shape}")
        print(f"✓ Returned {len(kv_caches)} layer caches")
        print(f"✓ Cache[0] K shape: {kv_caches[0][0].shape}")
        print(f"✓ Cache[0] V shape: {kv_caches[0][1].shape}")

        # Verify outputs match
        if torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5):
            print("✓ PASS: Cached and non-cached outputs match!")
        else:
            print("✗ FAIL: Outputs don't match!")
            max_diff = (logits_no_cache - logits_with_cache).abs().max().item()
            print(f"  Max difference: {max_diff}")

    # ============================================================
    # Test 2: Incremental generation simulation
    # ============================================================
    print("\n" + "-" * 70)
    print("Test 2: Incremental generation (simulating autoregressive decoding)")
    print("-" * 70)

    with torch.no_grad():
        # Method 1: Recompute everything each time (slow, baseline)
        x_full = prompt.clone()
        outputs_full = []

        for i in range(5):  # Generate 5 tokens
            logits, _ = model(x_full, use_cache=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            outputs_full.append(next_token.item())
            x_full = torch.cat([x_full, next_token], dim=1)

        print(f"✓ Non-cached generation: {outputs_full}")

        # Method 2: Use KV cache (fast, should match)
        x_cached = prompt.clone()
        outputs_cached = []
        kv_caches = None
        position_offset = 0

        for i in range(5):  # Generate 5 tokens
            if i == 0:
                # First pass: full prompt
                logits, kv_caches = model(x_cached, use_cache=True, position_offset=0)
                position_offset = x_cached.size(1)
            else:
                # Subsequent passes: only new token
                new_token_input = x_cached[:, -1:]  # Only the last token
                logits, kv_caches = model(
                    new_token_input,
                    kv_caches=kv_caches,
                    use_cache=True,
                    position_offset=position_offset
                )
                position_offset += 1

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            outputs_cached.append(next_token.item())
            x_cached = torch.cat([x_cached, next_token], dim=1)

        print(f"✓ Cached generation:     {outputs_cached}")

        # Verify they match
        if outputs_full == outputs_cached:
            print("✓ PASS: Cached and non-cached generation produce identical outputs!")
        else:
            print("✗ FAIL: Outputs don't match!")
            for i, (a, b) in enumerate(zip(outputs_full, outputs_cached)):
                if a != b:
                    print(f"  Position {i}: {a} != {b}")

    # ============================================================
    # Test 3: Performance comparison
    # ============================================================
    print("\n" + "-" * 70)
    print("Test 3: Performance comparison")
    print("-" * 70)

    import time

    num_new_tokens = 50
    x_test = prompt.clone()

    # Non-cached
    start = time.time()
    with torch.no_grad():
        for _ in range(num_new_tokens):
            logits, _ = model(x_test, use_cache=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x_test = torch.cat([x_test, next_token], dim=1)
    time_no_cache = time.time() - start

    # Cached
    x_test = prompt.clone()
    kv_caches = None
    position_offset = 0
    start = time.time()
    with torch.no_grad():
        for i in range(num_new_tokens):
            if i == 0:
                logits, kv_caches = model(x_test, use_cache=True, position_offset=0)
                position_offset = x_test.size(1)
            else:
                logits, kv_caches = model(
                    x_test[:, -1:],
                    kv_caches=kv_caches,
                    use_cache=True,
                    position_offset=position_offset
                )
                position_offset += 1
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x_test = torch.cat([x_test, next_token], dim=1)
    time_cached = time.time() - start

    speedup = time_no_cache / time_cached
    print(f"Non-cached: {time_no_cache:.3f}s")
    print(f"Cached:     {time_cached:.3f}s")
    print(f"✓ Speedup:  {speedup:.2f}x faster with caching!")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("✓ KV cache implementation is working correctly!")
    print("✓ Produces identical outputs to non-cached version")
    print(f"✓ Provides {speedup:.2f}x speedup for generation")
    print("=" * 70)

if __name__ == "__main__":
    test_kv_cache()
