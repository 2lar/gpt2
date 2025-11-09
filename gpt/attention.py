"""
Modern Frontier-Style Attention Mechanisms

This module implements cutting-edge attention mechanisms used in modern LLMs (2023-2024):
- Grouped-Query Attention (GQA) - Meta LLaMA 2/3, Mistral
- Rotary Position Embeddings (RoPE) - Most modern models
- Flash Attention 2 support
- KV caching for efficient inference

Compared to GPT-2 style (attention_gpt2.py), this provides:
  - 30-50% memory savings (GQA)
  - Better position encoding (RoPE)
  - 2x faster inference (Flash Attention 2 + KV cache)
  - Better length extrapolation
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import GPTConfig


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    What is RoPE?
    -------------
    Instead of adding position embeddings to tokens (like GPT-2), RoPE rotates the Query and Key
    vectors based on their position. This encodes positional information directly into the attention
    mechanism through rotation matrices.

    How it works (simplified):
    --------------------------
    1. For each position m, create rotation angles based on position and dimension
    2. Rotate Q and K vectors by these angles
    3. When computing attention Q @ K^T, the rotation ensures relative positions are encoded
    4. Token at position m attending to position n gets a rotation difference of (m-n)

    Mathematical intuition:
    -----------------------
    RoPE applies a rotation matrix R(m) to queries and keys:
        Q_m = R(m) @ Q    # Rotate query at position m
        K_n = R(n) @ K    # Rotate key at position n

    Attention score becomes:
        Q_m^T @ K_n = (R(m) @ Q)^T @ (R(n) @ K)
                    = Q^T @ R(m)^T @ R(n) @ K
                    = Q^T @ R(m-n) @ K    # Rotation property!

    This means attention depends on RELATIVE position (m-n), not absolute positions!

    Benefits over learned positions:
    ---------------------------------
      - Encodes relative positions (not just absolute)
      - Extrapolates to longer sequences than seen during training
      - No extra parameters (unlike learned embeddings)
      - Better performance empirically
      - Works seamlessly with attention mechanism

    Used by: LLaMA, GPT-NeoX, PaLM, Falcon, Mistral, and most models after 2021
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Initialize RoPE frequencies.

        Args:
            dim: Dimension of each attention head (typically n_embd // n_head = 64)
                 Must be even (we rotate pairs of dimensions)
            max_position_embeddings: Maximum sequence length to precompute (e.g., 2048)
                                    Can extrapolate beyond this if needed
            base: Base for the geometric progression (10000 in original paper)
                  Controls how quickly rotation frequencies decrease across dimensions
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies for rotation angles
        # We use different frequencies for different dimension pairs
        # Lower frequencies (slower rotation) for lower dimensions
        # Higher frequencies (faster rotation) for higher dimensions
        #
        # Formula: inv_freq[i] = 1 / (base^(2i/dim))
        # For dim=64, this gives 32 frequencies (one per dimension pair)
        #
        # Example for dim=64:
        #   inv_freq[0]  = 1 / 10000^(0/64)  = 1.0      (slowest rotation)
        #   inv_freq[16] = 1 / 10000^(32/64) = 0.01     (medium rotation)
        #   inv_freq[31] = 1 / 10000^(62/64) = 0.000464 (fastest rotation)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        # Register as buffer (not a parameter, but saved with model)
        # persistent=False means it won't be saved in state_dict (can be recomputed)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for common sequence lengths
        # This is an optimization - we could compute on-the-fly but caching is faster
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """
        Precompute cosine and sine values for positions 0 to seq_len-1.

        This is an optimization to avoid recomputing these values every forward pass.
        """
        self.max_seq_len_cached = seq_len

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Compute frequencies for each (position, dimension) pair
        # This is an outer product: position × inv_freq
        # Shape: (seq_len,) × (dim//2,) = (seq_len, dim//2)
        #
        # Example for position m and dimension pair i:
        #   freqs[m, i] = m * inv_freq[i]
        #
        # This gives us the rotation angle for position m in dimension i
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Concatenate frequencies to match full dimension
        # We apply the same rotation to both elements of each dimension pair
        # Shape: (seq_len, dim//2) -> (seq_len, dim)
        #
        # Example for dim=4:
        #   freqs = [[θ₀, θ₁], [θ₂, θ₃], ...]  (seq_len, 2)
        #   emb   = [[θ₀, θ₁, θ₀, θ₁], ...]    (seq_len, 4)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Precompute cos and sin of these angles
        # Shape: (seq_len, dim) -> (1, 1, seq_len, dim)
        # The extra dimensions are for broadcasting with (B, n_head, seq_len, head_dim)
        #
        # Why cos and sin?
        # Rotation matrix for angle θ:
        #   R(θ) = [[cos(θ), -sin(θ)]
        #           [sin(θ),  cos(θ)]]
        # We'll apply this to pairs of dimensions
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        The rotation is applied to pairs of dimensions:
        For dimensions [x₁, x₂, x₃, x₄, ...]:
          - Rotate (x₁, x₂) by angle θ₀
          - Rotate (x₃, x₄) by angle θ₁
          - And so on...

        Args:
            q: Query tensor of shape (B, n_head, T, head_dim)
            k: Key tensor of shape (B, n_kv_head, T, head_dim)
            seq_len: Sequence length T

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        # Extend cache if we encounter a longer sequence than precomputed
        # This allows the model to handle sequences longer than max_position_embeddings
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # Get precomputed cos and sin for this sequence length
        # Shape: (1, 1, seq_len, head_dim)
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        # Apply rotation using the formula:
        # rotated = x * cos + rotate_half(x) * sin
        #
        # This implements the 2D rotation matrix:
        #   [cos(θ), -sin(θ)]   [x₁]   [x₁*cos(θ) - x₂*sin(θ)]
        #   [sin(θ),  cos(θ)] @ [x₂] = [x₁*sin(θ) + x₂*cos(θ)]
        #
        # rotate_half(x) rearranges [x₁, x₂, x₃, x₄] -> [-x₃, -x₄, x₁, x₂]
        # so the formula becomes the rotation for each pair
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.

        This rearranges the vector to apply 2D rotations to dimension pairs.

        For a vector [x₁, x₂, x₃, x₄]:
          - First half:  [x₁, x₂]
          - Second half: [x₃, x₄]
          - Output: [-x₃, -x₄, x₁, x₂]

        When combined with the cos/sin formula in forward(), this applies:
          [x₁, x₂] -> [x₁*cos(θ) - x₂*sin(θ), x₁*sin(θ) + x₂*cos(θ)]
          [x₃, x₄] -> [x₃*cos(θ) - x₄*sin(θ), x₃*sin(θ) + x₄*cos(θ)]

        This is the 2D rotation formula from the RoPE paper.

        Args:
            x: Tensor of shape (..., dim) where dim is even

        Returns:
            Tensor of same shape with dimensions rotated
        """
        # Split into two halves along last dimension
        x1 = x[..., : x.shape[-1] // 2]  # First half:  [x₁, x₂, ...]
        x2 = x[..., x.shape[-1] // 2 :]  # Second half: [x₃, x₄, ...]

        # Return [-second_half, first_half]
        return torch.cat((-x2, x1), dim=-1)


# =============================================================================
# Grouped-Query Attention (GQA) Helper Function
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads in Grouped-Query Attention.

    What is this for?
    -----------------
    In GQA, we have fewer Key/Value heads than Query heads to save memory.
    For example, if n_head=32 and n_kv_head=8, we have:
      - 32 independent query heads
      - 8 key/value heads (each shared by 4 query heads)

    This function "expands" the 8 KV heads to 32 by repeating each one 4 times.

    Example:
    --------
    If n_head=32, n_kv_head=8, then n_rep=4
    Input KV tensor: (B, 8, T, head_dim)   # 8 KV heads
    Output tensor:   (B, 32, T, head_dim)  # 32 heads (each of 8 repeated 4 times)

    Visual representation for 1 batch, 1 token:
    Input:  [KV₀, KV₁, KV₂, KV₃, KV₄, KV₅, KV₆, KV₇]
    Output: [KV₀, KV₀, KV₀, KV₀, KV₁, KV₁, KV₁, KV₁, ..., KV₇, KV₇, KV₇, KV₇]
             └──── 4 copies ────┘ └──── 4 copies ────┘       └──── 4 copies ────┘

    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep) but faster.

    Args:
        hidden_states: Tensor of shape (B, n_kv_head, T, head_dim)
        n_rep: Number of repetitions per KV head (n_head // n_kv_head)

    Returns:
        Tensor of shape (B, n_head, T, head_dim)
    """
    # If n_rep=1, we have standard MHA (n_head == n_kv_head), no repetition needed
    if n_rep == 1:
        return hidden_states

    batch, n_kv_heads, slen, head_dim = hidden_states.shape

    # Step 1: Add a new dimension and expand
    # (B, n_kv_head, T, head_dim) -> (B, n_kv_head, 1, T, head_dim)
    # Then expand: (B, n_kv_head, n_rep, T, head_dim)
    #
    # This creates n_rep copies of each KV head without actually copying memory
    # (expand is a view operation, not a copy)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)

    # Step 2: Reshape to merge n_kv_head and n_rep dimensions
    # (B, n_kv_head, n_rep, T, head_dim) -> (B, n_kv_head * n_rep, T, head_dim)
    #                                      = (B, n_head, T, head_dim)
    #
    # This interleaves the repetitions:
    # [KV₀, KV₀, KV₀, KV₀, KV₁, KV₁, KV₁, KV₁, ...]
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# =============================================================================
# Grouped-Query Attention (GQA)
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) - Modern attention mechanism with memory efficiency.

    What is GQA?
    ------------
    GQA is a memory-efficient variant of Multi-Head Attention that reduces the size of
    the KV cache during inference by having multiple query heads share the same key/value heads.

    Comparison of attention mechanisms:
    ------------------------------------

    1. Multi-Head Attention (MHA) - Original Transformer:
       - Each head has its own Q, K, V
       - n_head query heads, n_head key heads, n_head value heads
       - Example: 32 Q heads, 32 K heads, 32 V heads
       - KV cache size: Large (all heads)
       - Quality: Best

    2. Multi-Query Attention (MQA) - PaLM, Falcon:
       - All heads share a single K, V
       - n_head query heads, 1 key head, 1 value head
       - Example: 32 Q heads, 1 K head, 1 V head
       - KV cache size: Smallest (only 1 head)
       - Quality: Slight degradation

    3. Grouped-Query Attention (GQA) - LLaMA 2, Mistral:
       - Groups of query heads share K, V heads
       - n_head query heads, n_kv_head key/value heads (n_kv_head < n_head)
       - Example: 32 Q heads, 8 KV heads (groups of 4 Q heads share 1 KV head)
       - KV cache size: Medium (between MHA and MQA)
       - Quality: Nearly identical to MHA

    Example configuration (LLaMA 2 style):
    ---------------------------------------
        n_head = 32        # 32 query heads
        n_kv_head = 8      # 8 key/value heads
        n_rep = 4          # Each KV head is shared by 4 query heads

    Memory savings:
        MHA:  32 K heads + 32 V heads = 64 heads in KV cache
        GQA:  8 K heads + 8 V heads = 16 heads in KV cache
        Savings: 75% reduction in KV cache size!

    How it works:
    -------------
    1. Project input to 32 query heads and 8 key/value heads
    2. Repeat each of the 8 KV heads 4 times to get 32 KV heads
    3. Perform standard attention with 32 Q heads and 32 KV heads
    4. During inference, only cache 8 KV heads (not 32)

    Benefits:
    ---------
      - 30-75% less KV cache memory (critical for long context inference)
      - Faster inference (less data to load from memory)
      - Nearly identical quality to MHA (< 1% degradation)
      - Scales better to very large models
      - Better than MQA which has noticeable quality loss

    Used by: LLaMA 2, LLaMA 3, Mistral, Mixtral, Gemma, many modern models
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # GQA configuration parameters
        self.n_embd = config.n_embd          # Total embedding dimension (e.g., 768)
        self.n_head = config.n_head          # Number of query heads (e.g., 32)
        self.n_kv_head = getattr(config, 'n_kv_head', config.n_head)  # Number of KV heads (e.g., 8)
        self.n_rep = self.n_head // self.n_kv_head  # Repetition factor (e.g., 4)
        self.head_dim = self.n_embd // self.n_head  # Dimension per head (e.g., 24)

        # Validate configuration
        # n_head must be divisible by n_kv_head so we can evenly group queries
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        # n_embd must be divisible by n_head for even head dimension
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        # Projection layers
        # ----------------
        # Note: Different from GPT-2 which uses a single combined QKV projection
        #
        # Query projection: n_embd -> n_head * head_dim
        #   Projects input to all query heads
        #   Example: 768 -> 32 * 24 = 768
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)

        # Key/Value projection: n_embd -> 2 * n_kv_head * head_dim
        #   Projects input to FEWER key/value heads (this is the GQA magic!)
        #   Example: 768 -> 2 * 8 * 24 = 384 (half the size of Q projection!)
        #   The "2 *" is for both keys and values
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_kv_head * self.head_dim, bias=config.bias)

        # Output projection: n_embd -> n_embd
        #   Projects concatenated head outputs back to embedding dimension
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        # Special flag for scaled initialization (GPT-2 technique)
        # This helps stabilize training in deep networks by scaling down residual contributions
        self.o_proj.NANOGPT_SCALE_INIT = 1

        # Rotary Position Embeddings (instead of learned absolute positions)
        # This is how modern models encode position information
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 2048)
        )

        # Regularization
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check if Flash Attention is available (PyTorch 2.0+)
        # Flash Attention is a memory-efficient attention implementation
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Grouped-Query Attention + RoPE + optional KV caching.

        KV Caching Explanation:
        -----------------------
        During inference (e.g., text generation), we generate one token at a time.
        Without caching:
          - For token 1: Compute attention for token 1
          - For token 2: Compute attention for tokens 1, 2 (recompute token 1!)
          - For token 3: Compute attention for tokens 1, 2, 3 (recompute 1 and 2!)
          - Very wasteful!

        With KV caching:
          - For token 1: Compute K₁, V₁, cache them
          - For token 2: Compute K₂, V₂, use cached K₁, V₁
          - For token 3: Compute K₃, V₃, use cached K₁, K₂, V₁, V₂
          - Much faster!

        GQA makes caching even more efficient because we only cache n_kv_head heads
        instead of n_head heads.

        Args:
            x: Input tensor of shape (B, T, n_embd)
               B = batch size
               T = sequence length (e.g., 1 during generation, longer during training)
               n_embd = embedding dimension
            kv_cache: Optional cached (key, value) from previous forward passes
                     Tuple of (cached_keys, cached_values)
                     Each has shape (B, n_kv_head, past_T, head_dim)
                     Only used during inference for speed
            use_cache: Whether to return updated KV cache for next iteration
                      Set to True during inference, False during training

        Returns:
            output: Attention output of shape (B, T, n_embd)
            new_kv_cache: Updated KV cache if use_cache=True, else None
                         Tuple of (keys, values) each of shape (B, n_kv_head, total_T, head_dim)
        """
        B, T, C = x.size()  # B=batch, T=sequence length, C=n_embd

        # Step 1: Project input to Queries, Keys, Values
        # -----------------------------------------------
        # Query projection: (B, T, n_embd) -> (B, T, n_head * head_dim)
        #   Creates ALL query heads
        #   Example: (4, 10, 768) -> (4, 10, 768)
        q = self.q_proj(x)

        # Key/Value projection: (B, T, n_embd) -> (B, T, 2 * n_kv_head * head_dim)
        #   Creates FEWER key/value heads (the GQA efficiency gain!)
        #   Example: (4, 10, 768) -> (4, 10, 384)
        #   This is where GQA saves memory - fewer parameters in projection!
        kv = self.kv_proj(x)

        # Split KV into separate K and V tensors
        # (B, T, 2 * n_kv_head * head_dim) -> 2 × (B, T, n_kv_head * head_dim)
        k, v = kv.split(self.n_kv_head * self.head_dim, dim=-1)

        # Step 2: Reshape for multi-head attention
        # -----------------------------------------
        # Query: Reshape and transpose to get head dimension in the right place
        # (B, T, n_head * head_dim) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        #   Example: (4, 10, 768) -> (4, 10, 32, 24) -> (4, 32, 10, 24)
        #   Now we have 32 separate query heads, each with 24 dimensions
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Keys: Reshape with FEWER heads
        # (B, T, n_kv_head * head_dim) -> (B, T, n_kv_head, head_dim) -> (B, n_kv_head, T, head_dim)
        #   Example: (4, 10, 192) -> (4, 10, 8, 24) -> (4, 8, 10, 24)
        #   Only 8 key heads instead of 32!
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Values: Same as keys
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Step 3: Apply Rotary Position Embeddings (RoPE)
        # ------------------------------------------------
        # This encodes position information by rotating Q and K vectors
        # No position embeddings are added to the input; instead, Q and K are rotated
        # based on their position in the sequence
        #
        # After this, Q and K encode both content and position information
        q, k = self.rotary_emb(q, k, T)

        # Step 4: Handle KV cache (for efficient inference)
        # --------------------------------------------------
        if kv_cache is not None:
            # During generation, we've already computed K and V for previous tokens
            # past_k, past_v: (B, n_kv_head, past_T, head_dim)
            past_k, past_v = kv_cache

            # Concatenate past and current keys/values
            # (B, n_kv_head, past_T, head_dim) + (B, n_kv_head, T, head_dim)
            # -> (B, n_kv_head, past_T + T, head_dim)
            #
            # Example during generation:
            #   past_T = 9 (already generated 9 tokens)
            #   T = 1 (generating 1 new token)
            #   Result: total_T = 10
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Store updated cache for next iteration
        # Only done during inference (use_cache=True)
        new_kv_cache = (k, v) if use_cache else None

        # Step 5: Repeat KV heads to match query heads (GQA magic!)
        # ----------------------------------------------------------
        # We have 32 query heads but only 8 key/value heads
        # Repeat each KV head 4 times: 8 heads -> 32 heads
        #
        # Before: k has shape (B, 8, T, head_dim)
        # After:  k has shape (B, 32, T, head_dim)
        #
        # Grouping pattern:
        #   Q heads 0-3  use KV head 0
        #   Q heads 4-7  use KV head 1
        #   Q heads 8-11 use KV head 2
        #   ... and so on
        #
        # If n_rep=1 (n_head == n_kv_head), this is a no-op (standard MHA)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Step 6: Compute attention scores and apply attention
        # -----------------------------------------------------
        if self.flash:
            # Use Flash Attention (PyTorch 2.0+ built-in)
            # This is a memory-efficient implementation of attention
            # - Fuses operations (no intermediate attention matrix stored)
            # - Uses tiling to fit in GPU cache
            # - 2-4× faster than manual implementation
            # - Identical mathematical result
            attn_dropout = self.dropout_p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,      # Apply causal masking (can't attend to future)
                dropout_p=attn_dropout
            )
        else:
            # Fallback: Manual attention computation (if Flash Attention not available)
            # This is the standard attention mechanism from "Attention Is All You Need"

            # Compute attention scores: Q @ K^T / sqrt(d_k)
            # (B, n_head, T_q, head_dim) @ (B, n_head, head_dim, T_k) -> (B, n_head, T_q, T_k)
            #
            # Scaling by sqrt(head_dim) prevents dot products from getting too large
            # (which would make softmax saturate and gradients vanish)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask to prevent attending to future tokens
            # During training on "The cat sat", when processing "cat":
            #   Can attend to: "The", "cat"
            #   Cannot attend to: "sat" (future token)
            #
            # Create lower triangular mask (1s below diagonal, 0s above)
            T_q, T_k = q.size(2), k.size(2)
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=x.device)).view(1, 1, T_q, T_k)

            # Set future positions to -inf so softmax gives them 0 probability
            att = att.masked_fill(causal_mask == 0, float('-inf'))

            # Apply softmax to get attention probabilities
            # Each row sums to 1.0 (probability distribution over keys)
            att = F.softmax(att, dim=-1)

            # Apply attention dropout (randomly zero out some attention weights)
            if self.training and self.dropout_p > 0:
                att = F.dropout(att, p=self.dropout_p)

            # Apply attention weights to values
            # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
            #
            # This is a weighted average of value vectors based on attention probabilities
            y = att @ v

        # Step 7: Reassemble all attention heads
        # ---------------------------------------
        # Transpose: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        #   Move sequence dimension back to position 1
        #
        # View: (B, T, n_head, head_dim) -> (B, T, n_head * head_dim) = (B, T, n_embd)
        #   Concatenate all heads back into single embedding dimension
        #   Example: (4, 10, 32, 24) -> (4, 10, 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Output projection and dropout
        # --------------------------------------
        # Project concatenated heads back to embedding dimension
        # This is the "W^O" matrix from the Transformer paper
        # (B, T, n_embd) -> (B, T, n_embd)
        y = self.resid_dropout(self.o_proj(y))

        return y, new_kv_cache


# =============================================================================
# Sliding Window Attention (Optional - for very long contexts)
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention - For efficient handling of very long sequences.

    What is Sliding Window Attention?
    ----------------------------------
    Standard attention has quadratic complexity O(T²) in sequence length:
      - Each of T tokens attends to all T previous tokens
      - Attention matrix is T × T
      - For T=100,000 tokens, this is 10 billion operations!

    Sliding window attention has linear complexity O(T × W):
      - Each token only attends to the last W tokens (window size)
      - Attention matrix is T × W
      - For T=100,000 and W=4096, this is only 400 million operations (25× faster!)

    How it works:
    -------------
    For token at position i:
      Standard attention: attend to all tokens [0, 1, 2, ..., i-1, i]
      Sliding window:     attend to tokens [max(0, i-W+1), ..., i-1, i]

    Example with window_size=3:
      Token 0: attends to [0]                (start of sequence)
      Token 1: attends to [0, 1]
      Token 2: attends to [0, 1, 2]
      Token 3: attends to [1, 2, 3]          (window starts sliding)
      Token 4: attends to [2, 3, 4]
      Token 5: attends to [3, 4, 5]
      ...

    Attention mask pattern (W=3):
         0  1  2  3  4  5
      0  1  0  0  0  0  0
      1  1  1  0  0  0  0
      2  1  1  1  0  0  0
      3  0  1  1  1  0  0    <- Sliding window (only last 3)
      4  0  0  1  1  1  0
      5  0  0  0  1  1  1

    Benefits:
    ---------
      - Linear complexity in sequence length
      - Can handle sequences of 100K+ tokens
      - Constant memory regardless of sequence length
      - Still preserves recent context

    Trade-offs:
    -----------
      - Can't see very distant tokens (limited by window)
      - May lose long-range dependencies
      - Best for tasks where recent context matters most

    When to use:
    ------------
      - Very long documents (100K+ tokens)
      - Real-time streaming applications
      - Limited memory/compute environments
      - Tasks where distant context is less important

    Used by: Mistral (window=4096), Longformer, BigBird

    Note: This implementation combines sliding window with GQA and RoPE!
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Same configuration as GQA
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, 'n_kv_head', config.n_head)
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = self.n_embd // self.n_head

        # Sliding window size (e.g., 4096 tokens)
        # Larger window = more context but more computation
        # Smaller window = less context but faster
        self.window_size = getattr(config, 'window_size', 4096)

        # Projections (same as GQA)
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.o_proj.NANOGPT_SCALE_INIT = 1

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Dropout
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.size()

        # Project and reshape (same as GQA)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x)
        k, v = kv.split(self.n_kv_head * self.head_dim, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, T)

        # Repeat KV for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Compute attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create sliding window causal mask
        # Token i can attend to tokens [max(0, i-window_size+1), ..., i]
        #
        # Step 1: Create all-ones matrix
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool)

        # Step 2: Make it upper triangular with offset
        # triu(diagonal=-W+1) keeps elements on and above the (-W+1)th diagonal
        # This creates the "sliding" part of the window
        mask = torch.triu(mask, diagonal=-self.window_size + 1)

        # Step 3: Make it lower triangular (causal)
        # tril keeps elements on and below the main diagonal
        # This ensures we don't attend to future tokens
        mask = torch.tril(mask)

        # Add broadcast dimensions: (T, T) -> (1, 1, T, T)
        mask = mask.view(1, 1, T, T)

        # Apply mask: Set positions outside window to -inf
        # Softmax will convert -inf to 0 probability
        att = att.masked_fill(~mask, float('-inf'))

        # Apply softmax
        att = F.softmax(att, dim=-1)

        # Apply dropout
        if self.training and self.dropout_p > 0:
            att = F.dropout(att, p=self.dropout_p)

        # Apply attention to values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v

        # Reassemble heads and project
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.o_proj(y))

        return y


# =============================================================================
# Factory Function
# =============================================================================

def create_attention(config: GPTConfig, attention_type: str = 'gqa') -> nn.Module:
    """
    Factory function to create the desired attention mechanism.

    This allows easy switching between different attention types.

    Args:
        config: Model configuration (GPTConfig)
        attention_type: One of:
                       - 'gqa': Grouped-Query Attention (default, recommended)
                       - 'sliding_window': Sliding Window Attention (for very long contexts)

    Returns:
        Attention module (nn.Module)

    Raises:
        ValueError: If attention_type is not recognized

    Example:
        # Create GQA attention (default)
        attn = create_attention(config)

        # Create sliding window attention for long documents
        attn = create_attention(config, attention_type='sliding_window')
    """
    if attention_type == 'gqa':
        return GroupedQueryAttention(config)
    elif attention_type == 'sliding_window':
        return SlidingWindowAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
