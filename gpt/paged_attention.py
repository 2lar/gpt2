"""
Paged Attention Implementation

This module implements Paged Attention, a memory-efficient KV cache management technique
introduced in the vLLM paper (https://arxiv.org/abs/2309.06180).

=============================================================================
WHAT IS PAGED ATTENTION?
=============================================================================

Traditional KV cache stores keys and values in contiguous memory:
    Sequence 1: [K₁, K₂, K₃, K₄, K₅, ..., K_n]  ← Contiguous allocation
    Sequence 2: [K₁, K₂, K₃, ..., K_m]           ← Another contiguous allocation

Problems:
1. **Fragmentation**: If sequence 1 ends, the allocated space is wasted
2. **Over-allocation**: Must pre-allocate max_length memory per sequence
3. **No sharing**: Can't share KV cache between sequences (e.g., for beam search)

Paged Attention solution: Treat KV cache like virtual memory in OS!
- Divide KV cache into fixed-size **blocks** (like memory pages)
- Use **block tables** to map logical positions to physical blocks
- Share blocks between sequences (copy-on-write for beam search)

Example with block_size=4:
    Logical sequence: [K₀, K₁, K₂, K₃, K₄, K₅, K₆, K₇, K₈]

    Physical blocks:
        Block 0: [K₀, K₁, K₂, K₃]
        Block 2: [K₄, K₅, K₆, K₇]
        Block 5: [K₈, _, _, _]  ← Partially filled

    Block table: [0, 2, 5]  ← Maps logical to physical

Benefits:
- ✅ Near-zero memory waste (only last block partially filled)
- ✅ No fragmentation (blocks can be anywhere in memory)
- ✅ Easy sharing (just copy block table, not actual KV cache)
- ✅ Dynamic growth (allocate blocks on demand)

=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class BlockTable:
    """
    Block table for managing KV cache blocks.

    This maps logical sequence positions to physical memory blocks,
    similar to a page table in virtual memory.

    Example:
        block_size = 4
        sequence_length = 10

        Logical positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        Block mapping:
            Positions 0-3  → Block 0
            Positions 4-7  → Block 2
            Positions 8-9  → Block 5 (partially filled)

        block_table = [0, 2, 5]
    """

    def __init__(self, block_size: int):
        """
        Args:
            block_size: Number of tokens per block (typically 16-256)
        """
        self.block_size = block_size
        self.blocks: List[int] = []  # Physical block IDs

    def num_blocks(self) -> int:
        """Number of blocks allocated for this sequence."""
        return len(self.blocks)

    def add_block(self, block_id: int):
        """Allocate a new block for this sequence."""
        self.blocks.append(block_id)

    def get_block_id(self, logical_block_idx: int) -> int:
        """Get physical block ID for a logical block index."""
        return self.blocks[logical_block_idx]

    def get_all_blocks(self) -> List[int]:
        """Get all physical block IDs."""
        return self.blocks

    def copy(self) -> 'BlockTable':
        """Create a shallow copy (for copy-on-write in beam search)."""
        new_table = BlockTable(self.block_size)
        new_table.blocks = self.blocks.copy()
        return new_table


class BlockAllocator:
    """
    Manages allocation and deallocation of physical KV cache blocks.

    Similar to a memory allocator, this maintains a free list of available
    blocks and allocates them on demand.

    Example:
        allocator = BlockAllocator(num_blocks=8)

        # Initially: free_blocks = [0, 1, 2, 3, 4, 5, 6, 7]

        block_id = allocator.allocate()  # Returns 0
        # Now: free_blocks = [1, 2, 3, 4, 5, 6, 7]

        allocator.free(block_id)  # Returns block 0 to free list
        # Now: free_blocks = [0, 1, 2, 3, 4, 5, 6, 7]
    """

    def __init__(self, num_blocks: int):
        """
        Args:
            num_blocks: Total number of physical blocks available
        """
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks))
        self.allocated_blocks: set = set()

    def allocate(self) -> Optional[int]:
        """
        Allocate a free block.

        Returns:
            Physical block ID, or None if no blocks available (OOM)
        """
        if not self.free_blocks:
            return None  # Out of memory

        block_id = self.free_blocks.pop(0)
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        """Free a block back to the pool."""
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)

    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self.free_blocks)

    def get_num_free_tokens(self, block_size: int) -> int:
        """Number of tokens that can be stored in free blocks."""
        return self.num_free_blocks() * block_size


class PagedKVCache:
    """
    Paged KV cache storage using block-based allocation.

    Instead of storing KV cache as contiguous tensors per sequence,
    this stores them as blocks that can be allocated dynamically.

    Cache structure:
        k_cache: [num_blocks, num_heads, block_size, head_dim]
        v_cache: [num_blocks, num_heads, block_size, head_dim]

    Each sequence has a block table mapping logical positions to physical blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            num_blocks: Total number of physical blocks to allocate
            block_size: Number of tokens per block (e.g., 16, 32, 64)
            num_heads: Number of attention heads (for KV cache shape)
            head_dim: Dimension of each attention head
            dtype: Data type for cache tensors
            device: Device to store cache on
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Allocate physical memory for all blocks upfront
        # Shape: [num_blocks, num_heads, block_size, head_dim]
        self.k_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device=device
        )

        # Block allocator
        self.allocator = BlockAllocator(num_blocks)

        # Sequence ID to block table mapping
        self.block_tables: dict[int, BlockTable] = {}
        self.sequence_lengths: dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, initial_length: int = 0) -> bool:
        """
        Allocate blocks for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            initial_length: Initial number of tokens (for pre-allocation)

        Returns:
            True if successful, False if out of memory
        """
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already allocated")

        block_table = BlockTable(self.block_size)

        # Allocate blocks for initial length
        num_blocks_needed = math.ceil(initial_length / self.block_size) if initial_length > 0 else 0
        for _ in range(num_blocks_needed):
            block_id = self.allocator.allocate()
            if block_id is None:
                # OOM: free allocated blocks and return failure
                self._free_sequence(seq_id, block_table)
                return False
            block_table.add_block(block_id)

        self.block_tables[seq_id] = block_table
        self.sequence_lengths[seq_id] = initial_length
        return True

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        if seq_id not in self.block_tables:
            return

        block_table = self.block_tables[seq_id]
        self._free_sequence(seq_id, block_table)

    def _free_sequence(self, seq_id: int, block_table: BlockTable):
        """Internal helper to free blocks."""
        for block_id in block_table.get_all_blocks():
            self.allocator.free(block_id)

        if seq_id in self.block_tables:
            del self.block_tables[seq_id]
        if seq_id in self.sequence_lengths:
            del self.sequence_lengths[seq_id]

    def append_tokens(
        self,
        seq_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> bool:
        """
        Append new K, V tokens to a sequence's cache.

        Args:
            seq_id: Sequence identifier
            k_new: New keys [num_heads, num_new_tokens, head_dim]
            v_new: New values [num_heads, num_new_tokens, head_dim]

        Returns:
            True if successful, False if out of memory
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        block_table = self.block_tables[seq_id]
        current_length = self.sequence_lengths[seq_id]
        num_new_tokens = k_new.shape[1]

        # Calculate which blocks we need
        start_pos = current_length
        end_pos = current_length + num_new_tokens

        start_block = start_pos // self.block_size
        end_block = (end_pos - 1) // self.block_size + 1

        # Allocate new blocks if needed
        while block_table.num_blocks() < end_block:
            block_id = self.allocator.allocate()
            if block_id is None:
                return False  # Out of memory
            block_table.add_block(block_id)

        # Write new tokens to blocks
        token_idx = 0
        for pos in range(start_pos, end_pos):
            block_idx = pos // self.block_size
            offset_in_block = pos % self.block_size
            physical_block_id = block_table.get_block_id(block_idx)

            # Write to physical block
            self.k_cache[physical_block_id, :, offset_in_block, :] = k_new[:, token_idx, :]
            self.v_cache[physical_block_id, :, offset_in_block, :] = v_new[:, token_idx, :]
            token_idx += 1

        self.sequence_lengths[seq_id] = end_pos
        return True

    def get_kv_cache(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the full KV cache for a sequence.

        This assembles the cache from blocks into a contiguous tensor.

        Args:
            seq_id: Sequence identifier

        Returns:
            Tuple of (k, v) tensors, each [num_heads, seq_len, head_dim]
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        block_table = self.block_tables[seq_id]
        seq_len = self.sequence_lengths[seq_id]

        if seq_len == 0:
            return (
                torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
            )

        # Gather blocks and assemble into contiguous tensor
        k_list = []
        v_list = []

        for block_idx in range(block_table.num_blocks()):
            physical_block_id = block_table.get_block_id(block_idx)

            # Determine how many tokens to take from this block
            remaining = seq_len - block_idx * self.block_size
            tokens_in_block = min(self.block_size, remaining)

            k_list.append(self.k_cache[physical_block_id, :, :tokens_in_block, :])
            v_list.append(self.v_cache[physical_block_id, :, :tokens_in_block, :])

        k = torch.cat(k_list, dim=1)  # [num_heads, seq_len, head_dim]
        v = torch.cat(v_list, dim=1)  # [num_heads, seq_len, head_dim]

        return k, v

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> bool:
        """
        Fork a sequence (copy-on-write for beam search).

        Initially, child shares the same blocks as parent.
        When child writes, it copies the block first.

        Args:
            parent_seq_id: Source sequence
            child_seq_id: New sequence (must not exist)

        Returns:
            True if successful
        """
        if parent_seq_id not in self.block_tables:
            raise ValueError(f"Parent sequence {parent_seq_id} not found")
        if child_seq_id in self.block_tables:
            raise ValueError(f"Child sequence {child_seq_id} already exists")

        # Shallow copy: child shares parent's blocks initially
        parent_table = self.block_tables[parent_seq_id]
        child_table = parent_table.copy()

        self.block_tables[child_seq_id] = child_table
        self.sequence_lengths[child_seq_id] = self.sequence_lengths[parent_seq_id]

        # Note: Actual copy-on-write would require reference counting
        # For simplicity, we're doing a shallow copy here
        return True

    def get_num_free_tokens(self) -> int:
        """Number of tokens that can still be stored."""
        return self.allocator.get_num_free_tokens(self.block_size)


class PagedAttention(nn.Module):
    """
    Attention layer with paged KV cache support.

    This is a drop-in replacement for standard attention that uses
    paged KV cache for memory efficiency.

    Key differences from standard attention:
    1. KV cache is stored in blocks, not contiguous memory
    2. Sequences can share blocks (for beam search)
    3. Memory is allocated on-demand (no pre-allocation)
    4. Near-zero fragmentation
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            block_size: KV cache block size (number of tokens per block)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Q, K, V projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

        # Paged KV cache (will be initialized when needed)
        self.paged_cache: Optional[PagedKVCache] = None

    def init_paged_cache(
        self,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the paged KV cache.

        Args:
            num_blocks: Total number of blocks to allocate
            device: Device to store cache on
            dtype: Data type for cache
        """
        self.paged_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_heads=self.n_head,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_id: Optional[int] = None,
        use_paged_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional paged KV cache.

        Args:
            x: Input [B, T, n_embd]
            seq_id: Sequence ID for paged cache (required if use_paged_cache=True)
            use_paged_cache: Whether to use paged cache

        Returns:
            Output [B, T, n_embd]
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use paged cache if enabled
        if use_paged_cache and self.paged_cache is not None and seq_id is not None:
            # Append new K, V to paged cache
            k_new = k.squeeze(0)  # [n_head, T, head_dim]
            v_new = v.squeeze(0)

            success = self.paged_cache.append_tokens(seq_id, k_new, v_new)
            if not success:
                raise RuntimeError("Out of KV cache memory")

            # Retrieve full KV cache
            k_full, v_full = self.paged_cache.get_kv_cache(seq_id)
            k = k_full.unsqueeze(0)  # [1, n_head, full_len, head_dim]
            v = v_full.unsqueeze(0)

        # Compute attention
        attn_dropout = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=attn_dropout
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Paged Attention Demo")
    print("=" * 70)

    # Configuration
    block_size = 4
    num_blocks = 8
    n_embd = 64
    n_head = 4

    # Create paged attention layer
    attn = PagedAttention(n_embd=n_embd, n_head=n_head, block_size=block_size)
    attn.init_paged_cache(num_blocks=num_blocks, device=torch.device('cpu'))
    attn.eval()

    print(f"\nConfiguration:")
    print(f"  Block size: {block_size} tokens")
    print(f"  Total blocks: {num_blocks}")
    print(f"  Max capacity: {num_blocks * block_size} tokens")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Attention heads: {n_head}")

    # Allocate a sequence
    seq_id = 0
    attn.paged_cache.allocate_sequence(seq_id, initial_length=0)
    print(f"\n✓ Allocated sequence {seq_id}")

    # Simulate generation: add tokens one by one
    print("\nSimulating generation:")
    for i in range(10):
        x = torch.randn(1, 1, n_embd)  # One new token

        with torch.no_grad():
            y = attn(x, seq_id=seq_id, use_paged_cache=True)

        free_tokens = attn.paged_cache.get_num_free_tokens()
        seq_len = attn.paged_cache.sequence_lengths[seq_id]
        num_blocks_used = attn.paged_cache.block_tables[seq_id].num_blocks()

        print(f"  Step {i+1}: seq_len={seq_len}, blocks_used={num_blocks_used}, "
              f"free_tokens={free_tokens}")

    # Demonstrate block allocation
    print("\nBlock allocation details:")
    block_table = attn.paged_cache.block_tables[seq_id]
    print(f"  Physical blocks: {block_table.get_all_blocks()}")
    print(f"  Sequence length: {attn.paged_cache.sequence_lengths[seq_id]}")

    # Show memory efficiency
    total_capacity = num_blocks * block_size
    tokens_used = attn.paged_cache.sequence_lengths[seq_id]
    wasted_tokens = (block_table.num_blocks() * block_size) - tokens_used
    efficiency = (tokens_used / total_capacity) * 100

    print(f"\nMemory efficiency:")
    print(f"  Total capacity: {total_capacity} tokens")
    print(f"  Tokens used: {tokens_used}")
    print(f"  Wasted tokens: {wasted_tokens} (only in last block)")
    print(f"  Efficiency: {efficiency:.1f}%")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
