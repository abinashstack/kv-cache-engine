"""
Paged KV cache implementation.

Stores KV tensors in fixed-size blocks with a per-sequence block table,
eliminating memory fragmentation compared to one contiguous tensor per
sequence. Block IDs are shared across all transformer layers for a given
sequence (all layers advance token-by-token in lockstep); each layer gets
its own physical storage slice.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class BlockAllocatorConfig:
    block_size: int = 16  # tokens per block
    num_blocks: int = 1024  # total blocks available (shared across layers)
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64


class BlockAllocator:
    """Owns physical KV storage and hands out/reclaims block IDs."""

    def __init__(self, config: BlockAllocatorConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.free_blocks = list(range(config.num_blocks))

        # [num_layers, num_blocks, block_size, num_heads, head_dim]
        shape = (
            config.num_layers,
            config.num_blocks,
            config.block_size,
            config.num_heads,
            config.head_dim,
        )
        self.key_cache = torch.zeros(shape, device=device, dtype=torch.float32)
        self.value_cache = torch.zeros(shape, device=device, dtype=torch.float32)

    def allocate(self, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(
                f"Insufficient blocks: need {num_blocks}, have {len(self.free_blocks)}"
            )
        return [self.free_blocks.pop() for _ in range(num_blocks)]

    def free(self, block_ids: List[int]) -> None:
        self.free_blocks.extend(block_ids)

    def get_block(self, layer_idx: int, block_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx, block_id], self.value_cache[layer_idx, block_id]


class SequenceBlockTable:
    """Logical token position -> physical block ID, shared across layers."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: List[int] = []
        self.token_count = 0

    def get_blocks(self) -> List[int]:
        return self.blocks.copy()


class PagedKVCache:
    """Manages paged, multi-layer KV cache for multiple concurrent sequences."""

    def __init__(self, config: BlockAllocatorConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.allocator = BlockAllocator(config, device)
        self.sequences: Dict[int, SequenceBlockTable] = {}
        self.seq_counter = 0

    def new_sequence(self) -> int:
        seq_id = self.seq_counter
        self.seq_counter += 1
        self.sequences[seq_id] = SequenceBlockTable(self.config.block_size)
        return seq_id

    def _plan_write(
        self, seq: SequenceBlockTable, num_new_tokens: int
    ) -> List[Tuple[int, int, int]]:
        """Allocate any needed blocks (once) and return [(block_id, offset, count), ...]."""
        block_size = self.config.block_size
        plan = []
        remaining = num_new_tokens
        pos = seq.token_count

        while remaining > 0:
            block_idx = pos // block_size
            offset = pos % block_size

            if block_idx == len(seq.blocks):
                seq.blocks.append(self.allocator.allocate(1)[0])
            block_id = seq.blocks[block_idx]

            count = min(block_size - offset, remaining)
            plan.append((block_id, offset, count))
            pos += count
            remaining -= count

        return plan

    def add_kv(
        self,
        seq_id: int,
        keys_per_layer: List[torch.Tensor],
        values_per_layer: List[torch.Tensor],
    ) -> None:
        """Append new KV tokens for every layer of a sequence.

        keys_per_layer[i] / values_per_layer[i]: [num_new_tokens, num_heads, head_dim]
        """
        seq = self.sequences[seq_id]
        num_new_tokens = keys_per_layer[0].shape[0]
        plan = self._plan_write(seq, num_new_tokens)

        for layer_idx in range(self.config.num_layers):
            key = keys_per_layer[layer_idx]
            value = values_per_layer[layer_idx]
            src_offset = 0
            for block_id, offset, count in plan:
                k_block, v_block = self.allocator.get_block(layer_idx, block_id)
                k_block[offset : offset + count] = key[src_offset : src_offset + count]
                v_block[offset : offset + count] = value[src_offset : src_offset + count]
                src_offset += count

        seq.token_count += num_new_tokens

    def get_kv(self, seq_id: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Reconstruct full KV cache for every layer: list of (keys, values), each
        shaped [token_count, num_heads, head_dim]."""
        seq = self.sequences[seq_id]
        result = []
        for layer_idx in range(self.config.num_layers):
            keys, values = [], []
            for block_id in seq.blocks:
                k, v = self.allocator.get_block(layer_idx, block_id)
                keys.append(k)
                values.append(v)
            full_keys = torch.cat(keys, dim=0)[: seq.token_count]
            full_values = torch.cat(values, dim=0)[: seq.token_count]
            result.append((full_keys, full_values))
        return result

    def free_sequence(self, seq_id: int) -> None:
        if seq_id in self.sequences:
            self.allocator.free(self.sequences[seq_id].get_blocks())
            del self.sequences[seq_id]

    def memory_usage(self) -> Dict[str, float]:
        total_blocks = self.config.num_blocks
        used_blocks = total_blocks - len(self.allocator.free_blocks)
        block_bytes = (
            self.config.num_layers
            * self.config.block_size
            * self.config.num_heads
            * self.config.head_dim
            * 4  # float32
            * 2  # key + value
        )

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.allocator.free_blocks),
            "total_cache_mb": (total_blocks * block_bytes) / (1024 * 1024),
            "used_cache_mb": (used_blocks * block_bytes) / (1024 * 1024),
        }
