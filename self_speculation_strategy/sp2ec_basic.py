"""Block-level arms for sequential SP2EC-Basic arm construction."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence


def removable_block_indices(num_blocks: int) -> List[int]:
    if num_blocks < 6:
        raise ValueError("sp2ec_basic requires at least 6 transformer blocks")
    return list(range(2, num_blocks - 2))


def max_basic_skipped_blocks(
    num_blocks: int, max_skip_fraction: Optional[float] = None
) -> int:
    """Return the skip endpoint, preserving the historical default range.

    An explicit fraction is relative to ALL transformer blocks. The first and
    last two blocks remain protected, even when a fraction of 1 is requested.
    """
    removable_count = len(removable_block_indices(num_blocks))
    if max_skip_fraction is None:
        return removable_count // 2
    if not 0.0 <= max_skip_fraction <= 1.0:
        raise ValueError("max_skip_fraction must be in [0, 1]")
    return min(math.floor(max_skip_fraction * num_blocks), removable_count)


def max_basic_arms(
    num_blocks: int,
    max_skip_fraction: Optional[float] = None,
    include_zero_skip_arm: bool = False,
) -> int:
    return max_basic_skipped_blocks(num_blocks, max_skip_fraction) + int(
        include_zero_skip_arm
    )


def build_block_skip_set(num_blocks: int, removed_blocks: Sequence[int]) -> List[int]:
    removable = set(removable_block_indices(num_blocks))
    removed = {int(block_idx) for block_idx in removed_blocks}
    invalid = removed - removable
    if invalid:
        raise ValueError(f"protected or invalid blocks cannot be removed: {sorted(invalid)}")

    skip_set = [0] * (2 * num_blocks)
    for block_idx in removed:
        skip_set[2 * block_idx] = 1
        skip_set[2 * block_idx + 1] = 1
    return skip_set


@dataclass
class SP2ECBasicArm:
    budget: int
    skip_set: List[int]
    removed_blocks: List[int]
    removed_block_importance: float

    @property
    def estimated_tpt(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "budget": self.budget,
            "skip_set": list(self.skip_set),
            "removed_blocks": list(self.removed_blocks),
            "removed_block_importance": self.removed_block_importance,
            "skips": sum(self.skip_set),
            "attn_skips": sum(self.skip_set[::2]),
            "mlp_skips": sum(self.skip_set[1::2]),
        }
