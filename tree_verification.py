"""Sparse confidence-aware token-tree helpers for speculative decoding."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def confidence_topk(confidence: float) -> int:
    """Map top-1 draft confidence to the requested tree width."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if confidence <= 0.5:
        return 5
    if confidence <= 0.8:
        return 3
    if confidence <= 0.95:
        return 2
    return 2


@dataclass
class DraftTree:
    """Linearized sparse tree with a top-1 spine and ranked alternatives."""

    token_ids: List[int]
    parent_indices: List[int]
    depths: List[int]
    main_node_indices: List[int]

    @classmethod
    def with_root(cls, root_token_id: int) -> "DraftTree":
        return cls(
            token_ids=[int(root_token_id)],
            parent_indices=[-1],
            depths=[0],
            main_node_indices=[0],
        )

    def add_level(self, candidate_token_ids: Sequence[int]) -> int:
        """Attach ranked candidates to the current top-1 spine node."""
        if not candidate_token_ids:
            raise ValueError("a tree level must contain at least one candidate")

        parent_idx = self.main_node_indices[-1]
        depth = self.depths[parent_idx] + 1
        first_idx = len(self.token_ids)
        for token_id in candidate_token_ids:
            self.token_ids.append(int(token_id))
            self.parent_indices.append(parent_idx)
            self.depths.append(depth)
        self.main_node_indices.append(first_idx)
        return first_idx

    def children(self, parent_idx: int) -> List[int]:
        return [
            idx
            for idx, candidate_parent in enumerate(self.parent_indices)
            if candidate_parent == parent_idx
        ]

    @property
    def draft_depth(self) -> int:
        return len(self.main_node_indices) - 1

    @property
    def candidate_count(self) -> int:
        return len(self.token_ids) - 1


def evaluate_greedy_tree(
    tree: DraftTree,
    predicted_token_ids: Sequence[int],
    reference_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[int], int, List[int]]:
    """Select the longest target-consistent branch and its correction token.

    ``predicted_token_ids[i]`` is the target model's greedy prediction after
    tree node ``i``. ``reference_token_ids`` optionally replaces those target
    predictions, as benchmark comparison mode already does for linear SD.
    """
    if len(predicted_token_ids) != len(tree.token_ids):
        raise ValueError("predicted_token_ids must contain one prediction per tree node")

    accepted_tokens: List[int] = []
    selected_node_indices = [0]
    parent_idx = 0
    number_of_matches = 0

    while True:
        if reference_token_ids is not None and number_of_matches < len(reference_token_ids):
            target_token = int(reference_token_ids[number_of_matches])
        else:
            target_token = int(predicted_token_ids[parent_idx])

        matching_child = None
        for child_idx in tree.children(parent_idx):
            if tree.token_ids[child_idx] == target_token:
                matching_child = child_idx
                break

        if matching_child is None:
            accepted_tokens.append(target_token)
            break

        accepted_tokens.append(target_token)
        selected_node_indices.append(matching_child)
        number_of_matches += 1
        parent_idx = matching_child

    return accepted_tokens, number_of_matches, selected_node_indices


def count_top1_spine_matches(
    tree: DraftTree,
    predicted_token_ids: Sequence[int],
    reference_token_ids: Optional[Sequence[int]] = None,
) -> int:
    """Count consecutive target matches along the autoregressive top-1 spine."""
    if len(predicted_token_ids) != len(tree.token_ids):
        raise ValueError("predicted_token_ids must contain one prediction per tree node")

    matches = 0
    for depth in range(tree.draft_depth):
        parent_idx = tree.main_node_indices[depth]
        top1_child_idx = tree.main_node_indices[depth + 1]
        if reference_token_ids is not None and depth < len(reference_token_ids):
            target_token = int(reference_token_ids[depth])
        else:
            target_token = int(predicted_token_ids[parent_idx])
        if tree.token_ids[top1_child_idx] != target_token:
            break
        matches += 1
    return matches


def tree_importance_boundary_token(
    tree: DraftTree,
    committed_tokens: Sequence[int],
    matched_tree_tokens: int,
    top1_spine_matches: int,
) -> Optional[int]:
    """Return the non-top-1 stopping token, or the rejected top-1 fallback."""
    if top1_spine_matches >= tree.draft_depth:
        return None
    if matched_tree_tokens > top1_spine_matches:
        return int(committed_tokens[top1_spine_matches])
    return int(tree.token_ids[tree.main_node_indices[top1_spine_matches + 1]])
