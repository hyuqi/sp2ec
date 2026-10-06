"""Sequential block-importance arm construction with SP2EC UniUCB selection."""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from device_utils import model_input_device
from generation_limits import limit_generation_to_context
from llava_next import count_llava_next_visual_tokens
from multimodal import (
    get_text_model,
    is_supported_multimodal,
    prepare_multimodal_inputs,
)
from tree_verification import (
    DraftTree,
    confidence_topk,
    count_top1_spine_matches,
    evaluate_greedy_tree,
    tree_importance_boundary_token,
)
from utils import (
    Env,
    GenerationResult,
    apply_generation_constraints,
    crop_kv_cache,
    decode_next_token,
    forward,
    forward_draft_divided_multi,
    forward_draft_divided_multi_with_importance,
    forward_full_with_block_importance,
    forward_multimodal_prefill,
    forward_verify_divided_multi,
    forward_verify_tree_divided_multi,
    select_kv_cache_path,
)

from .sp2ec import SinglePeakArmSelector
from .sp2ec_basic import (
    SP2ECBasicArm,
    build_block_skip_set,
    max_basic_arms,
    max_basic_skipped_blocks,
    removable_block_indices,
)


@dataclass
class _BasicRoundResult:
    input_ids: torch.Tensor
    past_key_values: Any
    committed_tokens: List[int]
    matched_draft_tokens: int
    top1_spine_matches: int
    num_drafted: int
    draft_forward_count: int
    block_importance: Optional[torch.Tensor]
    tree_candidate_count: int
    empirical_cost: float
    live_verifier_reference_mismatch_count: int
    live_verifier_reference_first_mismatch_offset: Optional[int]


class SP2ECBasicGenerator:
    """Build nested block-removal arms online, then select them with UniUCB."""

    def __init__(
        self,
        env: Env,
        gamma: int = 4,
        optimize_interval: int = 512,
        coefficients: Optional[tuple] = None,
        tree: bool = False,
        beta: float = 0.1,
        draft_confidence_threshold: float = 0.7,
        dynamic_draft_stopping: bool = True,
        importance_protected_edge_blocks: int = 2,
        audit_reference_verifier_logits: bool = False,
        max_skip_fraction: Optional[float] = None,
        include_zero_skip_arm: bool = False,
        min_skipped_blocks: Optional[int] = None,
        max_skipped_blocks: Optional[int] = None,
    ):
        self.env = env
        self.model = env.model
        self.gamma = int(gamma)
        self.optimize_interval = int(optimize_interval)
        self.coefficients = coefficients
        self.tree = bool(tree)
        self.beta = float(beta)
        self.dynamic_draft_stopping = bool(dynamic_draft_stopping)
        self.audit_reference_verifier_logits = bool(
            audit_reference_verifier_logits
        )
        if not 0.0 <= draft_confidence_threshold <= 1.0:
            raise ValueError("draft_confidence_threshold must be in [0, 1]")
        self.threshold = float(draft_confidence_threshold)
        self.L = len(get_text_model(self.model).layers)
        self.importance_protected_edge_blocks = int(
            importance_protected_edge_blocks
        )
        if (
            self.importance_protected_edge_blocks < 0
            or 2 * self.importance_protected_edge_blocks >= self.L
        ):
            raise ValueError(
                "importance_protected_edge_blocks must be non-negative and "
                "leave at least one unprotected transformer block"
            )
        self.max_skip_fraction = (
            float(max_skip_fraction) if max_skip_fraction is not None else None
        )
        self.include_zero_skip_arm = bool(include_zero_skip_arm)
        self.explicit_skip_range = (
            min_skipped_blocks is not None or max_skipped_blocks is not None
        )
        self.min_skipped_blocks = min_skipped_blocks
        if self.explicit_skip_range:
            if min_skipped_blocks is None or max_skipped_blocks is None:
                raise ValueError(
                    "min_skipped_blocks and max_skipped_blocks must be supplied together"
                )
            if self.max_skip_fraction is not None or self.include_zero_skip_arm:
                raise ValueError(
                    "explicit skipped-block bounds cannot be combined with "
                    "max_skip_fraction or include_zero_skip_arm"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (min_skipped_blocks, max_skipped_blocks)
            ):
                raise ValueError("skipped-block bounds must be integers")
            removable_count = len(removable_block_indices(self.L))
            if not 0 <= min_skipped_blocks <= max_skipped_blocks <= removable_count:
                raise ValueError(
                    "skipped-block bounds must satisfy "
                    f"0 <= min <= max <= {removable_count}"
                )
            self.max_skipped_blocks = max_skipped_blocks
            self.num_basic_arms = max_skipped_blocks - min_skipped_blocks + 1
            # Nested removal still needs the preceding 1..min-1 rounds.
            initialization_rounds = max_skipped_blocks + int(min_skipped_blocks == 0)
        else:
            self.max_skipped_blocks = max_basic_skipped_blocks(
                self.L, self.max_skip_fraction
            )
            self.num_basic_arms = max_basic_arms(
                self.L, self.max_skip_fraction, self.include_zero_skip_arm
            )
            initialization_rounds = self.num_basic_arms
        if self.num_basic_arms < 1:
            raise ValueError(
                "sp2ec_basic requires at least one arm; include_zero_skip_arm "
                "must be enabled when max_skip_fraction permits no skipped blocks"
            )
        if self.gamma < 1:
            raise ValueError("gamma must be at least 1")
        if self.optimize_interval <= initialization_rounds:
            raise ValueError(
                "optimize_interval must exceed the number of sp2ec_basic initialization arms "
                f"and preparatory construction rounds ({initialization_rounds})"
            )

        self.cuda_devices = sorted(
            {parameter.device for parameter in self.model.parameters() if parameter.device.type == "cuda"},
            key=str,
        )
        self.arm_selector = self._new_arm_selector()
        self.completed_arm_sets: List[Dict[str, Any]] = []
        self.multimodal_prefill_inputs = None
        self.min_new_tokens = 0
        self.repetition_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self._constraint_output_ids: List[int] = []
        self._reset_statistics()

    def _new_arm_selector(self):
        return SinglePeakArmSelector(
            max_draft_length=self.gamma,
            beta=self.beta,
        )

    def _configure_generation_constraints(
        self,
        output_ids: List[int],
        *,
        min_new_tokens: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> None:
        self.min_new_tokens = int(min_new_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self._constraint_output_ids = output_ids

    def _constrain_draft_logits(
        self,
        logits: torch.Tensor,
        output_length: int,
        draft_output_ids: List[int],
        eos_token_ids: List[int],
        reference_output_ids: Optional[List[int]],
    ) -> torch.Tensor:
        if reference_output_ids is not None:
            committed_prefix = reference_output_ids[:output_length]
        else:
            committed_prefix = self._constraint_output_ids[:output_length]
        return apply_generation_constraints(
            logits,
            list(committed_prefix) + draft_output_ids,
            eos_token_ids=eos_token_ids,
            min_new_tokens=self.min_new_tokens,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )

    def _fixed_greedy_verification_tokens(
        self,
        verification_logits: torch.Tensor,
        output_length: int,
        draft_output_ids: List[int],
        eos_token_ids: List[int],
    ) -> torch.Tensor:
        """Decode a chain verifier without confidence/softmax overhead."""
        if verification_logits.dim() != 3:
            raise ValueError("verification logits must have shape [B, T, V]")
        committed_prefix = self._constraint_output_ids[:output_length]
        verified_ids: List[int] = []
        for offset in range(verification_logits.shape[1]):
            # Logit offset j is conditioned on the first j draft tokens. Once
            # a mismatch occurs later positions are discarded, as in standard
            # chain speculative decoding.
            constrained = apply_generation_constraints(
                verification_logits[:, offset, :],
                committed_prefix + draft_output_ids[:offset],
                eos_token_ids=eos_token_ids,
                min_new_tokens=self.min_new_tokens,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )
            verified_ids.append(int(constrained.argmax(dim=-1).item()))
        return torch.tensor(
            [verified_ids],
            dtype=torch.long,
            device=verification_logits.device,
        )

    def _reset_statistics(self) -> None:
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0
        self.total_optimization_time = 0.0
        self.total_accepted_length = 0
        self.total_drafted = 0
        self.total_matched = 0
        self.total_tokens = 0
        self.total_layers = 0
        self.step_count = 0
        self.runtime_arm_rounds = 0
        self.runtime_skip_sum = 0
        self.runtime_attn_skip_sum = 0
        self.runtime_mlp_skip_sum = 0
        self.total_tree_candidates = 0
        self.tree_rounds = 0
        self.last_arm_build_rounds = 0
        self.last_arm_build_preparatory_budgets: List[int] = []

    def _synchronize_cuda(self) -> None:
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)

    def _archive_current_arm_set(self) -> None:
        if self.arm_selector.has_arms:
            self.completed_arm_sets.append(self._current_arm_snapshot())

    def _current_arm_snapshot(self) -> Dict[str, Any]:
        snapshot = self.arm_selector.snapshot()
        if self.explicit_skip_range:
            snapshot.update(
                {
                    "active_skipped_block_range": [
                        self.min_skipped_blocks, self.max_skipped_blocks
                    ],
                    "construction_rounds": self.last_arm_build_rounds,
                    "preparatory_rounds": len(self.last_arm_build_preparatory_budgets),
                    "preparatory_budgets": list(self.last_arm_build_preparatory_budgets),
                }
            )
        return snapshot

    def _arm_history(self) -> List[Dict[str, Any]]:
        history = list(self.completed_arm_sets)
        if self.arm_selector.has_arms:
            history.append(self._current_arm_snapshot())
        return history

    def _record_round(self, skip_set: List[int], result: _BasicRoundResult) -> None:
        skip_count = sum(skip_set)
        self.runtime_arm_rounds += 1
        self.runtime_skip_sum += skip_count
        self.runtime_attn_skip_sum += sum(skip_set[::2])
        self.runtime_mlp_skip_sum += sum(skip_set[1::2])
        self.total_drafted += result.num_drafted
        self.total_matched += result.matched_draft_tokens
        self.total_tokens += len(result.committed_tokens)
        self.total_accepted_length += len(result.committed_tokens)
        self.total_layers += (
            (2 * self.L - skip_count) * result.draft_forward_count
            + 2 * self.L
        )
        self.step_count += 1
        if self.tree:
            self.total_tree_candidates += result.tree_candidate_count
            self.tree_rounds += 1

    @staticmethod
    def _trim_at_eos(output_ids: List[int], eos_token_ids: List[int]) -> bool:
        eos_positions = [output_ids.index(eos_id) for eos_id in eos_token_ids if eos_id in output_ids]
        if not eos_positions:
            return False
        del output_ids[min(eos_positions):]
        return True

    def _target_token(
        self,
        logits: torch.Tensor,
        output_length: int,
        reference_output_ids: Optional[List[int]],
        sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        if reference_output_ids is not None and output_length < len(reference_output_ids):
            return int(reference_output_ids[output_length])
        token, _ = decode_next_token(
            logits,
            token_idx=-1,
            sample=sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return int(token.item() if isinstance(token, torch.Tensor) else token)

    def _full_model_warmup(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        output_ids: List[int],
        max_new_tokens: int,
        eos_token_ids: List[int],
        reference_output_ids: Optional[List[int]],
        sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, Any, List[int], Optional[torch.Tensor]]:
        warmup_tokens: List[int] = []
        importance_rows: List[torch.Tensor] = []
        self._synchronize_cuda()
        warmup_start = time.perf_counter()

        for warmup_idx in range(5):
            if len(output_ids) + len(warmup_tokens) >= max_new_tokens:
                break
            block_importance = None
            if self.multimodal_prefill_inputs is not None:
                logits, past_key_values = forward_multimodal_prefill(
                    self.model,
                    self.multimodal_prefill_inputs,
                )
                self.multimodal_prefill_inputs = None
            elif past_key_values is None:
                logits, past_key_values = forward(self.model, input_ids, past_key_values)
            else:
                logits, past_key_values, block_importance = forward_full_with_block_importance(
                    self.model,
                    input_ids,
                    past_key_values,
                    protected_edge_blocks=self.importance_protected_edge_blocks,
                )

            if block_importance is not None and 1 <= warmup_idx <= 3:
                importance_rows.append(block_importance)

            next_id = self._target_token(
                logits,
                len(output_ids) + len(warmup_tokens),
                reference_output_ids,
                sample,
                temperature,
                top_k,
                top_p,
            )
            warmup_tokens.append(next_id)
            input_ids = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
            if next_id in eos_token_ids:
                break

        self._synchronize_cuda()
        self.total_verify_time += time.perf_counter() - warmup_start
        importance = (
            torch.stack(importance_rows).mean(dim=0)
            if len(warmup_tokens) == 5 and len(importance_rows) == 3
            else None
        )
        return input_ids, past_key_values, warmup_tokens, importance

    def _speculative_round(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        output_length: int,
        num_speculations: int,
        skip_set: List[int],
        eos_token_ids: List[int],
        reference_output_ids: Optional[List[int]],
        sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        collect_importance: bool,
    ) -> _BasicRoundResult:
        draft_input_ids = input_ids.clone()
        draft_cache = past_key_values
        draft_output_ids: List[int] = []
        draft_importance_rows: List[torch.Tensor] = []
        draft_tree = DraftTree.with_root(int(input_ids[0, -1].item())) if self.tree else None

        self._synchronize_cuda()
        draft_start = time.perf_counter()
        for draft_idx in range(num_speculations):
            if collect_importance:
                next_logits, draft_cache, current_importance = (
                    forward_draft_divided_multi_with_importance(
                        self.model,
                        draft_input_ids,
                        skip_set,
                        draft_cache,
                        protected_edge_blocks=self.importance_protected_edge_blocks,
                    )
                )
                if draft_idx > 0:
                    draft_importance_rows.append(current_importance)
            else:
                next_logits, draft_cache = forward_draft_divided_multi(
                    self.model,
                    draft_input_ids,
                    skip_set,
                    draft_cache,
                )

            next_logits = self._constrain_draft_logits(
                next_logits,
                output_length,
                draft_output_ids,
                eos_token_ids,
                reference_output_ids,
            )

            fixed_greedy_chain = (
                not self.dynamic_draft_stopping
                and not sample
                and draft_tree is None
            )
            if fixed_greedy_chain:
                next_token = next_logits[:, -1, :].argmax(dim=-1)
                next_prob = None
            else:
                next_token, next_prob = decode_next_token(
                    next_logits,
                    token_idx=-1,
                    sample=sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            next_id = int(next_token.item() if isinstance(next_token, torch.Tensor) else next_token)
            confidence = (
                float(next_prob[0, next_id].item())
                if next_prob is not None
                else 1.0
            )
            draft_output_ids.append(next_id)

            if draft_tree is not None:
                candidate_count = min(confidence_topk(confidence), next_prob.shape[-1])
                candidate_ids = torch.topk(next_prob[0], candidate_count).indices.tolist()
                draft_tree.add_level(candidate_ids)

            draft_input_ids = torch.tensor(
                [[next_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if next_id in eos_token_ids or (
                self.dynamic_draft_stopping and confidence < self.threshold
            ):
                break

        self._synchronize_cuda()
        draft_elapsed = time.perf_counter() - draft_start
        self.total_draft_time += draft_elapsed

        verification_prefix_length = past_key_values[0][0].shape[2]
        selected_tree_indices = None
        tree_candidate_count = 0
        live_verifier_reference_mismatch_count = 0
        live_verifier_reference_first_mismatch_offset = None
        self._synchronize_cuda()
        verify_start = time.perf_counter()

        if draft_tree is not None:
            tree_candidate_count = draft_tree.candidate_count
            tree_input_ids = torch.tensor(
                [draft_tree.token_ids],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            verification_logits, verified_cache = forward_verify_tree_divided_multi(
                self.model,
                tree_input_ids,
                draft_tree.parent_indices,
                draft_tree.depths,
                past_key_values,
            )
            target_predictions = verification_logits[0].argmax(dim=-1).tolist()
            reference_suffix = None
            if reference_output_ids is not None:
                reference_suffix = reference_output_ids[output_length:]
            committed_tokens, matched_draft_tokens, selected_tree_indices = evaluate_greedy_tree(
                draft_tree,
                target_predictions,
                reference_token_ids=reference_suffix,
            )
            top1_matches = count_top1_spine_matches(
                draft_tree,
                target_predictions,
                reference_token_ids=reference_suffix,
            )
            # The verifier returns a fresh DynamicCache result; the original
            # target and draft caches remain available for importance probes.
            past_key_values = select_kv_cache_path(
                verified_cache,
                verification_prefix_length,
                selected_tree_indices,
                allow_in_place=True,
            )
        else:
            draft_tensor = torch.tensor(
                [draft_output_ids],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            verify_input = torch.cat([input_ids, draft_tensor], dim=-1)
            verification_logits, verified_cache = forward_verify_divided_multi(
                self.model,
                verify_input,
                past_key_values,
            )
            if reference_output_ids is not None:
                verified_list = reference_output_ids[
                    output_length: output_length + len(draft_output_ids) + 1
                ]
                verified_tokens = torch.tensor(
                    [verified_list],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                live_verified_tokens = None
                if self.audit_reference_verifier_logits and not sample:
                    # Canonical-reference profiling still performs the same
                    # target-logit constraints and argmax work as native
                    # verification. Decisions below use the serial reference
                    # so every arm follows one identical token trajectory.
                    live_verified_tokens = (
                        self._fixed_greedy_verification_tokens(
                            verification_logits,
                            output_length,
                            draft_output_ids,
                            eos_token_ids,
                        ).to(input_ids.device)
                    )
            else:
                live_verified_tokens = None
                if not self.dynamic_draft_stopping and not sample:
                    verified_tokens = self._fixed_greedy_verification_tokens(
                        verification_logits,
                        output_length,
                        draft_output_ids,
                        eos_token_ids,
                    )
                else:
                    verified_tokens, _ = decode_next_token(
                        verification_logits,
                        sample=sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

            # With a sharded model, the LM head can return logits on a
            # different GPU from the input embedding.  Keep the comparison
            # and subsequent token handling on the input/cache device for
            # every verifier decoding path, including fixed greedy decoding.
            verified_tokens = verified_tokens.to(input_ids.device)

            draft_1d = draft_tensor
            comparison_length = min(draft_1d.shape[1], max(verified_tokens.shape[1] - 1, 0))
            verified = draft_1d[:, :comparison_length] == verified_tokens[:, :comparison_length]
            if comparison_length:
                matched_draft_tokens = int(((~verified).cumsum(dim=-1) < 1).sum().item())
            else:
                matched_draft_tokens = 0
            top1_matches = matched_draft_tokens
            committed_tokens = draft_output_ids[:matched_draft_tokens]
            if matched_draft_tokens < verified_tokens.shape[1]:
                committed_tokens.append(int(verified_tokens[0, matched_draft_tokens].item()))
            if live_verified_tokens is not None:
                # Only positions through the correction/bonus token are
                # conditioned on the canonical accepted prefix. Later logits
                # may be conditioned on an already-rejected draft token.
                relevant_length = min(
                    matched_draft_tokens + 1,
                    live_verified_tokens.shape[1],
                    verified_tokens.shape[1],
                )
                mismatch_offsets = torch.nonzero(
                    live_verified_tokens[:, :relevant_length]
                    != verified_tokens[:, :relevant_length],
                    as_tuple=False,
                )
                live_verifier_reference_mismatch_count = int(
                    mismatch_offsets.shape[0]
                )
                if live_verifier_reference_mismatch_count:
                    live_verifier_reference_first_mismatch_offset = int(
                        mismatch_offsets[0, 1].item()
                    )
            past_key_values = crop_kv_cache(
                verified_cache,
                verification_prefix_length + 1 + matched_draft_tokens,
            )

        self._synchronize_cuda()
        verify_elapsed = time.perf_counter() - verify_start
        self.total_verify_time += verify_elapsed

        block_importance = None
        if collect_importance and draft_output_ids:
            importance_samples = list(draft_importance_rows[:top1_matches])
            if top1_matches < len(draft_output_ids):
                if draft_tree is not None:
                    boundary_token = tree_importance_boundary_token(
                        draft_tree,
                        committed_tokens,
                        matched_draft_tokens,
                        top1_matches,
                    )
                else:
                    boundary_token = draft_output_ids[top1_matches]
                boundary_cache = crop_kv_cache(
                    draft_cache,
                    verification_prefix_length + 1 + top1_matches,
                )
                boundary_input = torch.tensor(
                    [[boundary_token]],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            else:
                boundary_input = draft_input_ids
                boundary_cache = draft_cache

            self._synchronize_cuda()
            importance_probe_start = time.perf_counter()
            _, _, boundary_importance = forward_draft_divided_multi_with_importance(
                self.model,
                boundary_input,
                skip_set,
                boundary_cache,
                protected_edge_blocks=self.importance_protected_edge_blocks,
            )
            self._synchronize_cuda()
            self.total_optimization_time += time.perf_counter() - importance_probe_start
            importance_samples.append(boundary_importance)
            block_importance = torch.stack(importance_samples).mean(dim=0)

        next_input_id = committed_tokens[-1] if committed_tokens else int(input_ids[0, -1].item())
        next_input = torch.tensor(
            [[next_input_id]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return _BasicRoundResult(
            input_ids=next_input,
            past_key_values=past_key_values,
            committed_tokens=committed_tokens,
            matched_draft_tokens=matched_draft_tokens,
            top1_spine_matches=top1_matches,
            num_drafted=len(draft_output_ids),
            draft_forward_count=len(draft_output_ids),
            block_importance=block_importance,
            tree_candidate_count=tree_candidate_count,
            empirical_cost=draft_elapsed + verify_elapsed,
            live_verifier_reference_mismatch_count=(
                live_verifier_reference_mismatch_count
            ),
            live_verifier_reference_first_mismatch_offset=(
                live_verifier_reference_first_mismatch_offset
            ),
        )

    def _build_arm_set(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        output_ids: List[int],
        initial_importance: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: List[int],
        reference_output_ids: Optional[List[int]],
        sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        arms: List[SP2ECBasicArm] = []
        observations: List[Tuple[int, float]] = []
        removed_blocks: List[int] = []
        importance = initial_importance
        hit_eos = False
        self.last_arm_build_rounds = 0
        self.last_arm_build_preparatory_budgets = []

        first_budget = 0 if self.include_zero_skip_arm else 1
        last_budget = first_budget + self.num_basic_arms - 1
        if self.explicit_skip_range:
            first_budget = 0 if self.min_skipped_blocks == 0 else 1
            last_budget = self.max_skipped_blocks
        for arm_number in range(first_budget, last_budget + 1):
            if len(output_ids) >= max_new_tokens:
                break
            removed_importance = 0.0
            if arm_number > 0:
                eligible = [
                    block_idx
                    for block_idx in removable_block_indices(self.L)
                    if block_idx not in removed_blocks
                ]
                eligible_tensor = torch.tensor(
                    eligible,
                    dtype=torch.long,
                    device=importance.device,
                )
                local_idx = int(importance.index_select(0, eligible_tensor).argmin().item())
                block_to_remove = eligible[local_idx]
                removed_importance = float(importance[block_to_remove].item())
                removed_blocks.append(block_to_remove)
            skip_set = build_block_skip_set(self.L, removed_blocks)
            arm = SP2ECBasicArm(
                budget=arm_number,
                skip_set=skip_set,
                removed_blocks=list(removed_blocks),
                removed_block_importance=removed_importance,
            )

            num_speculations = min(
                self.gamma,
                max_new_tokens - len(output_ids) - 1,
            )
            result = self._speculative_round(
                input_ids,
                past_key_values,
                len(output_ids),
                num_speculations,
                skip_set,
                eos_token_ids,
                reference_output_ids,
                sample,
                temperature,
                top_k,
                top_p,
                collect_importance=arm_number > 0,
            )

            is_active_arm = (
                not self.explicit_skip_range or arm_number >= self.min_skipped_blocks
            )
            if is_active_arm:
                arms.append(arm)
                observations.append((len(result.committed_tokens), result.empirical_cost))
            else:
                # These sequential removals prepare the first retained arm.
                # They generate real tokens and incur normal measured costs,
                # but are never selectable arms or bandit observations.
                self.last_arm_build_preparatory_budgets.append(arm_number)
            self.last_arm_build_rounds += 1
            self._record_round(skip_set, result)
            input_ids = result.input_ids
            past_key_values = result.past_key_values
            output_ids.extend(result.committed_tokens)

            if self._trim_at_eos(output_ids, eos_token_ids):
                hit_eos = True
                break
            if arm_number == 0:
                # This is a full-model DRAFTER plus verification, not the AR
                # baseline. Keep warmup importance for the first removal and
                # avoid the redundant importance probe for this extra arm.
                continue
            if result.block_importance is None:
                break
            importance = result.block_importance

        return input_ids, past_key_values, arms, observations, hit_eos

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample: bool = False,
        reference_output_ids: Optional[List[int]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        video_num_frames: int = 64,
        video_max_visual_tokens: int = 8192,
        llava_video_contact_sheets: int = 1,
    ) -> GenerationResult:
        if self.tree and sample:
            raise ValueError("sp2ec_basic tree verification currently supports greedy decoding only")

        self.multimodal_prefill_inputs = None
        if is_supported_multimodal(self.model):
            enc = prepare_multimodal_inputs(
                self.model,
                self.env.processor,
                prompt,
                image=image,
                video=video,
                video_num_frames=video_num_frames,
                video_max_visual_tokens=video_max_visual_tokens,
                llava_video_contact_sheets=llava_video_contact_sheets,
            )
            self.multimodal_prefill_inputs = enc
        else:
            enc = self.env.tok(prompt, return_tensors="pt")

        input_ids_list = enc["input_ids"][0].tolist()
        input_ids = enc["input_ids"].to(model_input_device(self.model))
        if self.multimodal_prefill_inputs is not None:
            del enc

        eos_token_ids = [self.env.eos_id] if self.env.eos_id is not None else []
        generation_limit = (
            min(max_new_tokens, len(reference_output_ids))
            if reference_output_ids is not None
            else max_new_tokens
        )
        generation_limit, min_new_tokens = limit_generation_to_context(
            self.model,
            len(input_ids_list),
            generation_limit,
            min_new_tokens,
            label="SP2EC-Basic",
        )
        output_ids: List[int] = []
        self._configure_generation_constraints(
            output_ids,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        past_key_values = None
        self.arm_selector = self._new_arm_selector()
        self.completed_arm_sets = []
        self._reset_statistics()
        rounds_since_rebuild = self.optimize_interval

        self._synchronize_cuda()
        total_start = time.perf_counter()
        stop = False
        while len(output_ids) < generation_limit and not stop:
            if reference_output_ids is not None and len(output_ids) >= len(reference_output_ids):
                break

            if not self.arm_selector.has_arms or rounds_since_rebuild >= self.optimize_interval:
                self._archive_current_arm_set()
                self.arm_selector = self._new_arm_selector()
                input_ids, past_key_values, warmup_tokens, importance = self._full_model_warmup(
                    input_ids,
                    past_key_values,
                    output_ids,
                    generation_limit,
                    eos_token_ids,
                    reference_output_ids,
                    sample,
                    temperature,
                    top_k,
                    top_p,
                )
                output_ids.extend(warmup_tokens)
                if self._trim_at_eos(output_ids, eos_token_ids):
                    break
                if importance is None or len(output_ids) >= generation_limit:
                    break

                input_ids, past_key_values, arms, observations, hit_eos = self._build_arm_set(
                    input_ids,
                    past_key_values,
                    output_ids,
                    importance,
                    generation_limit,
                    eos_token_ids,
                    reference_output_ids,
                    sample,
                    temperature,
                    top_k,
                    top_p,
                )
                if not arms:
                    break
                self.arm_selector.reset(arms)
                for arm_idx, (reward, cost) in enumerate(observations):
                    self.arm_selector.update(arm_idx, reward, cost)
                rounds_since_rebuild = self.last_arm_build_rounds
                if hit_eos or len(output_ids) >= generation_limit:
                    break

            arm_idx = self.arm_selector.select_arm()
            arm = self.arm_selector.arms[arm_idx]
            num_speculations = min(
                self.gamma,
                generation_limit - len(output_ids) - 1,
            )
            result = self._speculative_round(
                input_ids,
                past_key_values,
                len(output_ids),
                num_speculations,
                arm.skip_set,
                eos_token_ids,
                reference_output_ids,
                sample,
                temperature,
                top_k,
                top_p,
                collect_importance=False,
            )
            self.arm_selector.update(
                arm_idx,
                len(result.committed_tokens),
                result.empirical_cost,
            )
            self._record_round(arm.skip_set, result)
            input_ids = result.input_ids
            past_key_values = result.past_key_values
            output_ids.extend(result.committed_tokens)
            rounds_since_rebuild += 1
            stop = self._trim_at_eos(output_ids, eos_token_ids)
            if not result.committed_tokens:
                break

        self._synchronize_cuda()
        total_time = time.perf_counter() - total_start
        acceptance_rate = (
            self.total_matched / self.total_drafted
            if self.total_drafted > 0
            else None
        )
        tokens_per_layer = (
            self.total_tokens / self.total_layers
            if self.total_layers > 0
            else None
        )
        text = self.env.tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""
        return GenerationResult(
            text=text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            num_input_tokens=len(input_ids_list),
            num_visual_tokens=count_llava_next_visual_tokens(self.model, input_ids_list),
            acceptance_rate=acceptance_rate,
            tokens_per_layer=tokens_per_layer,
            draft_time=self.total_draft_time,
            verify_time=self.total_verify_time,
            optimization_time=self.total_optimization_time,
            total_time=total_time,
            total_accepted_length=self.total_accepted_length,
            total_steps=self.step_count,
            arm_set_history=self._arm_history(),
            tree_candidate_tokens=self.total_tree_candidates if self.tree else None,
            avg_tree_candidates=(
                self.total_tree_candidates / self.tree_rounds
                if self.tree and self.tree_rounds > 0
                else None
            ),
        )
