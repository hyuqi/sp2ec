# knapspec.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from utils import crop_kv_cache, _make_branch_parallel_causal_mask
import transformers
import time
from .sp2ec import select_top_tpt_arms
from device_utils import model_uses_multiple_cuda_devices, module_device
from multimodal import (
    get_qwen_vl_rope_deltas,
    get_text_model,
    is_qwen_vl_with_mrope,
)


@dataclass
class KnapspecArm:
    skip_set: List[int]
    budget: int
    alpha: float
    estimated_tpt: float
    estimated_draft_length: int
    tpt_rank: int = 0
    full_weighted_budget: Optional[int] = None
    dp_budget_limit: Optional[int] = None
    dp_budget_fraction: Optional[float] = None
    attn_weight: Optional[int] = None
    mlp_weight: Optional[int] = None
    original_arm_index: Optional[int] = None
    source_pool_size: Optional[int] = None
    scoring_metadata: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "skip_set": list(self.skip_set),
            "budget": self.budget,
            "alpha": self.alpha,
            "estimated_tpt": self.estimated_tpt,
            "estimated_draft_length": self.estimated_draft_length,
            "tpt_rank": self.tpt_rank,
            "skips": sum(self.skip_set),
            "attn_skips": sum(self.skip_set[::2]),
            "mlp_skips": sum(self.skip_set[1::2]),
            "full_weighted_budget": self.full_weighted_budget,
            "dp_budget_limit": self.dp_budget_limit,
            "dp_budget_fraction": self.dp_budget_fraction,
            "budget_fraction_of_full": (
                self.budget / self.full_weighted_budget
                if self.full_weighted_budget
                else None
            ),
            "attn_weight": self.attn_weight,
            "mlp_weight": self.mlp_weight,
            "original_arm_index": self.original_arm_index,
            "source_pool_size": self.source_pool_size,
            "scoring_metadata": self.scoring_metadata,
        }


def validate_knapspec_arm_index_range(
    arm_index_start: Optional[int], arm_index_end: Optional[int]
) -> None:
    """Validate an optional inclusive range in the original sorted shortlist."""
    if arm_index_start is None and arm_index_end is None:
        return
    if arm_index_start is None or arm_index_end is None:
        raise ValueError("arm_index_start and arm_index_end must be supplied together")
    if type(arm_index_start) is not int or type(arm_index_end) is not int:
        raise ValueError("KnapSpec arm indices must be integers")
    if not 0 <= arm_index_start <= arm_index_end:
        raise ValueError("KnapSpec arm indices must satisfy 0 <= start <= end")


def select_knapspec_arm_subset(
    arms: List[KnapspecArm],
    arm_index_start: Optional[int] = None,
    arm_index_end: Optional[int] = None,
) -> List[KnapspecArm]:
    """Slice the existing budget-ordered shortlist, without re-ranking candidates.

    Indices refer to the pre-filter pool (including its zero arm when enabled),
    not to DP budgets. A requested fixed-size experiment must not silently run
    fewer arms if similarity pruning leaves a smaller pool at a later epoch.
    """
    validate_knapspec_arm_index_range(arm_index_start, arm_index_end)
    if arm_index_start is None:
        return arms
    ordered = sorted(arms, key=lambda arm: arm.budget)
    if arm_index_end >= len(ordered):
        raise RuntimeError(
            f"Requested KnapSpec source arm indices {arm_index_start}..{arm_index_end}, "
            f"but DP produced only {len(ordered)} candidates. "
            "The requested fixed-size arm subset is unavailable; no fallback "
            "or padded arms will be used."
        )
    selected = ordered[arm_index_start:arm_index_end + 1]
    for original_index, arm in enumerate(selected, start=arm_index_start):
        arm.original_arm_index = original_index
        arm.source_pool_size = len(ordered)
    return selected


def select_knapspec_arms(
    arms: List[KnapspecArm],
    num_arms: int,
    zero_skip_arm: Optional[KnapspecArm] = None,
) -> List[KnapspecArm]:
    """Optionally reserve a no-skip arm without changing the other arm policy.

    One-arm (plain KnapSpec) runs still choose the predicted best candidate;
    zero is allowed to compete but is not forced. Multi-arm runs reserve one
    of the requested slots for zero and fill the rest by predicted throughput.
    """
    if zero_skip_arm is None:
        return select_top_tpt_arms(arms, num_arms)
    nonzero_arms = [arm for arm in arms if any(arm.skip_set)]
    if num_arms == 1:
        return select_top_tpt_arms(nonzero_arms + [zero_skip_arm], num_arms)
    if num_arms < 1:
        raise ValueError("num_arms must be at least 1")
    selected = select_top_tpt_arms(nonzero_arms, num_arms - 1) + [zero_skip_arm]
    for rank, arm in enumerate(
        sorted(selected, key=lambda arm: arm.estimated_tpt, reverse=True), start=1
    ):
        arm.tpt_rank = rank
    return sorted(selected, key=lambda arm: arm.budget)


class Knapspec:
    """
    Dynamic programming to select skip-set candidates by latency budget.
    - L: number of decoder layers
    - optimize() ranks DP candidates with the original KnapSpec TPT and returns arms
    - skip-set entries use 1 = skip and 0 = keep
    """

    def __init__(
        self,
        L: int,
        M: int,
        model,
        device: Optional[torch.device] = None,
        coefficients: Optional[tuple] = None,
        sim_threshold: float = 0.5,
        dp_budget_fraction: float = 0.5,
        include_zero_skip_arm: bool = False,
        scoring_mode: str = "legacy",
    ):
        self.L = int(L)
        self.M = int(M)
        self.model = model
        self.text_model = get_text_model(model)
        self.layers = self.text_model.layers
        self.device = device or next(self.text_model.parameters()).device
        self.multi_gpu = model_uses_multiple_cuda_devices(model)
        self.is_prefill_stage = True
        self.cached_hidden_states = [[] for i in range(2 * self.L + 1)]
        self.skip_set: List[int] = [0] * (2 * self.L)
        self.total_skip = 0

        # Stats tracking
        self.sum_best_tpt = 0.0
        self.sum_skip = 0
        self.sum_attn_skip = 0
        self.sum_mlp_skip = 0
        self.optimize_count = 0
        self.arm_candidates: List[KnapspecArm] = []
        self.scoring_history = []
        self.last_budget = 0
        self.last_attn_weight = 1
        self.last_mlp_weight = 1
        
        self.c_1, self.c_2, self.c_3 = coefficients
        self.sim_threshold = sim_threshold
        if not 0.0 < dp_budget_fraction <= 1.0:
            raise ValueError("dp_budget_fraction must be in (0, 1]")
        self.dp_budget_fraction = float(dp_budget_fraction)
        self.include_zero_skip_arm = bool(include_zero_skip_arm)
        if scoring_mode != "legacy":
            raise ValueError("This sample implementation uses only original KnapSpec scoring.")
        self.scoring_mode = "legacy"

    def _score_candidate(self, alpha, skip_set, t_attn, t_mlp):
        draft_length, tpt = self.best_tpt_for_candidate(alpha, skip_set, t_attn, t_mlp)
        return draft_length, tpt, {
            "mode": "legacy", "length_search": "0..17", "tpt_units": "tokens/ms"
        }

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        # x = x.float()
        return F.normalize(x, p=2, dim=-1)

    def clear_cached_hidden_states(self):
        """Reset cached hidden states for all layers (used at the start of each SD round)."""
        self.cached_hidden_states = [[] for i in range(2 * self.L + 1)]

    def TPT(self, l, d, alpha, skip_set, t_attn, t_mlp):
        L = len(skip_set) // 2
        a = alpha[l]
        generated_tokens = (d + 1) if a == 1.0 else (a**(d + 1) - 1.0) / (a - 1.0)

        n_attn = sum(1 - skip_set[2*i + 0] for i in range(L))
        n_mlp  = sum(1 - skip_set[2*i + 1] for i in range(L))

        loaded_time = t_attn * (n_attn * d + L) + t_mlp * (n_mlp * d + L)
        return generated_tokens / loaded_time if loaded_time > 0 else 0.0

    def optimize_tpt(self, alpha, skip_sets, t_attn, t_mlp,d_max=18):
        max_skip = len(skip_sets) - 1
        max_tpt = -1
        best_l, best_d = -1, -1
        for l in range(1, max_skip + 1):
            for d in range(d_max):
                tpt = self.TPT(l, d, alpha, skip_sets[l], t_attn, t_mlp)
                if tpt > max_tpt:
                    max_tpt = tpt
                    best_l, best_d = l, d
        return best_l, best_d, max_tpt

    def best_tpt_for_candidate(self, alpha, skip_set, t_attn, t_mlp, d_max=18):
        """Rank each bandit candidate with the unchanged original TPT objective."""
        best_d = 0
        best_tpt = -1.0
        for d in range(d_max):
            tpt = self.TPT(0, d, [alpha], skip_set, t_attn, t_mlp)
            if tpt > best_tpt:
                best_d = d
                best_tpt = tpt
        return best_d, best_tpt

    def _apply_layer_single(self, idx: int, x: torch.Tensor, past_key_values=None, branch_len=None, num_branches=None) -> torch.Tensor:
        layer_idx = idx // 2  # Which layer (0, 1, 2, ...)
        is_attention = (idx % 2 == 0)  # Even = attention, Odd = MLP
            
        decoder_layer = self.layers[layer_idx]
        device = module_device(decoder_layer, self.device) if self.multi_gpu else self.device
        
        x = x.to(device)
        batch_size, seq_length = x.shape[0], x.shape[1]
        hidden_states = x
        
        if is_attention:
            # --- Apply Attention Block ---
            # Handle past key values for attention
            seq_length_with_past = seq_length
            past_key_values_length = 0
            
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
                seq_length_with_past = seq_length + past_key_values_length
        
            # Convert to DynamicCache
            past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

            if branch_len is not None and num_branches is not None:
                pos_one = torch.arange(
                    past_key_values_length,
                    past_key_values_length + branch_len,
                    dtype=torch.long,
                    device=device,
                )
                pos_all = pos_one.repeat(num_branches)
                position_ids = pos_all.unsqueeze(0).expand(batch_size, -1)
                
            else:
                seq_length_with_past = seq_length + past_key_values_length
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length_with_past,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1)

            attention_mask = _make_branch_parallel_causal_mask(
                input_ids_shape=(batch_size, seq_length),
                dtype=hidden_states.dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                branch_len=branch_len,
                num_branches=num_branches,
            )
            if is_qwen_vl_with_mrope(self.model):
                rope_deltas = get_qwen_vl_rope_deltas(self.model)
                if rope_deltas is None:
                    raise RuntimeError(
                        "Qwen-VL native prefill must run before KnapSpec optimization."
                    )
                position_ids = position_ids + rope_deltas.to(device=device, dtype=torch.long).reshape(-1, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
            # Apply input layer norm
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            
            # Self-attention forward pass
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                output_attentions=False,
                use_cache=True,
            )
            
            # Residual connection
            hidden_states = hidden_states + attn_output
            
        else:
            # --- Apply MLP Block ---
            # Apply post-attention layer norm
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            
            # MLP forward pass
            mlp_output = decoder_layer.mlp(normed_hidden)
            
            # Residual connection
            hidden_states = hidden_states + mlp_output
        
        hidden_states = hidden_states.squeeze(1)
        return hidden_states.to(self.device) if self.multi_gpu else hidden_states

    @torch.inference_mode()
    def optimize(self, past_key_values=None, num_arms: int = 1) -> List[KnapspecArm]:
        if num_arms < 1:
            raise ValueError("num_arms must be at least 1")
        optimize_start_time = time.perf_counter()
        
        past_key_values_length = past_key_values[0][0].shape[2]
        t_mlp = self.c_1
        t_attn = self.c_2 * past_key_values_length + self.c_3
        
        if t_attn > t_mlp:
            w_attn, w_mlp = min(int(round(t_attn / t_mlp)), 5), int(1.0)
        else:
            w_attn, w_mlp = int(1.0), min(int(round(t_mlp / t_attn)), 5)
        
        max_time_saved = (w_attn + w_mlp) * self.L
        budget = int(max_time_saved * self.dp_budget_fraction)
        self.last_budget = budget
        self.last_attn_weight = w_attn
        self.last_mlp_weight = w_mlp

        xs: List[torch.Tensor] = []
        if self.cached_hidden_states:
            for layer_idx in range(2 * self.L + 1):
                if self.cached_hidden_states[layer_idx]:
                    xs.append(torch.cat(self.cached_hidden_states[layer_idx], dim=1).to(self.device))

        opt_num = xs[0].shape[1]
        optimization_cache = crop_kv_cache(past_key_values, past_key_values_length - opt_num)

        g: List[List[Optional[torch.Tensor]]] = [[None] * (budget + 1) for _ in range(2 * self.L + 1)]
        parent: List[List[Optional[tuple]]] = [[None] * (budget + 1) for _ in range(2 * self.L + 1)]
        best_sims = torch.full((2 * self.L + 1, budget + 1), -float('inf'), device=self.device)
        
        sim_threshold = self.sim_threshold
        
        # --- Optimization: Start from block 2 (xs[2]) directly ---
        g[2][0] = xs[2]
        best_sims[2][0] = 0.0 # Reset baseline similarity at the start of DP
        
        total_forward_branches = 0
        entries_per_layer = {} # Track g entries count

        # Start DP loop from i=3 (Processing Block 2)
        for i in range(3, 2 * self.L + 1):
            block_idx = i - 1
            is_attn = (block_idx % 2 == 0)
            w = w_attn if is_attn else w_mlp
            xi_norm = self._norm(xs[i].squeeze(0))

            # Find valid states from the previous layer
            prev_js = [j for j, state in enumerate(g[i-1]) if state is not None]
            
            # g entry count tracking
            entries_per_layer[i-1] = len(prev_js)
            
            if not prev_js: continue

            # --- Vectorized Parallel Processing ---
            G_input = torch.cat([g[i-1][j] for j in prev_js], dim=1)
            num_branches = len(prev_js)
            total_forward_branches += num_branches
            
            # 1. Forward Execute (Mandatory for the last layer blocks)
            applied = self._apply_layer_single(block_idx, G_input, optimization_cache, branch_len=opt_num, num_branches=num_branches)
            G_exec_batch = applied.reshape(num_branches, opt_num, -1)
            sims_exec = torch.einsum('ntd,td->n', self._norm(G_exec_batch), xi_norm)
            
            for idx, j in enumerate(prev_js):
                s_exec = sims_exec[idx]
                if (
                    sim_threshold <= -1.0
                    or (s_exec / opt_num) >= sim_threshold
                ):
                    if s_exec > best_sims[i, j]:
                        best_sims[i, j] = s_exec
                        g[i][j] = G_exec_batch[idx:idx+1]
                        parent[i][j] = (j, False)

            # 2. Forward Skip (Only if NOT the last layer blocks)
            if block_idx < (2 * self.L - 2):
                G_skip_batch = G_input.reshape(num_branches, opt_num, -1)
                sims_skip = torch.einsum('ntd,td->n', self._norm(G_skip_batch), xi_norm)

                for idx, j in enumerate(prev_js):
                    target_j = j + w
                    if target_j <= budget:
                        s_skip = sims_skip[idx]
                        if (
                            sim_threshold <= -1.0
                            or (s_skip / opt_num) >= sim_threshold
                        ):
                            if s_skip > best_sims[i, target_j]:
                                best_sims[i, target_j] = s_skip
                                g[i][target_j] = G_skip_batch[idx:idx+1]
                                parent[i][target_j] = (j, True)

        # 3. Final Path Selection
        valid_js = [j for j, s in enumerate(g[-1]) if s is not None]
        entries_per_layer[2 * self.L] = len(valid_js)
        
        if not valid_js:
            print("[Warning] All paths pruned by threshold. Falling back to default.")
            self.skip_set = [0] * (2 * self.L)
            fallback_d, fallback_tpt, fallback_metadata = self._score_candidate(
                1.0,
                self.skip_set,
                t_attn,
                t_mlp,
            )
            self.arm_candidates = [
                KnapspecArm(
                    skip_set=list(self.skip_set),
                    budget=0,
                    alpha=1.0,
                    estimated_tpt=fallback_tpt,
                    estimated_draft_length=fallback_d,
                    tpt_rank=1,
                    full_weighted_budget=max_time_saved,
                    dp_budget_limit=budget,
                    dp_budget_fraction=self.dp_budget_fraction,
                    attn_weight=w_attn,
                    mlp_weight=w_mlp,
                    scoring_metadata=fallback_metadata,
                )
            ]
            self.scoring_history.append({
                "context_length": past_key_values_length,
                "arms": [arm.to_dict() for arm in self.arm_candidates],
            })
            self.clear_cached_hidden_states()
            return list(self.arm_candidates)

        # Preserve the original projection and 0..17 ranking.
        hidden_stack = torch.cat([g[-1][j] for j in valid_js], dim=0)
        if self.multi_gpu:
            norm_device = module_device(self.text_model.norm, self.device)
            head_device = module_device(self.model.lm_head, norm_device)
            normed = self.text_model.norm(hidden_stack.to(norm_device)).to(head_device)
            teacher_normed = self.text_model.norm(xs[-1].to(norm_device)).to(head_device)
            top1_tokens = torch.argmax(self.model.lm_head(normed), dim=-1)
            teacher_top1 = torch.argmax(self.model.lm_head(teacher_normed), dim=-1).squeeze(0)
        else:
            normed = self.text_model.norm(hidden_stack)
            top1_tokens = torch.argmax(self.model.lm_head(normed), dim=-1)
            teacher_top1 = torch.argmax(self.model.lm_head(self.text_model.norm(xs[-1])), dim=-1).squeeze(0)
        alpha_list = ((top1_tokens == teacher_top1.unsqueeze(0)).sum(dim=-1).float() / opt_num).tolist()

        all_skip_sets = []
        for j_start in valid_js:
            mask = [0] * (2 * self.L)
            curr_j = j_start
            for i in range(2 * self.L, 2, -1):
                p_info = parent[i][curr_j]
                if p_info is None: break
                prev_j, skipped = p_info
                if skipped: mask[i-1] = 1
                curr_j = prev_j
            all_skip_sets.append(mask)

        # Plain KnapSpec and both selectors share the original TPT ranking.
        scored_candidates: List[KnapspecArm] = []
        seen_masks = set()
        for candidate_j, candidate_alpha, candidate_mask in zip(
            valid_js, alpha_list, all_skip_sets
        ):
            if candidate_j == 0 and len(valid_js) > 1:
                continue
            mask_key = tuple(candidate_mask)
            if mask_key in seen_masks:
                continue
            seen_masks.add(mask_key)
            candidate_d, candidate_tpt, candidate_metadata = self._score_candidate(
                candidate_alpha,
                candidate_mask,
                t_attn,
                t_mlp,
            )
            scored_candidates.append(
                KnapspecArm(
                    skip_set=list(candidate_mask),
                    budget=candidate_j,
                    alpha=float(candidate_alpha),
                    estimated_tpt=float(candidate_tpt),
                    estimated_draft_length=candidate_d,
                    scoring_metadata=candidate_metadata,
                )
            )

        if not scored_candidates:
            fallback_idx = valid_js.index(0) if 0 in valid_js else 0
            fallback_d, fallback_tpt, fallback_metadata = self._score_candidate(
                alpha_list[fallback_idx],
                all_skip_sets[fallback_idx],
                t_attn,
                t_mlp,
            )
            scored_candidates.append(
                KnapspecArm(
                    skip_set=list(all_skip_sets[fallback_idx]),
                    budget=valid_js[fallback_idx],
                    alpha=float(alpha_list[fallback_idx]),
                    estimated_tpt=float(fallback_tpt),
                    estimated_draft_length=fallback_d,
                    scoring_metadata=fallback_metadata,
                )
            )

        zero_skip_arm = None
        if self.include_zero_skip_arm:
            # A no-skip draft is the full model. Construct it explicitly so
            # numerical similarity pruning cannot remove the zero endpoint.
            zero_mask = [0] * (2 * self.L)
            zero_d, zero_tpt, zero_metadata = self._score_candidate(
                1.0, zero_mask, t_attn, t_mlp
            )
            zero_skip_arm = KnapspecArm(
                skip_set=zero_mask,
                budget=0,
                alpha=1.0,
                estimated_tpt=zero_tpt,
                estimated_draft_length=zero_d,
                scoring_metadata=zero_metadata,
            )
        self.arm_candidates = select_knapspec_arms(
            scored_candidates, num_arms, zero_skip_arm=zero_skip_arm
        )
        for arm in self.arm_candidates:
            arm.full_weighted_budget = max_time_saved
            arm.dp_budget_limit = budget
            arm.dp_budget_fraction = self.dp_budget_fraction
            arm.attn_weight = w_attn
            arm.mlp_weight = w_mlp
        self.scoring_history.append({
            "context_length": past_key_values_length,
            "arms": [arm.to_dict() for arm in self.arm_candidates],
        })
        predicted_best = min(self.arm_candidates, key=lambda arm: arm.tpt_rank)
        best_j = predicted_best.budget
        best_cos = (
            float(opt_num)
            if self.include_zero_skip_arm and best_j == 0
            else best_sims[2 * self.L, best_j].item()
        )
        self.skip_set = list(predicted_best.skip_set)

        # Housekeeping
        self.is_prefill_stage = False
        self.clear_cached_hidden_states()
        self.total_skip += sum(self.skip_set)

        # Accumulate stats
        self.sum_best_tpt += predicted_best.estimated_tpt
        self.sum_skip += sum(self.skip_set)
        self.sum_attn_skip += sum(self.skip_set[::2])
        self.sum_mlp_skip += sum(self.skip_set[1::2])
        self.optimize_count += 1
        
        optimize_end_time = time.perf_counter()
        opt_duration = optimize_end_time - optimize_start_time

        print("[Knapspec] Scoring=legacy (original length search 0..17)")
        print(f"[Knapspec] Predicted-best skip set {self.skip_set}")
        print(f"skips {sum(self.skip_set)}", 
              f"attn_skips {sum(self.skip_set[::2])}", 
              f"mlp_skips {sum(self.skip_set[1::2])}", 
              f"best_cos_avg {best_cos/opt_num:.4f}",
              f"best_alpha {predicted_best.alpha:.4f}",
              f"best_tpt {predicted_best.estimated_tpt:.4f}",
              f"best_budget {predicted_best.budget}/{budget}",
              f"past_len {past_key_values_length}")
        print(
            f"[Knapspec] SP^2EC arms: {len(self.arm_candidates)}/{num_arms} "
            f"(w_attn={w_attn}, w_mlp={w_mlp})"
        )
        for arm_idx, arm in enumerate(self.arm_candidates):
            print(
                f"  arm {arm_idx}: budget={arm.budget}, tpt_rank={arm.tpt_rank}, "
                f"alpha={arm.alpha:.4f}, estimated_tpt={arm.estimated_tpt:.4f}, "
                f"skips={sum(arm.skip_set)}"
            )

        print(f"[OPT STATS] Duration: {opt_duration:.4f}s")
        print(f"[OPT STATS] Total Forward Branches: {total_forward_branches}")
        print(f"[OPT STATS] g[-1] entries: {len(valid_js)}")
        # Optional: Print all layers entries if needed, or just average
        avg_entries = sum(entries_per_layer.values()) / len(entries_per_layer) if entries_per_layer else 0
        print(f"[OPT STATS] Avg g entries per layer: {avg_entries:.2f}")
        # print(f"[OPT STATS] Per layer entries: {entries_per_layer}")
        return list(self.arm_candidates)
