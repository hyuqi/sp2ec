# knapspec_generator.py
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from device_utils import model_input_device
from generation_limits import limit_generation_to_context
from llava_next import count_llava_next_visual_tokens
from utils import (
    Env,
    apply_generation_constraints,
    forward,
    forward_draft_divided_multi,
    forward_verify_divided_multi,
    forward_verify_tree_divided_multi,
    forward_multimodal_prefill,
    crop_kv_cache,
    select_kv_cache_path,
    decode_next_token,
    GenerationResult,
)
from multimodal import (
    get_text_model,
    is_supported_multimodal,
    prepare_multimodal_inputs,
)
from .knapspec import (
    Knapspec,
    KnapspecArm,
    select_knapspec_arm_subset,
    validate_knapspec_arm_index_range,
)
from .sp2ec import SinglePeakArmSelector
from tree_verification import DraftTree, confidence_topk, evaluate_greedy_tree

class KnapspecGenerator:
    def __init__(
        self,
        env: Env,
        gamma: int = 4,
        skip_budget_M: int = 8,
        optimize_interval: int = 64,
        coefficients: Optional[tuple] = None,
        sim_threshold: float = 0.5,
        num_arms: int = 1,
        enable_sp2ec: bool = False,
        tree: bool = False,
        beta: float = 0.1,
        draft_confidence_threshold: float = 0.7,
        dp_budget_fraction: float = 0.5,
        dynamic_draft_stopping: bool = True,
        include_zero_skip_arm: bool = False,
        arm_index_start: Optional[int] = None,
        arm_index_end: Optional[int] = None,
        scoring_mode: str = "legacy",
    ):
        self.env = env
        self.model = env.model
        self.gamma = int(gamma)
        self.optimize_interval = int(optimize_interval)
        self.coefficients = coefficients
        self.sim_threshold = float(sim_threshold)
        self.num_arms = int(num_arms)
        if self.num_arms < 1:
            raise ValueError("num_arms must be at least 1")
        self.enable_sp2ec = bool(enable_sp2ec)
        validate_knapspec_arm_index_range(arm_index_start, arm_index_end)
        if arm_index_start is not None:
            if not self.enable_sp2ec:
                raise ValueError("KnapSpec arm-index filtering requires enable_sp2ec=True")
            if arm_index_end >= self.num_arms:
                raise ValueError("arm_index_end must be smaller than source-pool num_arms")
        self.arm_index_start = arm_index_start
        self.arm_index_end = arm_index_end
        # num_arms remains the original DP shortlist size. The selector explores
        # only the retained arms, with local indices 0..num_active_arms-1.
        self.num_active_arms = (
            arm_index_end - arm_index_start + 1
            if arm_index_start is not None
            else self.num_arms
        )
        self.tree = bool(tree)
        self.beta = float(beta)
        self.dynamic_draft_stopping = bool(dynamic_draft_stopping)
        if scoring_mode != "legacy":
            raise ValueError("This sample implementation uses only original KnapSpec scoring.")
        self.scoring_mode = scoring_mode
        if not 0.0 <= draft_confidence_threshold <= 1.0:
            raise ValueError("draft_confidence_threshold must be in [0, 1]")
        if self.enable_sp2ec and self.num_active_arms > self.optimize_interval:
            raise ValueError("active num_arms cannot exceed optimize_interval")

        self.L = len(get_text_model(self.model).layers)
        self.M = int(skip_budget_M)
        self.cuda_devices = sorted(
            {parameter.device for parameter in self.model.parameters() if parameter.device.type == "cuda"},
            key=str,
        )
        self.step_count = 0
        self.threshold = float(draft_confidence_threshold)
        self.draft_confidence_threshold = self.threshold
        if not 0.0 < dp_budget_fraction <= 1.0:
            raise ValueError("dp_budget_fraction must be in (0, 1]")
        self.dp_budget_fraction = float(dp_budget_fraction)
        self.include_zero_skip_arm = bool(include_zero_skip_arm)
        
        # Timing statistics
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0
        self.total_optimization_time = 0.0
        self.total_accepted_length = 0
        self.last_round_skip_count = 0
        self.arm_selector = self._new_arm_selector()
        self.completed_arm_sets: List[Dict[str, Any]] = []
        self.runtime_arm_rounds = 0
        self.runtime_skip_sum = 0
        self.runtime_attn_skip_sum = 0
        self.runtime_mlp_skip_sum = 0
        self.total_tree_candidates = 0
        self.tree_rounds = 0

    def _new_arm_selector(self):
        return SinglePeakArmSelector(
            max_draft_length=self.gamma,
            beta=self.beta,
        )

    def _synchronize_cuda(self) -> None:
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)

    def _archive_current_arm_set(self) -> None:
        if self.arm_selector.has_arms:
            self.completed_arm_sets.append(self.arm_selector.snapshot())

    def _install_arms(self, arms: List[KnapspecArm]) -> None:
        arms = select_knapspec_arm_subset(
            arms, self.arm_index_start, self.arm_index_end
        )
        if not arms:
            return
        self._archive_current_arm_set()
        self.arm_selector.reset(arms)
        self.num_active_arms = len(arms)
        if self.arm_index_start is not None:
            print(
                f"[SP2EC KnapSpec] Retaining source indices "
                f"{self.arm_index_start}..{self.arm_index_end}: "
                f"{len(arms)} active arms (local indices 0..{len(arms) - 1})"
            )

    def _arm_history(self) -> List[Dict[str, Any]]:
        history = list(self.completed_arm_sets)
        if self.arm_selector.has_arms:
            history.append(self.arm_selector.snapshot())
        return history

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
            raise ValueError("Tree verification currently supports greedy decoding only.")
        self.knapspec_model = Knapspec(
            L=self.L,
            M=int(self.M),
            model=self.model,
            coefficients=self.coefficients,
            sim_threshold=self.sim_threshold,
            dp_budget_fraction=self.dp_budget_fraction,
            include_zero_skip_arm=self.include_zero_skip_arm,
            scoring_mode=self.scoring_mode,
        )
        
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
        input_ids = enc["input_ids"].to(model_input_device(self.env.model))
        if self.multimodal_prefill_inputs is not None:
            del enc

        eos_token_ids: List[int] = []
        if self.env.eos_id is not None:
            eos_token_ids.append(self.env.eos_id)

        self.step_count = 0
        self.total_accepted_length = 0
        output_ids: List[int] = []
        past_key_values = None

        total_accepted = 0
        total_drafted = 0
        total_tokens = 0
        total_layers = 0
        
        # Reset timing statistics
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0
        self.total_optimization_time = 0.0
        self.last_round_skip_count = 0
        self.arm_selector = self._new_arm_selector()
        self.completed_arm_sets = []
        self.runtime_arm_rounds = 0
        self.runtime_skip_sum = 0
        self.runtime_attn_skip_sum = 0
        self.runtime_mlp_skip_sum = 0
        self.total_tree_candidates = 0
        self.tree_rounds = 0

        generation_limit = max_new_tokens
        if reference_output_ids is not None:
            generation_limit = min(generation_limit, len(reference_output_ids))
        generation_limit, min_new_tokens = limit_generation_to_context(
            self.model,
            len(input_ids_list),
            generation_limit,
            min_new_tokens,
            label="KnapSpec",
        )

        total_start_time = time.perf_counter()
        while len(output_ids) < generation_limit:
            num_speculations = min(self.gamma, generation_limit - len(output_ids) - 1)
            prev_len = len(output_ids)
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_drafted,
            ) = self.single_step_speculation(
                input_ids=input_ids,
                input_ids_list=input_ids_list,
                output_ids=output_ids,
                num_speculations=num_speculations,
                past_key_values=past_key_values,
                eos_token_ids=eos_token_ids,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                reference_output_ids=reference_output_ids,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            # Check for zero progress
            if len(output_ids) == prev_len:
                break

            total_accepted += number_of_matches
            total_drafted += num_drafted
            total_tokens += (number_of_matches + 1)  # accepted + correction/bonus
            skip_count = self.last_round_skip_count
            total_layers += (2 * self.L - skip_count) * num_drafted + 2 * self.L
            self.step_count += 1

            # Stop if EOS appeared in committed output
            stop = False
            for eid in eos_token_ids:
                if eid in output_ids:
                    output_ids = output_ids[: output_ids.index(eid)]
                    stop = True
                    break
            if stop:
                break

        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        text = self.env.tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""
        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else None
        tpl = (total_tokens / total_layers) if total_layers > 0 else None
        avg_best_tpt = self.knapspec_model.sum_best_tpt / self.knapspec_model.optimize_count if self.knapspec_model.optimize_count > 0 else None
        if self.knapspec_model.optimize_count > 0:
            print("Average Skip_num", self.knapspec_model.total_skip / self.knapspec_model.optimize_count)

        # Print timing breakdown
        if self.total_draft_time + self.total_verify_time + self.total_optimization_time > 0:
            total_measured = self.total_draft_time + self.total_verify_time + self.total_optimization_time
            draft_pct = (self.total_draft_time / total_measured) * 100
            verify_pct = (self.total_verify_time / total_measured) * 100
            opt_pct = (self.total_optimization_time / total_measured) * 100
            
            print(f"[TIMING] Draft: {self.total_draft_time:.3f}s ({draft_pct:.1f}%)")
            print(f"[TIMING] Verify: {self.total_verify_time:.3f}s ({verify_pct:.1f}%)")
            print(f"[TIMING] Optimization: {self.total_optimization_time:.3f}s ({opt_pct:.1f}%)")
            print(f"[TIMING] D+V+O measured: {total_measured:.3f}s / Total: {total_time:.3f}s")

        return GenerationResult(
            text=text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            num_input_tokens=len(input_ids_list),
            num_visual_tokens=count_llava_next_visual_tokens(self.model, input_ids_list),
            acceptance_rate=acc_rate,
            tokens_per_layer=tpl,
            draft_time=self.total_draft_time,
            verify_time=self.total_verify_time,
            optimization_time=self.total_optimization_time,
            total_time=total_time,
            total_accepted_length=self.total_accepted_length,
            total_steps=self.step_count,
            avg_best_tpt=avg_best_tpt,
            arm_set_history=self._arm_history() if self.enable_sp2ec else None,
            candidate_scoring_history=self.knapspec_model.scoring_history,
            tree_candidate_tokens=self.total_tree_candidates if self.tree else None,
            avg_tree_candidates=(
                self.total_tree_candidates / self.tree_rounds
                if self.tree and self.tree_rounds > 0
                else None
            ),
        )

    def single_step_speculation(
        self,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        sample: bool = False,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        reference_output_ids: Optional[List[int]] = None,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Tuple[torch.Tensor, List[int], Optional[List[Tuple[torch.Tensor, torch.Tensor]]], int, int]:
        if self.step_count == 0:
            if self.multimodal_prefill_inputs is not None:
                logits, past_key_values = forward_multimodal_prefill(
                    self.model,
                    self.multimodal_prefill_inputs,
                )
                self.multimodal_prefill_inputs = None
            else:
                logits, past_key_values = forward(
                    self.model,
                    input_ids,
                    past_key_values,
                )
            logits = apply_generation_constraints(
                logits,
                output_ids,
                eos_token_ids=eos_token_ids,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            next_token, probabilities = decode_next_token(
                logits,
                token_idx = -1,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_token_id = int(next_token.item() if isinstance(next_token, torch.Tensor) else next_token)
            
            output_ids.append(next_token_id)
            input_ids = torch.tensor([[next_token_id]], device=input_ids.device)
            self.knapspec_model.is_prefill_stage = False
            return (
                input_ids,
                output_ids,
                past_key_values,
                0,
                0,
            )

        # accumulate_steps = 10
        accumulate_steps = 5
        accumulate_phase = (1 < (self.step_count % self.optimize_interval) <= (1 + accumulate_steps))
        optimization_phase = ((self.step_count % self.optimize_interval) == (1 + accumulate_steps))

        selected_arm_idx = None
        if self.enable_sp2ec and self.arm_selector.has_arms:
            selected_arm_idx = self.arm_selector.select_arm()
            selected_arm = self.arm_selector.arms[selected_arm_idx]
            self.knapspec_model.skip_set = list(selected_arm.skip_set)
            self.runtime_arm_rounds += 1
            self.runtime_skip_sum += sum(selected_arm.skip_set)
            self.runtime_attn_skip_sum += sum(selected_arm.skip_set[::2])
            self.runtime_mlp_skip_sum += sum(selected_arm.skip_set[1::2])

        self.last_round_skip_count = sum(self.knapspec_model.skip_set)
        round_start = None
        if selected_arm_idx is not None:
            self._synchronize_cuda()
            round_start = time.perf_counter()
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        draft_cache = past_key_values
        draft_tree = DraftTree.with_root(int(input_ids[0, -1].item())) if self.tree else None

        # Draft phase
        draft_start = time.perf_counter()
        for i in range(num_speculations):
            next_logits, draft_cache = forward_draft_divided_multi(
                self.model,
                draft_input_ids,
                self.knapspec_model.skip_set,
                draft_cache,
            )
            next_logits = apply_generation_constraints(
                next_logits,
                output_ids + draft_output_ids,
                eos_token_ids=eos_token_ids,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
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
            draft_output_ids.append(next_id)
            draft_token_confidence_score = (
                float(next_prob[0, next_id].item())
                if next_prob is not None
                else 1.0
            )
            if draft_tree is not None:
                candidate_count = min(
                    confidence_topk(draft_token_confidence_score),
                    next_prob.shape[-1],
                )
                candidate_ids = torch.topk(
                    next_prob[0],
                    k=candidate_count,
                    dim=-1,
                ).indices.tolist()
                draft_tree.add_level(candidate_ids)
            draft_input_ids = torch.tensor([[next_id]], device=draft_input_ids.device)
            if next_id in eos_token_ids:
                break

            if (
                self.dynamic_draft_stopping
                and not self.knapspec_model.is_prefill_stage
                and draft_token_confidence_score < self.threshold
            ):
                break

        draft_end = time.perf_counter()
        self.total_draft_time += (draft_end - draft_start)
        if draft_tree is not None:
            self.total_tree_candidates += draft_tree.candidate_count
            self.tree_rounds += 1

        # Verify phase
        verify_start = time.perf_counter()
        
        selected_tree_indices = None
        verification_prefix_length = past_key_values[0][0].shape[2]
        if draft_tree is not None:
            tree_input_ids = torch.tensor(
                [draft_tree.token_ids],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            verification_logits, past_key_values = forward_verify_tree_divided_multi(
                self.model,
                tree_input_ids,
                draft_tree.parent_indices,
                draft_tree.depths,
                past_key_values,
                self.knapspec_model,
                accumulate_phase,
            )
            target_predictions = verification_logits[0].argmax(dim=-1).tolist()
            reference_suffix = None
            if reference_output_ids is not None:
                reference_suffix = reference_output_ids[len(output_ids):]
            accepted_tokens, number_of_matches, selected_tree_indices = evaluate_greedy_tree(
                draft_tree,
                target_predictions,
                reference_token_ids=reference_suffix,
            )
        else:
            # Prepare tokens for linear verification.
            draft_tensor = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
            prefill_token_ids = torch.cat([input_ids, draft_tensor], dim=-1)
            verification_logits, past_key_values = forward_verify_divided_multi(
                self.model, prefill_token_ids, past_key_values, self.knapspec_model, accumulate_phase
            )

        # Determine verified tokens for the original linear path.
        if draft_tree is None and reference_output_ids is not None:
             # Use reference to get ground truth tokens
            current_idx = len(output_ids)
            verified_tokens_list = []
            
            check_len = len(draft_output_ids) + 1
            for i in range(check_len):
                if current_idx + i < len(reference_output_ids):
                    verified_tokens_list.append(reference_output_ids[current_idx + i])
                else:
                    break
            
            verified_tokens = torch.tensor([verified_tokens_list], device=prefill_token_ids.device)

        elif draft_tree is None:
            # Standard decoding from logits
            # Get logits for the drafted positions + one extra token
            prompt_length = input_ids.shape[1]  # Original input length
            verification_logits = verification_logits[:, prompt_length - 1:, :]  # [1, T_d + 1, V]

            # Decode verified tokens from the verification logits
            verified_tokens, verified_probabilities = decode_next_token(
                logits=verification_logits,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        if draft_tree is None:
            verified_tokens = verified_tokens.to(input_ids.device)

            # Compare draft vs verified
            draft_tensor_1d = torch.tensor(draft_output_ids, device=input_ids.device).unsqueeze(0)  # [1, T_d]

            # Handle potential shape mismatch (e.g. at the end of teacher forcing sequence)
            if verified_tokens.shape[1] > draft_tensor_1d.shape[1]:
                verified_comparison = verified_tokens[:, :-1]
            else:
                min_len = min(verified_tokens.shape[1], draft_tensor_1d.shape[1])
                verified_comparison = verified_tokens[:, :min_len]
                draft_tensor_1d = draft_tensor_1d[:, :min_len]

            verified = draft_tensor_1d == verified_comparison

            # Count number of matches (consecutive from the beginning)
            if not sample:
                number_of_matches = ((~verified).cumsum(dim=-1) < 1).sum().item()
            else:
                number_of_matches = 0
                for i in range(draft_tensor_1d.numel()):
                    if bool(verified[0, i].item()):
                        number_of_matches += 1
                    else:
                        break

            accepted_tokens = draft_output_ids[:number_of_matches]
            if number_of_matches < verified_tokens.shape[1]:
                additional_token = int(verified_tokens[0, number_of_matches].item())
                accepted_tokens.append(additional_token)

        if not accepted_tokens:
             if selected_arm_idx is not None:
                 self._synchronize_cuda()
                 round_elapsed = time.perf_counter() - round_start
                 self.arm_selector.update(selected_arm_idx, 0, round_elapsed)
             verify_end = time.perf_counter()
             self.total_verify_time += (verify_end - verify_start)
             return (
                 input_ids,
                 output_ids,
                 past_key_values,
                 number_of_matches,
                 len(draft_output_ids),
            )

        # State update
        new_token_id = accepted_tokens[-1]
        input_ids = torch.tensor([[new_token_id]], device=input_ids.device)
        output_ids.extend(accepted_tokens)

        # Keep only committed cache entries; rejected tree siblings are gathered out.
        if selected_tree_indices is not None:
            # Tree verification returns a fresh DynamicCache result, so its tail
            # can be consumed without mutating the cache passed into verification.
            past_key_values = select_kv_cache_path(
                past_key_values,
                verification_prefix_length,
                selected_tree_indices,
                allow_in_place=True,
            )
        else:
            new_len = len(input_ids_list) + len(output_ids) - 1
            past_key_values = crop_kv_cache(past_key_values, new_len)

        if selected_arm_idx is not None:
            self._synchronize_cuda()
            round_elapsed = time.perf_counter() - round_start
            self.arm_selector.update(
                selected_arm_idx,
                len(accepted_tokens),
                round_elapsed,
            )

        verify_end = time.perf_counter()
        self.total_verify_time += (verify_end - verify_start)
        self.total_accepted_length += len(accepted_tokens)


        if accumulate_phase:
            for i in range(2*self.L+1):
                hidden = self.knapspec_model.cached_hidden_states[i][-1]
                if selected_tree_indices is not None:
                    path_indices = torch.tensor(
                        selected_tree_indices,
                        dtype=torch.long,
                        device=hidden.device,
                    )
                    cropped = hidden.index_select(1, path_indices).clone()
                else:
                    cropped = hidden[:, :number_of_matches + 1, :].clone()
                self.knapspec_model.cached_hidden_states[i][-1] = cropped

        # Optimization phase with timing
        if optimization_phase:
            if self.enable_sp2ec:
                self._synchronize_cuda()
            opt_start = time.perf_counter()
            arms = self.knapspec_model.optimize(
                past_key_values=past_key_values,
                num_arms=self.num_arms if self.enable_sp2ec else 1,
            )
            if self.enable_sp2ec:
                self._synchronize_cuda()
            opt_end = time.perf_counter()
            self.total_optimization_time += (opt_end - opt_start)
            if self.enable_sp2ec:
                self._install_arms(arms)

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            len(draft_output_ids),
        )
