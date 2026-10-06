# utils.py
import operator
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import transformers
import math

from device_utils import model_uses_multiple_cuda_devices, module_device, move_to_device
from multimodal import (
    get_qwen_vl_rope_deltas,
    get_text_model,
    is_qwen_vl_with_mrope,
    is_supported_multimodal,
)
from qwen3_vl import is_qwen3_vl

@dataclass
class Env:
    model: PreTrainedModel
    tok: PreTrainedTokenizerBase
    device: str
    eos_id: Optional[int]
    pad_id: Optional[int]
    processor: Optional[Any] = None

@dataclass
class GenerationResult:
    text: str
    num_output_tokens: int
    output_ids: Optional[List[int]] = None
    num_input_tokens: Optional[int] = None
    # Speculative Decoding fields
    acceptance_rate: float = None
    tokens_per_layer: float = None
    draft_time: float = None
    verify_time: float = None
    optimization_time: float = None
    total_time: float = None
    total_accepted_length: int = None
    total_steps: int = None # speculation step
    avg_best_tpt: float = None
    arm_set_history: Optional[List[Dict[str, Any]]] = None
    tree_candidate_tokens: int = None
    avg_tree_candidates: float = None
    num_visual_tokens: Optional[int] = None
    # Predicted candidate scores, separate from empirical bandit arm statistics.
    candidate_scoring_history: Optional[List[Dict[str, Any]]] = None

# ------------------------
# Attention mask helpers
# ------------------------

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Create causal attention mask for decoder."""
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask

def _make_branch_parallel_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int,
    branch_len: int,
    num_branches: Optional[int] = None,
):
    bsz, tgt_len = input_ids_shape
    if num_branches is None:
        assert tgt_len % branch_len == 0
        num_branches = tgt_len // branch_len

    total_src_len = past_key_values_length + tgt_len

    mask = torch.full(
        (tgt_len, total_src_len),
        torch.finfo(dtype).min,
        device=device,
    )
    mask = mask.to(dtype)

    for b in range(num_branches):
        for t in range(branch_len):
            q_idx = b * branch_len + t

            if past_key_values_length > 0:
                mask[q_idx, :past_key_values_length] = 0.0

            branch_start = past_key_values_length + b * branch_len
            branch_end = branch_start + t + 1
            mask[q_idx, branch_start:branch_end] = 0.0

    mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, total_src_len)
    return mask


def _make_tree_causal_mask(
    parent_indices: List[int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int,
) -> torch.Tensor:
    """Allow each linearized tree node to attend only to its ancestors."""
    tree_len = len(parent_indices)
    mask = torch.full(
        (tree_len, past_key_values_length + tree_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = 0.0

    for node_idx in range(tree_len):
        ancestor_idx = node_idx
        while ancestor_idx >= 0:
            if ancestor_idx > node_idx:
                raise ValueError("tree parents must precede their children")
            mask[node_idx, past_key_values_length + ancestor_idx] = 0.0
            ancestor_idx = parent_indices[ancestor_idx]

    return mask[None, None, :, :]

# ------------------------
# KV Cache helpers
# ------------------------

_FAST_TREE_CACHE_ENABLED = os.environ.get(
    "KNAPSPEC_FAST_TREE_CACHE",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}


def crop_kv_cache(kv_cache, target_length: int):
    """Crop KV cache to target_length"""
    if kv_cache is None:
        return None
    
    cropped_cache = []
    for layer_cache in kv_cache:
        if layer_cache is None:
            cropped_cache.append(None)
        else:
            key, value = layer_cache
            cropped_key = key[:, :, :target_length, :]
            cropped_value = value[:, :, :target_length, :]
            cropped_cache.append((cropped_key, cropped_value))
    
    return tuple(cropped_cache)


def _validate_kv_cache_path(
    kv_cache,
    prefix_length: int,
    selected_tree_indices: Sequence[int],
) -> Tuple[int, Tuple[int, ...]]:
    """Validate the legacy ``[batch, kv_heads, sequence, head_dim]`` cache."""
    try:
        normalized_prefix_length = operator.index(prefix_length)
    except TypeError as exc:
        raise TypeError("prefix_length must be an integer") from exc
    if normalized_prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")

    normalized_tree_indices = []
    for tree_index in selected_tree_indices:
        try:
            normalized_tree_indices.append(operator.index(tree_index))
        except TypeError as exc:
            raise TypeError("selected_tree_indices must contain integers") from exc
    normalized_tree_indices = tuple(normalized_tree_indices)

    if not isinstance(kv_cache, (tuple, list)):
        raise TypeError("kv_cache must be a legacy tuple/list of layer caches")

    for layer_idx, layer_cache in enumerate(kv_cache):
        if layer_cache is None:
            continue
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
            raise TypeError(f"layer {layer_idx} cache must be a (key, value) pair")

        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"layer {layer_idx} key and value must be tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"layer {layer_idx} key and value must be rank-4 tensors")
        if key.shape[2] != value.shape[2]:
            raise ValueError(f"layer {layer_idx} key/value sequence lengths must match")

        sequence_length = key.shape[2]
        if normalized_prefix_length > sequence_length:
            raise ValueError(
                f"prefix_length exceeds the sequence length for layer {layer_idx}"
            )
        tree_length = sequence_length - normalized_prefix_length
        for tree_index in normalized_tree_indices:
            if tree_index < 0 or tree_index >= tree_length:
                raise ValueError(
                    f"tree index {tree_index} is outside [0, {tree_length}) "
                    f"for layer {layer_idx}"
                )

    return normalized_prefix_length, normalized_tree_indices


def _cache_path_indices(
    absolute_indices: Sequence[int],
    device: torch.device,
    indices_by_device: Dict[torch.device, torch.Tensor],
) -> torch.Tensor:
    """Create one reusable index tensor per cache device."""
    device = torch.device(device)
    indices = indices_by_device.get(device)
    if indices is None:
        indices = torch.tensor(absolute_indices, dtype=torch.long, device=device)
        indices_by_device[device] = indices
    return indices


def _select_kv_cache_path_allocating(
    kv_cache,
    prefix_length: int,
    selected_tree_indices: Sequence[int],
):
    """Reference/fallback path that allocates and gathers the whole prefix."""
    absolute_tree_indices = tuple(
        prefix_length + idx for idx in selected_tree_indices
    )
    indices_by_device: Dict[torch.device, torch.Tensor] = {}
    selected_cache = []
    for layer_cache in kv_cache:
        if layer_cache is None:
            selected_cache.append(None)
            continue
        key, value = layer_cache
        key_device = torch.device(key.device)
        key_indices = indices_by_device.get(key_device)
        if key_indices is None:
            key_indices = torch.cat(
                (
                    torch.arange(prefix_length, dtype=torch.long, device=key_device),
                    torch.tensor(
                        absolute_tree_indices,
                        dtype=torch.long,
                        device=key_device,
                    ),
                )
            )
            indices_by_device[key_device] = key_indices

        value_device = torch.device(value.device)
        value_indices = indices_by_device.get(value_device)
        if value_indices is None:
            value_indices = torch.cat(
                (
                    torch.arange(prefix_length, dtype=torch.long, device=value_device),
                    torch.tensor(
                        absolute_tree_indices,
                        dtype=torch.long,
                        device=value_device,
                    ),
                )
            )
            indices_by_device[value_device] = value_indices
        selected_cache.append(
            (
                key.index_select(2, key_indices),
                value.index_select(2, value_indices),
            )
        )
    return tuple(selected_cache)


def _can_compact_kv_cache_path_in_place(
    kv_cache,
    prefix_length: int,
    path_length: int,
) -> bool:
    """Limit mutation to the contiguous legacy caches used during inference."""
    if torch.is_grad_enabled():
        return False

    for layer_cache in kv_cache:
        if layer_cache is None:
            continue
        key, value = layer_cache
        if prefix_length + path_length > key.shape[2]:
            # Repeated indices can make the output longer than the tree tail.
            # Preserve the allocating implementation's behavior in that case.
            return False
        for tensor in (key, value):
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                return False
            if tensor.storage_offset() != 0 or getattr(tensor, "_base", None) is not None:
                # Avoid mutating ordinary views. The caller must still guarantee
                # ownership because detached tensors can share storage without _base.
                return False
            is_inference = getattr(tensor, "is_inference", None)
            if (
                callable(is_inference)
                and is_inference()
                and not torch.is_inference_mode_enabled()
            ):
                return False
    return True


def _select_kv_cache_path_in_place(
    kv_cache,
    prefix_length: int,
    selected_tree_indices: Sequence[int],
):
    """Gather only the selected tree tail and leave the committed prefix in place."""
    path_length = len(selected_tree_indices)
    new_length = prefix_length + path_length
    absolute_tree_indices = tuple(
        prefix_length + idx for idx in selected_tree_indices
    )
    indices_by_device: Dict[torch.device, torch.Tensor] = {}
    selected_cache = []

    for layer_cache in kv_cache:
        if layer_cache is None:
            selected_cache.append(None)
            continue

        key, value = layer_cache
        if path_length:
            key_indices = _cache_path_indices(
                absolute_tree_indices,
                key.device,
                indices_by_device,
            )
            value_indices = _cache_path_indices(
                absolute_tree_indices,
                value.device,
                indices_by_device,
            )

            # Materialize the complete path before overwriting any tree slots.
            # This is required for non-consecutive or overlapping source indices.
            selected_key = key.index_select(2, key_indices)
            selected_value = value.index_select(2, value_indices)
            key.narrow(2, prefix_length, path_length).copy_(selected_key)
            value.narrow(2, prefix_length, path_length).copy_(selected_value)

        selected_cache.append(
            (
                key.narrow(2, 0, new_length),
                value.narrow(2, 0, new_length),
            )
        )
    return tuple(selected_cache)


def select_kv_cache_path(
    kv_cache,
    prefix_length: int,
    selected_tree_indices: Sequence[int],
    *,
    allow_in_place: bool = False,
):
    """Compact a linearized tree cache to the committed prefix and path.

    Tree verification appends all linearized tree nodes after an already committed
    prefix. When ``allow_in_place`` is true during inference, only the selected
    path is gathered into the cache tail and the unchanged prefix is returned as
    a view. The caller must exclusively own the verified cache because its tree
    tail is mutated. Grad-enabled calls and unexpected tensor layouts retain the
    original allocating implementation. Set ``KNAPSPEC_FAST_TREE_CACHE=0`` before
    starting Python to force the allocating path for an A/B comparison.
    """
    if kv_cache is None:
        return None

    prefix_length, selected_tree_indices = _validate_kv_cache_path(
        kv_cache,
        prefix_length,
        selected_tree_indices,
    )
    if (
        allow_in_place
        and _FAST_TREE_CACHE_ENABLED
        and _can_compact_kv_cache_path_in_place(
            kv_cache,
            prefix_length,
            len(selected_tree_indices),
        )
    ):
        return _select_kv_cache_path_in_place(
            kv_cache,
            prefix_length,
            selected_tree_indices,
        )
    return _select_kv_cache_path_allocating(
        kv_cache,
        prefix_length,
        selected_tree_indices,
    )


def _decode_position_ids(
    model,
    start: int,
    length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build text decode positions, including Qwen-VL's multimodal offset."""
    positions = torch.arange(start, start + length, dtype=torch.long, device=device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    if not is_qwen_vl_with_mrope(model):
        return positions

    rope_deltas = get_qwen_vl_rope_deltas(model)
    if rope_deltas is None:
        raise RuntimeError(
            "Qwen-VL native prefill must run before speculative decoding."
        )
    deltas = rope_deltas.to(device=device, dtype=torch.long).reshape(-1, 1)
    if deltas.shape[0] == 1 and batch_size > 1:
        deltas = deltas.expand(batch_size, -1)
    if deltas.shape[0] != batch_size:
        raise ValueError(
            "Qwen-VL rope_deltas batch size does not match the decode batch."
        )
    return (positions + deltas).unsqueeze(0).expand(3, -1, -1)


def _tree_position_ids(
    model,
    past_key_values_length: int,
    depths: List[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.tensor(depths, dtype=torch.long, device=device)
    positions = (positions + past_key_values_length).unsqueeze(0).expand(batch_size, -1)
    if not is_qwen_vl_with_mrope(model):
        return positions

    rope_deltas = get_qwen_vl_rope_deltas(model)
    if rope_deltas is None:
        raise RuntimeError(
            "Qwen-VL native prefill must run before tree verification."
        )
    deltas = rope_deltas.to(device=device, dtype=torch.long).reshape(-1, 1)
    if deltas.shape[0] == 1 and batch_size > 1:
        deltas = deltas.expand(batch_size, -1)
    if deltas.shape[0] != batch_size:
        raise ValueError(
            "Qwen-VL rope_deltas batch size does not match the tree batch."
        )
    return (positions + deltas).unsqueeze(0).expand(3, -1, -1)


def forward_multimodal_prefill(model, model_inputs):
    """Run a native multimodal prefill and return a legacy text KV cache."""
    if not is_supported_multimodal(model):
        raise TypeError("forward_multimodal_prefill requires a supported multimodal model.")
    outputs = model(**model_inputs, use_cache=True, logits_to_keep=1)
    past_key_values = outputs.past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return outputs.logits, past_key_values


def forward_qwen3_vl_prefill(model, model_inputs):
    """Compatibility wrapper for the original Qwen3-VL prefill helper."""
    if not is_qwen3_vl(model):
        raise TypeError("forward_qwen3_vl_prefill requires a Qwen3-VL model.")
    return forward_multimodal_prefill(model, model_inputs)


# ------------------------
# Sampling helpers
# ------------------------

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(
            top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(
            top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
        )(None, logits)

    return logits


def apply_generation_constraints(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    *,
    eos_token_ids: Optional[Sequence[int]] = None,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Apply one shared next-token policy to target and draft logits."""
    if min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")
    if logits.dim() not in {2, 3}:
        raise ValueError("next-token constraints require rank-2 or rank-3 logits")

    constrained = logits.clone()
    scores = constrained[:, -1, :] if constrained.dim() == 3 else constrained
    history_ids = [int(token_id) for token_id in generated_ids]
    if history_ids:
        history = torch.tensor(
            [history_ids],
            dtype=torch.long,
            device=scores.device,
        )
        if repetition_penalty != 1.0:
            processor = transformers.generation.logits_process.RepetitionPenaltyLogitsProcessor(
                repetition_penalty
            )
            scores = processor(history, scores)
        if no_repeat_ngram_size > 0 and len(history_ids) + 1 >= no_repeat_ngram_size:
            processor = transformers.generation.logits_process.NoRepeatNGramLogitsProcessor(
                no_repeat_ngram_size
            )
            scores = processor(history, scores)

    if len(history_ids) < min_new_tokens and eos_token_ids:
        valid_eos_ids = [
            int(eos_id)
            for eos_id in eos_token_ids
            if 0 <= int(eos_id) < scores.shape[-1]
        ]
        if valid_eos_ids:
            scores[..., valid_eos_ids] = torch.finfo(scores.dtype).min

    if constrained.dim() == 3:
        constrained[:, -1, :] = scores
        return constrained
    return scores


# ------------------------
# Decode helpers
# ------------------------

def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        if not token_idx:
            logits = logits.squeeze(dim=0)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return next_token, probabilities
    else:
        if not token_idx:
            logits = logits.squeeze(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token = next_token.transpose(1, 0)
        return next_token, probabilities


# ------------------------
# Forward passes
# ------------------------

def _move_decoder_state_to_layer(
    decoder_layer,
    hidden_states,
    attention_mask,
    position_embeddings,
    cache_position,
):
    """Move manually dispatched decoder state to one sharded layer."""
    layer_device = module_device(decoder_layer, hidden_states.device)
    return (
        move_to_device(hidden_states, layer_device),
        move_to_device(attention_mask, layer_device),
        move_to_device(position_embeddings, layer_device),
        move_to_device(cache_position, layer_device),
    )


def _finish_sharded_decoder_forward(model, text_model, hidden_states):
    """Run final norm/head on their assigned devices for a sharded model."""
    norm_device = module_device(text_model.norm, hidden_states.device)
    hidden_states = text_model.norm(hidden_states.to(norm_device))
    lm_head_device = module_device(model.lm_head, hidden_states.device)
    return model.lm_head(hidden_states.to(lm_head_device))

def forward(model, input_ids: torch.Tensor, past_kv_cache=None):
    if is_supported_multimodal(model):
        if past_kv_cache is None:
            raise RuntimeError("Use forward_multimodal_prefill for a multimodal model's first forward pass.")
        return forward_verify_divided_multi(model, input_ids, past_kv_cache)

    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        past_key_values_length = past_kv_cache[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_divided(model, input_ids: torch.Tensor, past_kv_cache=None,):
    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        past_key_values_length = past_kv_cache[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    for idx, decoder_layer in enumerate(model.model.layers): 
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache

def forward_draft(model, input_ids: torch.Tensor, skip_set: List[int], past_kv_cache=None):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        max_cache_length = 0
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_length = layer_cache[0].shape[2]
                max_cache_length = max(max_cache_length, cache_length)
        past_key_values_length = max_cache_length
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    for i, decoder_layer in enumerate(model.model.layers):
        if skip_set[i] == 1:
            continue
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify(model, verify_input: torch.Tensor, draft_tokens: List[int], past_kv_cache=None, clasp=None, optimization_phase=False):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    seq_length_with_past = seq_length
    draft_past_key_values_length = 0
    full_past_key_values_length = 0
    prompt_length = seq_length - len(draft_tokens)

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:
        draft_past_key_values_length = past_kv_cache[0][0].shape[2]
        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
        
        seq_length_with_past = seq_length + draft_past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    layer_hidden = hidden_states[:, prompt_length-1:, :]
    if optimization_phase:
        clasp.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers and collect hidden_states for CLASP
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states= decoder_layer(
            hidden_states,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        
        # Store layer output for CLASP (verification tokens only)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]
            clasp.cached_hidden_states[idx+1].append(layer_hidden.detach())

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_draft_multi(model, input_ids: torch.Tensor, skip_set: List[int], past_kv_cache=None):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        max_cache_length = 0
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_length = layer_cache[0].shape[2]
                max_cache_length = max(max_cache_length, cache_length)
        past_key_values_length = max_cache_length
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    # Decoder layers with skipping using skip_set directly
    for i, decoder_layer in enumerate(model.model.layers):
        if skip_set[i] == 1:  # 1 means skip this layer
            continue
            
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_multi(
    model, 
    verify_input: torch.Tensor, 
    past_kv_cache=None, 
    clasp=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds    
    if optimization_phase:
        if clasp.is_prefill_stage:
            layer_hidden = hidden_states[:, -64:, :]
        else:
            layer_hidden = hidden_states
        clasp.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers and collect hidden_states for CLASP
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        
        # Store layer output for CLASP (verification tokens only)
        if optimization_phase:
            if clasp.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            clasp.cached_hidden_states[idx+1].append(layer_hidden.detach())

    past_kv_cache = past_kv_cache.to_legacy_cache()    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_draft_divided(
    model,
    input_ids: torch.Tensor,
    skip_set: List[int],
    past_kv_cache: Optional[list] = None,
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values_length = 0
    if past_kv_cache is not None:
        # Find maximum cache length across all layers
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_len = layer_cache[0].shape[2]  # key tensor: [batch, heads, seq_len, head_dim]
                past_key_values_length = max(past_key_values_length, cache_len)

    seq_with_past = seq_length + past_key_values_length

    # Convert legacy cache to DynamicCache for processing
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    position_ids = torch.arange(
        past_key_values_length, 
        past_key_values_length + seq_length, 
        dtype=torch.long, 
        device=device
    ).unsqueeze(0)  # [1, seq_length]

    attention_mask_bool = input_ids.new_ones((batch_size, seq_with_past), dtype=torch.bool)
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model, 
        attention_mask_bool, 
        (batch_size, seq_length), 
        inputs_embeds, 
        past_key_values_length
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    for idx, decoder_layer in enumerate(model.model.layers):
        attn_skip = bool(skip_set[2 * idx + 0])
        mlp_skip = bool(skip_set[2 * idx + 1])

        if not attn_skip:
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_output
        else:
            pass

        if not mlp_skip:
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(normed_hidden)
            hidden_states = hidden_states + mlp_output
        else:
            pass

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_divided(
    model, 
    verify_input: torch.Tensor, 
    draft_tokens: List[int], 
    past_kv_cache=None, 
    claspd=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0
    prompt_length = seq_length - len(draft_tokens)

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    
    # Store initial embedding hidden states (index 0)
    if optimization_phase:
        layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
        claspd.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers with attention/MLP division
    for idx, decoder_layer in enumerate(model.model.layers):
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output
        
        # Store hidden states after attention (index 2*idx + 1)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
            claspd.cached_hidden_states[2*idx + 1].append(layer_hidden.detach())

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output
        
        # Store hidden states after MLP (index 2*idx + 2)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
            claspd.cached_hidden_states[2*idx + 2].append(layer_hidden.detach())
    
    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache



def forward_draft_divided_multi(
    model,
    input_ids: torch.Tensor,
    skip_set: List[int],
    past_kv_cache: Optional[list] = None,
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values_length = 0
    if past_kv_cache is not None:
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_len = layer_cache[0].shape[2]
                past_key_values_length = max(past_key_values_length, cache_len)

    seq_with_past = seq_length + past_key_values_length

    # Convert legacy cache to DynamicCache for processing
    past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # ---- Prepare position IDs ----
    position_ids = _decode_position_ids(
        model,
        past_key_values_length,
        seq_length,
        batch_size,
        device,
    )
    cache_position = torch.arange(
        past_key_values_length,
        past_key_values_length + seq_length,
        dtype=torch.long,
        device=device,
    )

    # ---- Prepare attention mask ----
    attention_mask_bool = input_ids.new_ones(
        (batch_size, seq_with_past), 
        dtype=torch.bool
    )
    text_model = get_text_model(model)
    multi_gpu = model_uses_multiple_cuda_devices(model)
    inputs_embeds = text_model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model, 
        attention_mask_bool, 
        (batch_size, seq_length), 
        inputs_embeds, 
        past_key_values_length
    )

    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    for idx, decoder_layer in enumerate(text_model.layers):
        attn_skip = bool(skip_set[2 * idx + 0])
        mlp_skip = bool(skip_set[2 * idx + 1])

        if multi_gpu and not (attn_skip and mlp_skip):
            hidden_states, attention_mask, position_embeddings, cache_position = (
                _move_decoder_state_to_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    cache_position,
                )
            )

        if not attn_skip:
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                cache_position=cache_position,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_output
        else:
            pass

        if not mlp_skip:
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(normed_hidden)
            hidden_states = hidden_states + mlp_output
        else:
            pass

    past_kv_cache = past_kv_cache.to_legacy_cache()
    if multi_gpu:
        logits = _finish_sharded_decoder_forward(model, text_model, hidden_states)
    else:
        hidden_states = text_model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


@torch.inference_mode()
def forward_draft_divided_multi_with_importance(
    model,
    input_ids: torch.Tensor,
    skip_set: List[int],
    past_kv_cache: Optional[list] = None,
    protected_edge_blocks: int = 2,
):
    """Draft forward pass returning GPU-reduced block cosine importance."""
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values_length = 0
    if past_kv_cache is not None:
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                past_key_values_length = max(
                    past_key_values_length,
                    layer_cache[0].shape[2],
                )

    dynamic_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    position_ids = _decode_position_ids(
        model,
        past_key_values_length,
        seq_length,
        batch_size,
        device,
    )
    cache_position = torch.arange(
        past_key_values_length,
        past_key_values_length + seq_length,
        dtype=torch.long,
        device=device,
    )

    text_model = get_text_model(model)
    num_blocks = len(text_model.layers)
    protected_edge_blocks = int(protected_edge_blocks)
    if protected_edge_blocks < 0 or 2 * protected_edge_blocks >= num_blocks:
        raise ValueError(
            "protected_edge_blocks must be non-negative and leave at least "
            "one unprotected transformer block"
        )
    multi_gpu = model_uses_multiple_cuda_devices(model)
    inputs_embeds = text_model.embed_tokens(input_ids)
    attention_mask_bool = input_ids.new_ones(
        (batch_size, seq_length + past_key_values_length),
        dtype=torch.bool,
    )
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask_bool,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    block_inputs = []
    block_outputs = []
    sharded_importance = []
    importance_mask = []

    for idx, decoder_layer in enumerate(text_model.layers):
        attn_skip = bool(skip_set[2 * idx])
        mlp_skip = bool(skip_set[2 * idx + 1])

        if multi_gpu and not (attn_skip and mlp_skip):
            hidden_states, attention_mask, position_embeddings, cache_position = (
                _move_decoder_state_to_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    cache_position,
                )
            )
        block_input = hidden_states

        if not attn_skip:
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=dynamic_cache,
                cache_position=cache_position,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_output

        if not mlp_skip:
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + decoder_layer.mlp(normed_hidden)

        protected_block = (
            idx < protected_edge_blocks
            or idx >= num_blocks - protected_edge_blocks
        )
        include_importance = not protected_block and not (attn_skip and mlp_skip)
        importance_mask.append(include_importance)
        if multi_gpu:
            similarity = F.cosine_similarity(
                block_input.float(),
                hidden_states.float(),
                dim=-1,
                eps=1e-8,
            )
            importance = 1.0 - similarity.mean()
            if not include_importance:
                importance = importance.new_zeros(())
            sharded_importance.append(importance.to(device))
        else:
            block_inputs.append(block_input)
            block_outputs.append(hidden_states)

    if multi_gpu:
        block_importance = torch.stack(sharded_importance)
    else:
        stacked_inputs = torch.stack(block_inputs).float()
        stacked_outputs = torch.stack(block_outputs).float()
        similarities = F.cosine_similarity(
            stacked_inputs,
            stacked_outputs,
            dim=-1,
            eps=1e-8,
        )
        block_importance = 1.0 - similarities.mean(dim=tuple(range(1, similarities.dim())))
        block_importance = block_importance.masked_fill(
            ~torch.tensor(importance_mask, dtype=torch.bool, device=device),
            0.0,
        )

    past_kv_cache = dynamic_cache.to_legacy_cache()
    if multi_gpu:
        logits = _finish_sharded_decoder_forward(model, text_model, hidden_states)
    else:
        logits = model.lm_head(text_model.norm(hidden_states))
    return logits, past_kv_cache, block_importance


def forward_full_with_block_importance(
    model,
    input_ids: torch.Tensor,
    past_kv_cache: Optional[list] = None,
    protected_edge_blocks: int = 2,
):
    """Full-model decode forward with one importance scalar per block."""
    num_blocks = len(get_text_model(model).layers)
    return forward_draft_divided_multi_with_importance(
        model,
        input_ids,
        [0] * (2 * num_blocks),
        past_kv_cache,
        protected_edge_blocks=protected_edge_blocks,
    )


def forward_verify_divided_multi(
    model, 
    verify_input: torch.Tensor, 
    past_kv_cache=None, 
    claspd=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:
        if len(past_kv_cache) == len(get_text_model(model).layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    text_model = get_text_model(model)
    multi_gpu = model_uses_multiple_cuda_devices(model)
    inputs_embeds = text_model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = _decode_position_ids(
        model,
        full_past_key_values_length,
        seq_length,
        batch_size,
        device,
    )
    cache_position = torch.arange(
        full_past_key_values_length,
        full_past_key_values_length + seq_length,
        dtype=torch.long,
        device=device,
    )
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    if optimization_phase:
        if claspd.is_prefill_stage:
            layer_hidden = hidden_states[:, -64:, :]
        else:
            layer_hidden = hidden_states
        claspd.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers with attention/MLP division
    for idx, decoder_layer in enumerate(text_model.layers):
        if multi_gpu:
            hidden_states, full_attention_mask, position_embeddings, cache_position = (
                _move_decoder_state_to_layer(
                    decoder_layer,
                    hidden_states,
                    full_attention_mask,
                    position_embeddings,
                    cache_position,
                )
            )
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            cache_position=cache_position,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output
        
        # Store hidden states after attention (index 2*idx + 1)
        if optimization_phase:
            if claspd.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            claspd.cached_hidden_states[2*idx+1].append(layer_hidden.detach())

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output
        
        # Store hidden states after MLP (index 2*idx + 2)
        if optimization_phase:
            if claspd.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            claspd.cached_hidden_states[2*idx+2].append(layer_hidden.detach())
    
    past_kv_cache = past_kv_cache.to_legacy_cache()
    if multi_gpu:
        logits = _finish_sharded_decoder_forward(model, text_model, hidden_states)
    else:
        hidden_states = text_model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_tree_divided_multi(
    model,
    tree_input: torch.Tensor,
    parent_indices: List[int],
    depths: List[int],
    past_kv_cache=None,
    claspd=None,
    optimization_phase: bool = False,
):
    """Verify a linearized draft tree in one target-model forward pass."""
    if tree_input.dim() != 2 or tree_input.shape[0] != 1:
        raise ValueError("tree verification currently requires batch size 1")
    if tree_input.shape[1] != len(parent_indices) or len(parent_indices) != len(depths):
        raise ValueError("tree token, parent, and depth lengths must match")

    device = tree_input.device
    batch_size, tree_len = tree_input.shape
    past_length = 0
    if past_kv_cache is not None:
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                past_length = max(past_length, layer_cache[0].shape[2])

    dynamic_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    text_model = get_text_model(model)
    multi_gpu = model_uses_multiple_cuda_devices(model)
    inputs_embeds = text_model.embed_tokens(tree_input)
    position_ids = _tree_position_ids(
        model,
        past_length,
        depths,
        batch_size,
        device,
    )
    cache_position = torch.arange(
        past_length,
        past_length + tree_len,
        dtype=torch.long,
        device=device,
    )
    attention_mask = _make_tree_causal_mask(
        parent_indices,
        inputs_embeds.dtype,
        device,
        past_length,
    )
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    if optimization_phase:
        claspd.cached_hidden_states[0].append(hidden_states.detach())

    for idx, decoder_layer in enumerate(text_model.layers):
        if multi_gpu:
            hidden_states, attention_mask, position_embeddings, cache_position = (
                _move_decoder_state_to_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    cache_position,
                )
            )
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=dynamic_cache,
            cache_position=cache_position,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output
        if optimization_phase:
            claspd.cached_hidden_states[2 * idx + 1].append(hidden_states.detach())

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + decoder_layer.mlp(normed_hidden)
        if optimization_phase:
            claspd.cached_hidden_states[2 * idx + 2].append(hidden_states.detach())

    past_kv_cache = dynamic_cache.to_legacy_cache()
    if multi_gpu:
        logits = _finish_sharded_decoder_forward(model, text_model, hidden_states)
    else:
        logits = model.lm_head(text_model.norm(hidden_states))
    return logits, past_kv_cache
# ------------------------
# DEL Forward functions
# ------------------------

def forward_early_DEL(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    DEL,
) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Check if past_key_values is already DynamicCache or needs conversion
    if past_key_values is None:
         past_key_values = transformers.cache_utils.DynamicCache()
    elif not isinstance(past_key_values, transformers.cache_utils.DynamicCache):
         past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    for idx, decoder_layer in enumerate(model.model.layers[:exit_layer]):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        if idx in DEL.eligible_exit_layers:
            DEL.cached_hidden_states[idx].append(hidden_states)

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    return logits, past_key_values, exit_query_cache


def forward_remainder_DEL(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    DEL,
) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    if past_key_values is not None and past_key_values[0] is not None:
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0
    
        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length
    
    # Check if past_key_values is already DynamicCache or needs conversion
    if past_key_values is None:
         past_key_values = transformers.cache_utils.DynamicCache()
    elif not isinstance(past_key_values, transformers.cache_utils.DynamicCache):
         past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    inputs_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    early_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, num_tokens_to_generate),
        inputs_embeds,
        draft_past_key_values_length,
    )

    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,  # we have no past for the full model
    )
    
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    next_decoder_cache = []
    hidden_states = inputs_embeds
    # TODO simplify
    full_hidden_states: Optional[torch.FloatTensor] = None
    
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer
        
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            
            # Recalculate position embeddings for early exit if needed, or slice existing ones if possible?
            # Since model.model.rotary_emb returns (cos, sin)
            cos, sin = position_embeddings
            early_pos_embeddings = (cos[:, -num_tokens_to_generate:, :], sin[:, -num_tokens_to_generate:, :])

            hidden_states = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                position_embeddings=early_pos_embeddings,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
            if idx in DEL.eligible_exit_layers:
                DEL.cached_hidden_states[idx].append(hidden_states)
        else:
            if full_hidden_states is None and exit_query_cache is not None:
                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
                
            hidden_states = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
            if idx in DEL.eligible_exit_layers:
                DEL.cached_hidden_states[idx].append(hidden_states)

    past_key_values = past_key_values.to_legacy_cache()
    DEL.cached_hidden_states[len(model.model.layers)-1].append(hidden_states)

    return None, past_key_values, exit_query_cache
