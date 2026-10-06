# profile_modules.py
import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from multimodal import get_text_model, is_qwen_vl_with_mrope


_DEFAULT_PROFILE_SEQ_LENGTHS = list(range(1000, 25001, 1000))


def _profile_position_ids(model, seq_len: int, device):
    """Build the position shape expected by the profiled rotary module."""
    position_ids = torch.tensor([[seq_len]], device=device, dtype=torch.long)
    if is_qwen_vl_with_mrope(model):
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return position_ids


def _resolve_profile_seq_lengths(config, seq_lengths=None):
    """Choose at least two KV-prefix lengths inside the model context.

    Profiling adds one query token after each synthetic cache prefix, so the
    largest valid prefix is ``max_position_embeddings - 1``.  This matters for
    Llama-3 LLaVA-NeXT, whose native context is 8K rather than the 32K context
    used by the Mistral checkpoint.
    """
    using_defaults = seq_lengths is None
    requested = (
        list(_DEFAULT_PROFILE_SEQ_LENGTHS)
        if using_defaults
        else list(seq_lengths)
    )
    if not requested:
        raise ValueError("seq_lengths must contain at least two positive values")

    normalized = []
    for value in requested:
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("seq_lengths must contain integers") from exc
        if resolved < 1:
            raise ValueError("seq_lengths must contain positive values")
        if resolved not in normalized:
            normalized.append(resolved)

    context_window = getattr(config, "max_position_embeddings", None)
    if context_window is not None:
        try:
            max_prefix_length = int(context_window) - 1
        except (TypeError, ValueError) as exc:
            raise ValueError("max_position_embeddings must be an integer") from exc
        if max_prefix_length < 2:
            raise ValueError(
                "Model context is too short to fit two profiling sequence lengths"
            )
        normalized = [
            length for length in normalized if length <= max_prefix_length
        ]

        if using_defaults and len(normalized) < 2:
            midpoint = max(1, max_prefix_length // 2)
            normalized = sorted({midpoint, max_prefix_length})

    if len(normalized) < 2:
        raise ValueError(
            "At least two distinct profiling sequence lengths must fit inside "
            "the model context"
        )
    return normalized


def _resolve_head_dim(config, self_attn, num_heads: int) -> int:
    """Resolve a concrete attention head width across model config variants.

    Some Transformers configs expose ``head_dim`` but leave it as ``None``.
    In that case a nested ``getattr`` fallback does not run, so prefer the
    instantiated attention module and then derive the width from its Q
    projection or the model hidden size.
    """
    candidates = (
        getattr(self_attn, "head_dim", None),
        getattr(config, "head_dim", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = int(candidate)
        if resolved > 0:
            return resolved

    q_projection = getattr(self_attn, "q_proj", None)
    q_output_size = getattr(q_projection, "out_features", None)
    if q_output_size is not None:
        q_output_size = int(q_output_size)
        if q_output_size > 0 and q_output_size % num_heads == 0:
            return q_output_size // num_heads

    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    if hidden_size > 0 and hidden_size % num_heads == 0:
        return hidden_size // num_heads

    raise ValueError(
        "Could not resolve a positive integer attention head dimension from "
        "the attention module, Q projection, or model hidden size."
    )

# ==========================================
# 1. Mask Functions (User Provided)
# ==========================================
def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    return combined_attention_mask

# ==========================================
# 2. Precise Profiling Logic (Manual Sync)
# ==========================================
def profile_model(
    model,
    tokenizer=None,
    seq_lengths: list = None,
    layer_idx: int = 0,
    warmup_steps: int = 5,
    measure_steps: int = 10 
):
    text_model = get_text_model(model)
    config = text_model.config
    num_layers = config.num_hidden_layers
    target_layer = text_model.layers[layer_idx]
    rotary_emb = text_model.rotary_emb
    target_parameter = next(target_layer.parameters())
    device = target_parameter.device
    model_dtype = target_parameter.dtype
    print(f"[Profiling] Profiling model on device: {device}, dtype: {model_dtype}...")
    
    # Dimension Extraction
    num_heads = int(config.num_attention_heads)
    if num_heads < 1:
        raise ValueError("num_attention_heads must be positive")
    configured_kv_heads = getattr(config, "num_key_value_heads", None)
    num_kv_heads = num_heads if configured_kv_heads is None else int(configured_kv_heads)
    if num_kv_heads < 1:
        raise ValueError("num_key_value_heads must be positive")
    head_dim = _resolve_head_dim(config, target_layer.self_attn, num_heads)

    print(f"[Profiling] Detected: heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    requested_seq_lengths = seq_lengths
    seq_lengths = _resolve_profile_seq_lengths(config, seq_lengths)
    if requested_seq_lengths is None and seq_lengths != _DEFAULT_PROFILE_SEQ_LENGTHS:
        print(
            "[Profiling] Capped default sequence lengths to the model's "
            f"{config.max_position_embeddings}-token context."
        )

    attn_times = []
    mlp_times = []

    print("[Profiling] Starting measurement loop...")
    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        
        # 1. Setup Inputs
        hidden_states = torch.randn(1, 1, config.hidden_size, device=device, dtype=model_dtype)
        position_ids = _profile_position_ids(model, seq_len, device)
        cos, sin = rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        
        # 2. Mask
        raw_attention_mask = torch.ones((1, 1 + seq_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_decoder_attention_mask(
            model=model, attention_mask=raw_attention_mask, input_shape=(1, 1),
            inputs_embeds=hidden_states, past_key_values_length=seq_len
        )
        
        legacy_list = []
        for i in range(num_layers):
            if i == layer_idx:
                pk = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=model_dtype)
                pv = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=model_dtype)
                legacy_list.append((pk, pv))
            else:
                 # Minimal dummy for other layers to satisfy DynamicCache format
                 legacy_list.append((
                     torch.empty(1, num_kv_heads, 0, head_dim, device=device, dtype=model_dtype), 
                     torch.empty(1, num_kv_heads, 0, head_dim, device=device, dtype=model_dtype)
                 ))
        
        legacy_snap = tuple(legacy_list)

        # --- Attn Measurement ---
        temp_attn = []
        for i in range(warmup_steps + measure_steps):
            dc = DynamicCache.from_legacy_cache(legacy_snap)
            normed = target_layer.input_layernorm(hidden_states)

            # Balanced model dispatch can place the profiled decoder layer on
            # any visible GPU. Record and synchronize events on that layer's
            # device instead of implicitly measuring CUDA device 0.
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record(torch.cuda.current_stream(device))
                with torch.inference_mode():
                    outputs = target_layer.self_attn(
                        hidden_states=normed,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                        past_key_values=dc,
                        use_cache=True,
                    )
                    _ = hidden_states + outputs[0]
                end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
            
            if i >= warmup_steps:
                temp_attn.append(start.elapsed_time(end))

        # --- MLP Measurement ---
        temp_mlp = []
        for i in range(warmup_steps + measure_steps):
            normed = target_layer.post_attention_layernorm(hidden_states)

            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record(torch.cuda.current_stream(device))
                with torch.inference_mode():
                    out = target_layer.mlp(normed)
                    _ = hidden_states + out
                end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
            
            if i >= warmup_steps:
                temp_mlp.append(start.elapsed_time(end))

        t_attn, t_mlp = np.mean(temp_attn), np.mean(temp_mlp)
        attn_times.append(t_attn)
        mlp_times.append(t_mlp)

    # Linear Regression Fitting
    c_1 = np.mean(mlp_times)
    slope, intercept = np.polyfit(np.array(seq_lengths), np.array(attn_times), 1)
    
    print(f"[Profiling] Done. c_1={c_1:.6f}, c_2={slope:.6e}, c_3={intercept:.6f}")
    
    # [Added for Plotting]
    print("seq_lengths =", seq_lengths)
    print("attn_times =", attn_times)
    print("mlp_times =", mlp_times)
    
    return c_1, slope, intercept
