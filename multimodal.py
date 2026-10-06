"""Dispatch helpers shared by supported vision-language model families."""

from typing import Any, Optional

from llava import (
    get_llava_language_model,
    is_llava,
    prepare_llava_inputs,
    prepare_llava_video_inputs,
)
from llava_next import (
    get_llava_next_language_model,
    is_llava_next,
    prepare_llava_next_inputs,
    prepare_llava_next_video_inputs as prepare_llava_next_contact_sheet_video_inputs,
)
from qwen2_5_vl import (
    get_qwen2_5_vl_language_model,
    is_qwen2_5_vl,
    prepare_qwen2_5_vl_inputs,
    prepare_qwen2_5_vl_video_inputs,
)
from qwen3_vl import (
    get_qwen3_vl_language_model,
    is_qwen3_vl,
    prepare_qwen3_vl_inputs,
    prepare_qwen3_vl_video_inputs,
)


def is_supported_multimodal(model_or_config: Any) -> bool:
    return (
        is_qwen2_5_vl(model_or_config)
        or is_qwen3_vl(model_or_config)
        or is_llava_next(model_or_config)
        or is_llava(model_or_config)
    )


def get_text_model(model: Any) -> Any:
    """Return the decoder for a supported text or vision-language model."""
    if is_qwen2_5_vl(model):
        return get_qwen2_5_vl_language_model(model)
    if is_qwen3_vl(model):
        return get_qwen3_vl_language_model(model)
    if is_llava_next(model):
        return get_llava_next_language_model(model)
    if is_llava(model):
        return get_llava_language_model(model)
    return model.model


def is_qwen_vl_with_mrope(model_or_config: Any) -> bool:
    """Return whether a Qwen vision-language model uses 3-D mRoPE."""
    return is_qwen2_5_vl(model_or_config) or is_qwen3_vl(model_or_config)


def get_qwen_vl_rope_deltas(model: Any) -> Any:
    """Return the multimodal position offset cached by native prefill."""
    if not is_qwen_vl_with_mrope(model):
        raise TypeError("Model is not a supported Qwen-VL mRoPE model.")
    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "rope_deltas"):
        raise AttributeError("Qwen-VL model does not expose model.rope_deltas.")
    return base_model.rope_deltas


def _prepare_multimodal_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: Optional[Any] = None,
    video: Optional[Any] = None,
    video_num_frames: int = 64,
    video_max_visual_tokens: int = 8192,
    llava_video_contact_sheets: int = 1,
) -> Any:
    """Prepare a native multimodal prefill for a supported model family."""
    if not is_supported_multimodal(model):
        raise TypeError("Model is not a supported vision-language model.")
    if processor is None:
        raise ValueError("A multimodal model requires Env.processor.")
    if image is not None and video is not None:
        raise ValueError("Multimodal generation accepts at most one image or video.")

    if is_qwen2_5_vl(model):
        if video is not None:
            return prepare_qwen2_5_vl_video_inputs(
                model,
                processor,
                prompt,
                video,
                num_frames=video_num_frames,
                max_visual_tokens=video_max_visual_tokens,
            )
        return prepare_qwen2_5_vl_inputs(
            model, processor, prompt, image,
            max_visual_tokens=video_max_visual_tokens,
        )

    if is_qwen3_vl(model):
        if video is not None:
            return prepare_qwen3_vl_video_inputs(
                model,
                processor,
                prompt,
                video,
                num_frames=video_num_frames,
                max_visual_tokens=video_max_visual_tokens,
            )
        return prepare_qwen3_vl_inputs(
            model, processor, prompt, image,
            max_visual_tokens=video_max_visual_tokens,
        )

    if is_llava_next(model):
        if video is not None:
            return prepare_llava_next_contact_sheet_video_inputs(
                model,
                processor,
                prompt,
                video,
                num_frames=video_num_frames,
                max_visual_tokens=video_max_visual_tokens,
                num_contact_sheets=llava_video_contact_sheets,
            )
        return prepare_llava_next_inputs(
            model, processor, prompt, image,
            max_visual_tokens=video_max_visual_tokens,
        )

    if video is not None:
        return prepare_llava_video_inputs(
            model,
            processor,
            prompt,
            video,
            num_frames=video_num_frames,
            max_visual_tokens=video_max_visual_tokens,
            num_contact_sheets=llava_video_contact_sheets,
        )
    return prepare_llava_inputs(model, processor, prompt, image)


def count_visual_tokens(model: Any, inputs: Any) -> int:
    """Count expanded image/video placeholders in this batch-size-one request."""
    config = getattr(model, "config", model)
    placeholder_ids = {
        int(value)
        for value in (
            getattr(config, "image_token_id", None),
            getattr(config, "video_token_id", None),
            getattr(config, "image_token_index", None),
        )
        if value is not None
    }
    if not placeholder_ids:
        return 0
    token_ids = inputs["input_ids"][0]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return sum(int(token) in placeholder_ids for token in token_ids)


def prepare_multimodal_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: Optional[Any] = None,
    video: Optional[Any] = None,
    video_num_frames: int = 64,
    video_max_visual_tokens: int = 8192,
    llava_video_contact_sheets: int = 1,
) -> Any:
    """Prepare inputs and enforce the visual-token cap after native expansion.

    Qwen preprocessing bounds pixels; LLaVA-NeXT bounds its native AnyRes grid.
    This final check prevents version-dependent processor behavior from silently
    producing a different workload. It runs before the timed model prefill.
    """
    if video_max_visual_tokens <= 0:
        raise ValueError("video_max_visual_tokens must be positive")
    inputs = _prepare_multimodal_inputs(
        model, processor, prompt,
        image=image, video=video,
        video_num_frames=video_num_frames,
        video_max_visual_tokens=video_max_visual_tokens,
        llava_video_contact_sheets=llava_video_contact_sheets,
    )
    visual_tokens = count_visual_tokens(model, inputs)
    if visual_tokens > video_max_visual_tokens:
        raise ValueError(
            f"Native processor produced {visual_tokens} visual tokens, exceeding "
            f"the fixed {video_max_visual_tokens}-token budget. Check the pinned "
            "processor/library versions; input frames will not be silently dropped."
        )
    return inputs
