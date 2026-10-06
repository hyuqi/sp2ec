"""LLaVA-NeXT image-model access and multimodal input helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

from device_utils import select_device_map
from llava import (
    LLAVA_VIDEO_MAX_FRAMES,
    _load_rgb_image,
    _move_inputs_to_model,
    make_llava_video_contact_sheet,
)
from qwen3_vl import _extract_video_frames_ffmpeg, _resolve_dtype


MediaSource = Union[str, Path]


def is_llava_next(model_or_config: Any) -> bool:
    """Return whether an object is a Hugging Face LLaVA-NeXT model or config."""
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", None) == "llava_next"


def count_llava_next_visual_tokens(model_or_config: Any, input_ids: List[int]) -> Optional[int]:
    """Count expanded image placeholders, including AnyRes newline positions."""
    if not is_llava_next(model_or_config):
        return None
    config = getattr(model_or_config, "config", model_or_config)
    return input_ids.count(config.image_token_index)


def get_llava_next_language_model(model: Any) -> Any:
    """Return the text decoder embedded in a LLaVA-NeXT conditional model."""
    if not is_llava_next(model):
        raise TypeError(
            "Expected a LLaVA-NeXT model (config.model_type='llava_next')."
        )

    base_model = getattr(model, "model", None)
    language_model = getattr(base_model, "language_model", None)
    if language_model is None:
        raise AttributeError(
            "LLaVA-NeXT model does not expose model.language_model."
        )

    if hasattr(language_model, "layers"):
        return language_model
    nested_model = getattr(language_model, "model", None)
    if nested_model is not None and hasattr(nested_model, "layers"):
        return nested_model
    raise AttributeError(
        "LLaVA-NeXT language model does not expose decoder layers."
    )


def build_llava_next_messages(
    prompt: str,
    *,
    has_image: bool,
    num_images: int = 1,
) -> List[Dict[str, Any]]:
    """Build a LLaVA-NeXT user message, with images before the text."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if num_images < 1:
        raise ValueError("num_images must be positive")

    content: List[Dict[str, Any]] = []
    if has_image:
        content.extend({"type": "image"} for _ in range(num_images))
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_llava_next_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[Any] = None,
    *,
    max_visual_tokens: int = 8192,
) -> Any:
    """Prepare one text-only or image-and-text LLaVA-NeXT request."""
    if not is_llava_next(model):
        raise TypeError("prepare_llava_next_inputs requires a LLaVA-NeXT model.")

    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("image sequence must not be empty")
        loaded_images = [_load_rgb_image(item) for item in image]
    elif image is not None:
        loaded_images = [_load_rgb_image(image)]
    else:
        loaded_images = []

    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    if loaded_images:
        _bound_llava_next_grid(model, processor, len(loaded_images), max_visual_tokens)

    messages = build_llava_next_messages(
        prompt,
        has_image=bool(loaded_images),
        num_images=len(loaded_images) or 1,
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_kwargs: Dict[str, Any] = {
        "text": text,
        "return_tensors": "pt",
    }
    if loaded_images:
        processor_kwargs["images"] = (
            loaded_images[0] if len(loaded_images) == 1 else loaded_images
        )
    inputs = processor(**processor_kwargs)
    _validate_llava_next_prompt_length(model, inputs)
    return _move_inputs_to_model(inputs, model)


def _bound_llava_next_grid(
    model: Any, processor: Any, num_images: int, max_visual_tokens: int
) -> None:
    """Constrain native AnyRes choices without dropping frames or image sheets.

    The checkpoint's processor AND model must use identical grid choices: the
    model uses them again when unpacking image features. Retain every original
    choice whose conservative patch/newline count fits the per-image budget.
    Four 16-frame sheets therefore select the native 1x2/2x1 grids at 8192
    tokens, instead of four square 2x2 grids (~11712 visual tokens).
    """
    if num_images < 1 or max_visual_tokens < 1:
        raise ValueError("num_images and max_visual_tokens must be positive")
    image_processor = getattr(processor, "image_processor", None)
    config = getattr(model, "config", None)
    if image_processor is None or config is None:
        raise ValueError("LLaVA-NeXT requires its native image processor and config")
    original = getattr(image_processor, "_sp2ec_original_image_grid_pinpoints", None)
    if original is None:
        original = getattr(config, "image_grid_pinpoints", None)
        if original is None:
            original = getattr(image_processor, "image_grid_pinpoints", None)
        if not original:
            raise ValueError("LLaVA-NeXT config is missing image_grid_pinpoints")
        original = tuple(tuple(int(value) for value in grid) for grid in original)
        image_processor._sp2ec_original_image_grid_pinpoints = original

    vision_config = getattr(config, "vision_config", None)
    image_size = int(getattr(vision_config, "image_size", 336))
    patch_size = int(getattr(vision_config, "patch_size", 14))
    if image_size < 1 or patch_size < 1 or image_size % patch_size:
        raise ValueError("LLaVA-NeXT requires a positive patch-aligned image size")
    base_tokens = (image_size // patch_size) ** 2
    if getattr(config, "vision_feature_select_strategy", "default") == "full":
        # The supported checkpoint uses 'default'; do not guess a CLS-bearing
        # reshape contract for an unsupported processor configuration.
        raise ValueError("The sample supports LLaVA-NeXT's default vision feature strategy")

    per_image_budget = max_visual_tokens // num_images
    bounded = []
    for height, width in original:
        if height <= 0 or width <= 0 or height % image_size or width % image_size:
            raise ValueError("LLaVA-NeXT AnyRes grids must contain whole image tiles")
        feature_rows, feature_columns = height // patch_size, width // patch_size
        upper_bound = base_tokens + feature_rows * feature_columns + feature_rows
        if upper_bound <= per_image_budget:
            bounded.append([height, width])
    if not bounded:
        raise ValueError(
            f"No native LLaVA-NeXT AnyRes grid fits {num_images} images within "
            f"{max_visual_tokens} visual tokens; refusing to drop input frames."
        )
    image_processor.image_grid_pinpoints = bounded
    config.image_grid_pinpoints = [list(grid) for grid in bounded]


def _validate_llava_next_prompt_length(model: Any, inputs: Any) -> None:
    """Reject an expanded AnyRes prompt that already exceeds text context."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    context_window = getattr(text_config, "max_position_embeddings", None)
    if context_window is None:
        return

    try:
        context_window = int(context_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_position_embeddings must be an integer") from exc
    if context_window < 1:
        raise ValueError("max_position_embeddings must be positive")

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if input_ids is None:
        return
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) > 0:
        prompt_length = int(shape[-1])
    else:
        try:
            first = input_ids[0]
            prompt_length = len(first) if hasattr(first, "__len__") else len(input_ids)
        except (IndexError, TypeError):
            return

    if prompt_length > context_window:
        raise ValueError(
            f"Expanded LLaVA-NeXT prompt has {prompt_length} tokens, exceeding "
            f"the model's {context_window}-token context. Reduce the number of "
            "images/contact sheets or shorten the prompt."
        )


def prepare_llava_next_video_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
    num_contact_sheets: int = 1,
) -> Any:
    """Represent a video as chronological contact sheets for LLaVA-NeXT."""
    if not is_llava_next(model):
        raise TypeError(
            "prepare_llava_next_video_inputs requires a LLaVA-NeXT model."
        )
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    if num_contact_sheets < 1:
        raise ValueError("num_contact_sheets must be positive")

    sampled_frames = min(num_frames, LLAVA_VIDEO_MAX_FRAMES * num_contact_sheets)
    with TemporaryDirectory(prefix="knapspec_llava_next_video_") as directory:
        frames, _ = _extract_video_frames_ffmpeg(
            video,
            num_frames=sampled_frames,
            output_dir=Path(directory),
        )
        sheet_count = min(num_contact_sheets, len(frames))
        base_size, remainder = divmod(len(frames), sheet_count)
        contact_sheets = []
        cursor = 0
        for sheet_idx in range(sheet_count):
            group_size = base_size + int(sheet_idx < remainder)
            group = frames[cursor : cursor + group_size]
            contact_sheets.append(make_llava_video_contact_sheet(group))
            cursor += group_size

        if sheet_count == 1:
            video_prompt = (
                "The image is a chronological contact sheet sampled uniformly from a video. "
                "Read the panels from left to right and top to bottom.\n\n"
                f"{prompt}"
            )
            visual_input: Any = contact_sheets[0]
        else:
            video_prompt = (
                f"The {sheet_count} images are consecutive chronological contact sheets "
                "sampled uniformly from one video. Read the images in order. Within each "
                "image, read panels from left to right and top to bottom.\n\n"
                f"{prompt}"
            )
            visual_input = contact_sheets
        return prepare_llava_next_inputs(
            model,
            processor,
            video_prompt,
            visual_input,
            max_visual_tokens=max_visual_tokens,
        )


def load_llava_next(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    *,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    tree_verification: bool = True,
) -> Tuple[Any, Any]:
    """Load a Hugging Face LLaVA-NeXT model and processor."""
    import torch
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    if tree_verification and attn_implementation == "flash_attention_2":
        raise ValueError(
            "LLaVA-NeXT tree verification requires eager or SDPA attention; "
            "FlashAttention 2 does not support its custom 4-D tree mask."
        )
    if device_map == "auto" and torch.cuda.is_available():
        device_map = select_device_map("cuda")

    model_kwargs: Dict[str, Any] = {
        "dtype": _resolve_dtype(dtype),
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(
        model_id,
        do_pad=True,
        use_fast=False,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    ).eval()
    if not is_llava_next(model):
        raise TypeError(
            f"'{model_id}' loaded as model_type="
            f"{getattr(model.config, 'model_type', None)!r}, not 'llava_next'."
        )
    return model, processor


