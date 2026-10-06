"""Qwen2.5-VL loading and native multimodal input preparation.

Qwen2.5-VL and Qwen3-VL expose similar text decoders, but their visual
preprocessing contracts are not interchangeable.  Qwen2.5-VL uses 14-pixel
vision patches (28 pixels per merged visual token) and consumes sampled video
FPS directly instead of Qwen3-VL's video-metadata objects.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

from device_utils import model_input_device, select_device_map
from qwen3_vl import (
    MediaSource,
    _extract_video_frames_ffmpeg,
    _normalize_qwen_vl_utils_source,
    _resolve_dtype,
    normalize_media_source,
)


QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN2_5_VL_IMAGE_PATCH_SIZE = 14
QWEN2_5_VL_SPATIAL_MERGE_SIZE = 2
QWEN2_5_VL_PIXELS_PER_TOKEN = (
    QWEN2_5_VL_IMAGE_PATCH_SIZE * QWEN2_5_VL_SPATIAL_MERGE_SIZE
) ** 2



def is_qwen2_5_vl(model_or_config: Any) -> bool:
    """Return whether an object is a Qwen2.5-VL model or config."""
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", None) == "qwen2_5_vl"


def get_qwen2_5_vl_language_model(model: Any) -> Any:
    """Return Qwen2.5-VL's text decoder without modifying the model."""
    if not is_qwen2_5_vl(model):
        raise TypeError(
            "Expected a Qwen2.5-VL model "
            "(config.model_type='qwen2_5_vl')."
        )

    base_model = getattr(model, "model", None)
    language_model = getattr(base_model, "language_model", None)
    if language_model is None:
        raise AttributeError(
            "Qwen2.5-VL model does not expose model.language_model."
        )
    return language_model


def get_text_model(model: Any) -> Any:
    """Return the decoder for either a text-only or Qwen2.5-VL model."""
    if is_qwen2_5_vl(model):
        return get_qwen2_5_vl_language_model(model)
    return model.model


def build_qwen2_5_vl_messages(
    prompt: str,
    image: MediaSource,
) -> List[Dict[str, Any]]:
    """Build one image-and-text user message."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": normalize_media_source(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_qwen2_5_vl_text_messages(prompt: str) -> List[Dict[str, Any]]:
    """Build one text-only user message."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


def _video_total_pixels(max_visual_tokens: int) -> int:
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    return max_visual_tokens * QWEN2_5_VL_PIXELS_PER_TOKEN


def build_qwen2_5_vl_video_messages(
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
) -> List[Dict[str, Any]]:
    """Build one uniformly sampled native-video message."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if num_frames < 4:
        raise ValueError("num_frames must be at least 4")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": _normalize_qwen_vl_utils_source(video),
                    "nframes": num_frames,
                    "total_pixels": _video_total_pixels(max_visual_tokens),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_qwen2_5_vl_video_frame_messages(
    prompt: str,
    frames: List[MediaSource],
    *,
    sample_fps: float,
    max_visual_tokens: int = 8192,
) -> List[Dict[str, Any]]:
    """Build a native-video message from externally decoded frames."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(frames) < 4:
        raise ValueError("at least 4 video frames are required")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [
                        _normalize_qwen_vl_utils_source(frame)
                        for frame in frames
                    ],
                    "sample_fps": sample_fps,
                    "total_pixels": _video_total_pixels(max_visual_tokens),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_qwen2_5_vl(
    model_id: str = QWEN2_5_VL_MODEL_ID,
    *,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL and its native processor."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device_map == "auto" and torch.cuda.is_available():
        device_map = select_device_map("cuda")

    model_kwargs: Dict[str, Any] = {
        "dtype": _resolve_dtype(dtype),
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs,
    ).eval()
    if not is_qwen2_5_vl(model):
        raise TypeError(
            f"'{model_id}' loaded as model_type="
            f"{getattr(model.config, 'model_type', None)!r}, not 'qwen2_5_vl'."
        )
    return model, processor


def prepare_qwen2_5_vl_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[MediaSource] = None,
    *,
    max_visual_tokens: int = 8192,
) -> Any:
    """Prepare one image/text or text-only native Qwen2.5-VL request."""
    messages = (
        build_qwen2_5_vl_text_messages(prompt)
        if image is None
        else build_qwen2_5_vl_messages(prompt, image)
    )
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    image_kwargs = (
        {"max_pixels": max_visual_tokens * QWEN2_5_VL_PIXELS_PER_TOKEN}
        if image is not None else {}
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **image_kwargs,
    )
    return inputs.to(model_input_device(model))


def prepare_qwen2_5_vl_video_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
) -> Any:
    """Decode and prepare one video using Qwen2.5-VL's native contract."""
    from qwen_vl_utils import process_vision_info

    if num_frames < 4:
        raise ValueError("num_frames must be at least 4")
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")

    video_reader = os.environ.get("KNAPSPEC_VIDEO_READER", "qwen").lower()
    frame_directory = None
    if video_reader == "ffmpeg":
        frame_directory = TemporaryDirectory(prefix="knapspec_video_")
        frames, sample_fps = _extract_video_frames_ffmpeg(
            video,
            num_frames=num_frames,
            output_dir=Path(frame_directory.name),
        )
        messages = build_qwen2_5_vl_video_frame_messages(
            prompt,
            frames,
            sample_fps=sample_fps,
            max_visual_tokens=max_visual_tokens,
        )
    elif video_reader == "qwen":
        messages = build_qwen2_5_vl_video_messages(
            prompt,
            video,
            num_frames=num_frames,
            max_visual_tokens=max_visual_tokens,
        )
    else:
        raise ValueError(
            "KNAPSPEC_VIDEO_READER must be either 'qwen' or 'ffmpeg'"
        )

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=QWEN2_5_VL_IMAGE_PATCH_SIZE,
            return_video_kwargs=True,
        )
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs.to(model_input_device(model))
    finally:
        if frame_directory is not None:
            frame_directory.cleanup()
