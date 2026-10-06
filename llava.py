"""LLaVA-1.5 model access and multimodal input preparation helpers."""

import base64
from io import BytesIO
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from device_utils import model_input_device, select_device_map
from qwen3_vl import _extract_video_frames_ffmpeg, _resolve_dtype


MediaSource = Union[str, Path]
LLAVA_VIDEO_MAX_FRAMES = 16


def is_llava(model_or_config: Any) -> bool:
    """Return whether an object is a Hugging Face LLaVA model or config."""
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", None) == "llava"


def get_llava_language_model(model: Any) -> Any:
    """Return the Llama decoder embedded in a LLaVA conditional model."""
    if not is_llava(model):
        raise TypeError("Expected a LLaVA model (config.model_type='llava').")

    base_model = getattr(model, "model", None)
    language_model = getattr(base_model, "language_model", None)
    if language_model is None:
        raise AttributeError("LLaVA model does not expose model.language_model.")

    if hasattr(language_model, "layers"):
        return language_model
    nested_model = getattr(language_model, "model", None)
    if nested_model is not None and hasattr(nested_model, "layers"):
        return nested_model
    raise AttributeError("LLaVA language model does not expose decoder layers.")


def build_llava_messages(
    prompt: str,
    *,
    has_image: bool,
    num_images: int = 1,
) -> List[Dict[str, Any]]:
    """Build the LLaVA-1.5 user message, with images before the text."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if num_images < 1:
        raise ValueError("num_images must be positive")

    content: List[Dict[str, Any]] = []
    if has_image:
        content.extend({"type": "image"} for _ in range(num_images))
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _load_rgb_image(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    value = str(image)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        with urlopen(value, timeout=60) as response:
            payload = response.read()
        with Image.open(BytesIO(payload)) as loaded:
            return loaded.convert("RGB")
    if parsed.scheme == "data":
        try:
            payload = base64.b64decode(value.split(",", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid data URI for LLaVA image input.") from exc
        with Image.open(BytesIO(payload)) as loaded:
            return loaded.convert("RGB")

    path = Path(unquote(parsed.path) if parsed.scheme == "file" else value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as loaded:
        return loaded.convert("RGB")


def make_llava_video_contact_sheet(frames: List[Any], *, tile_size: int = 336) -> Any:
    """Arrange sampled video frames chronologically in one square RGB image."""
    from PIL import Image, ImageOps

    if not frames:
        raise ValueError("At least one frame is required for a contact sheet.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    columns = int(math.ceil(math.sqrt(len(frames))))
    rows = int(math.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "black")
    for index, frame in enumerate(frames):
        image = _load_rgb_image(frame)
        tile = ImageOps.fit(image, (tile_size, tile_size))
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(tile, (x, y))
    return sheet


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    dtype = getattr(model, "dtype", None)
    input_device = model_input_device(model)
    if dtype is None:
        return inputs.to(input_device)
    return inputs.to(input_device, dtype)


def prepare_llava_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[Any] = None,
) -> Any:
    """Prepare one text-only or image-and-text LLaVA-1.5 request."""
    if not is_llava(model):
        raise TypeError("prepare_llava_inputs requires a LLaVA model.")

    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("image sequence must not be empty")
        loaded_images = [_load_rgb_image(item) for item in image]
    elif image is not None:
        loaded_images = [_load_rgb_image(image)]
    else:
        loaded_images = []

    messages = build_llava_messages(
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
        processor_kwargs["images"] = loaded_images[0] if len(loaded_images) == 1 else loaded_images
    inputs = processor(**processor_kwargs)
    return _move_inputs_to_model(inputs, model)


def prepare_llava_video_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
    num_contact_sheets: int = 1,
) -> Any:
    """Represent a video as consecutive chronological contact sheets for LLaVA-1.5."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    if num_contact_sheets < 1:
        raise ValueError("num_contact_sheets must be positive")

    sampled_frames = min(num_frames, LLAVA_VIDEO_MAX_FRAMES * num_contact_sheets)
    with TemporaryDirectory(prefix="knapspec_llava_video_") as directory:
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
                f"The {sheet_count} images are consecutive chronological contact sheets sampled uniformly "
                "from one video. Read the images in order. Within each image, read panels from left to "
                "right and top to bottom.\n\n"
                f"{prompt}"
            )
            visual_input = contact_sheets
        return prepare_llava_inputs(model, processor, video_prompt, visual_input)


def load_llava(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    *,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Any:
    """Load a Hugging Face LLaVA-1.5 model and processor."""
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

    processor = AutoProcessor.from_pretrained(model_id, do_pad=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs).eval()
    if not is_llava(model):
        raise TypeError(
            f"'{model_id}' loaded as model_type={getattr(model.config, 'model_type', None)!r}, "
            "not 'llava'."
        )
    return model, processor
