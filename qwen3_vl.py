"""Qwen3-VL native visual prefill and model-loading helpers.

The shared decoder utilities handle subsequent text decoding and multimodal
RoPE offsets; native prefill retains the checkpoint's DeepStack visual path.
"""

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

from device_utils import model_input_device, select_device_map

MediaSource = Union[str, Path]



def is_qwen3_vl(model_or_config: Any) -> bool:
    """Return whether an object is a Qwen3-VL model or config."""
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", None) == "qwen3_vl"


def get_qwen3_vl_language_model(model: Any) -> Any:
    """Return Qwen3-VL's text decoder without modifying the model object."""
    if not is_qwen3_vl(model):
        raise TypeError("Expected a Qwen3-VL model (config.model_type='qwen3_vl').")

    base_model = getattr(model, "model", None)
    language_model = getattr(base_model, "language_model", None)
    if language_model is None:
        raise AttributeError("Qwen3-VL model does not expose model.language_model.")
    return language_model


def get_text_model(model: Any) -> Any:
    """Return the decoder model for either a text-only or Qwen3-VL model."""
    if is_qwen3_vl(model):
        return get_qwen3_vl_language_model(model)
    return model.model


def normalize_media_source(source: MediaSource) -> str:
    """Convert a local media path to a file URI and preserve remote URLs."""
    value = str(source)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file", "data"}:
        return value

    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {path}")
    return path.resolve().as_uri()


def _normalize_qwen_vl_utils_source(source: MediaSource) -> str:
    """Keep local file URIs directly openable by qwen-vl-utils and Pillow."""
    normalized = normalize_media_source(source)
    if normalized.startswith("file://"):
        # qwen-vl-utils removes the prefix with image[7:] but does not URL-decode
        # paths. PBS array scratch directories contain brackets, which as_uri()
        # otherwise turns into a nonexistent literal "%5B...%5D" path.
        return unquote(normalized)
    return normalized


def build_qwen3_vl_messages(prompt: str, image: MediaSource) -> List[Dict[str, Any]]:
    """Build one image-and-text user message for the Qwen3-VL processor."""
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


def build_qwen3_vl_text_messages(prompt: str) -> List[Dict[str, Any]]:
    """Build one text-only user message for the Qwen3-VL processor."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


def build_qwen3_vl_video_messages(
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
) -> List[Dict[str, Any]]:
    """Build one uniformly sampled video-and-text Qwen3-VL message."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if num_frames < 4:
        raise ValueError("num_frames must be at least 4")
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": _normalize_qwen_vl_utils_source(video),
                    "nframes": num_frames,
                    "total_pixels": max_visual_tokens * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_qwen3_vl_video_frame_messages(
    prompt: str,
    frames: List[MediaSource],
    *,
    sample_fps: float,
    max_visual_tokens: int = 8192,
) -> List[Dict[str, Any]]:
    """Build a video message from frames decoded outside qwen-vl-utils."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(frames) < 4:
        raise ValueError("at least 4 video frames are required")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [_normalize_qwen_vl_utils_source(frame) for frame in frames],
                    "sample_fps": sample_fps,
                    "total_pixels": max_visual_tokens * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _local_media_path(source: MediaSource) -> Path:
    value = str(source)
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError("The ffmpeg video reader requires a local file path.")
    else:
        path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {path}")
    return path.resolve()


def _ffprobe_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found. Load or install FFmpeg before using "
            "KNAPSPEC_VIDEO_READER=ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}: {exc.stderr.strip()}") from exc

    metadata = json.loads(result.stdout)
    candidates = [metadata.get("format", {}).get("duration")]
    candidates.extend(stream.get("duration") for stream in metadata.get("streams", []))
    durations = [float(value) for value in candidates if value not in {None, "N/A"}]
    if not durations or max(durations) <= 0:
        raise RuntimeError(f"Could not determine a positive duration for {video_path}")
    return max(durations)


def _extract_video_frames_ffmpeg(
    video: MediaSource,
    *,
    num_frames: int,
    output_dir: Path,
) -> Tuple[List[Path], float]:
    """Uniformly sample frames with a single-threaded FFmpeg process."""
    video_path = _local_media_path(video)
    duration = _ffprobe_duration(video_path)
    output_pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-filter_threads",
        "1",
        "-vf",
        f"fps={num_frames}/{duration:.9f}:round=up",
        "-frames:v",
        str(num_frames),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Load or install FFmpeg before using "
            "KNAPSPEC_VIDEO_READER=ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {exc.stderr.strip()}") from exc

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    while len(frames) < 4:
        frames.append(frames[-1])
    return frames, len(frames) / duration


def _resolve_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"

    import torch

    supported = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return supported[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def load_qwen3_vl(
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    *,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load a Qwen3-VL image-text model and its processor."""
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

    if not is_qwen3_vl(model):
        raise TypeError(
            f"'{model_id}' loaded as model_type={getattr(model.config, 'model_type', None)!r}, "
            "not 'qwen3_vl'."
        )
    return model, processor


def prepare_qwen3_vl_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[MediaSource] = None,
    *,
    max_visual_tokens: int = 8192,
) -> Any:
    """Prepare an image-and-text or text-only Qwen3-VL request."""
    messages = (
        build_qwen3_vl_text_messages(prompt)
        if image is None
        else build_qwen3_vl_messages(prompt, image)
    )
    if max_visual_tokens <= 0:
        raise ValueError("max_visual_tokens must be positive")
    image_kwargs = (
        {"max_pixels": max_visual_tokens * 32 * 32}
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


def prepare_qwen3_vl_video_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    video: MediaSource,
    *,
    num_frames: int = 64,
    max_visual_tokens: int = 8192,
) -> Any:
    """Decode and prepare a video with Qwen or bounded FFmpeg frame sampling."""
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
        messages = build_qwen3_vl_video_frame_messages(
            prompt,
            frames,
            sample_fps=sample_fps,
            max_visual_tokens=max_visual_tokens,
        )
    elif video_reader == "qwen":
        messages = build_qwen3_vl_video_messages(
            prompt,
            video,
            num_frames=num_frames,
            max_visual_tokens=max_visual_tokens,
        )
    else:
        raise ValueError("KNAPSPEC_VIDEO_READER must be either 'qwen' or 'ffmpeg'")

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            video_metadata = None

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs.to(model_input_device(model))
    finally:
        if frame_directory is not None:
            frame_directory.cleanup()
