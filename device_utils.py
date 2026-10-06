"""Device-placement helpers for optional single-process model parallelism."""

from typing import Any, Optional


def select_device_map(device: str, cuda_device_count: Optional[int] = None):
    """Preserve the old loader on one GPU and balance weights on multiple GPUs."""
    if device != "cuda":
        return None
    if cuda_device_count is None:
        import torch

        cuda_device_count = torch.cuda.device_count()
    return "balanced" if cuda_device_count > 1 else "auto"


def model_uses_multiple_cuda_devices(model: Any) -> bool:
    """Return whether a dispatched model spans more than one CUDA device."""
    cached = getattr(model, "_knapspec_multi_cuda", None)
    if cached is not None:
        return bool(cached)

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        cuda_devices = set()
        for device in device_map.values():
            normalized = _normalize_cuda_device(device)
            if normalized is not None:
                cuda_devices.add(normalized)
        result = len(cuda_devices) > 1
        setattr(model, "_knapspec_multi_cuda", result)
        return result

    cuda_devices = {
        str(parameter.device)
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }
    result = len(cuda_devices) > 1
    setattr(model, "_knapspec_multi_cuda", result)
    return result


def module_device(module: Any, fallback: Any):
    """Find the execution device for a dispatched module."""
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return f"cuda:{execution_device}" if isinstance(execution_device, int) else execution_device

    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in module.buffers():
        if buffer.device.type != "meta":
            return buffer.device
    return fallback


def model_input_device(model: Any):
    """Return the device that should receive tokenized model inputs."""
    fallback = getattr(model, "device", "cpu")
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        if embeddings is not None:
            return module_device(embeddings, fallback)
    return fallback


def move_to_device(value: Any, device: Any):
    """Move nested tensor values while preserving their container types."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def _normalize_cuda_device(device: Any) -> Optional[str]:
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, str):
        normalized = device.lower()
        if normalized == "cuda" or normalized.startswith("cuda:"):
            return normalized
        return None
    try:
        import torch

        normalized = torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return None
    if normalized.type != "cuda":
        return None
    return str(normalized)
