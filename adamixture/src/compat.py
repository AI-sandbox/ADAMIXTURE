import importlib.util
import sys
from typing import Any

_TORCH_INSTALL_MESSAGE = (
    "GPU acceleration (CUDA / Apple Metal MPS) requires optional dependency 'torch'. "
    "Install it with: pip install 'adamixture[gpu]'"
)

_torch = None


def require_torch():
    """
    Lazily import PyTorch when GPU/MPS acceleration is requested.
    Raises an informative ImportError if PyTorch is not installed.
    """
    global _torch
    if "torch" in sys.modules and sys.modules["torch"] is None:
        raise ImportError(_TORCH_INSTALL_MESSAGE)
    if _torch is None:
        try:
            import torch
            if torch is None:
                raise ImportError(_TORCH_INSTALL_MESSAGE)
            _torch = torch
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(_TORCH_INSTALL_MESSAGE) from exc
    return _torch


def is_torch_available() -> bool:
    """Check if PyTorch is installed and available without eagerly importing it."""
    global _torch
    if "torch" in sys.modules and sys.modules["torch"] is None:
        return False
    if _torch is not None:
        return True
    if sys.modules.get("torch") is not None:
        _torch = sys.modules["torch"]
        return True
    try:
        spec = importlib.util.find_spec("torch")
        return spec is not None
    except (ImportError, AttributeError, ValueError):
        return False


def is_torch_loaded() -> bool:
    """Check if PyTorch has already been imported into sys.modules."""
    return sys.modules.get("torch") is not None


class Device:
    """
    Lightweight device descriptor compatible with CPU, CUDA, and MPS.
    Avoids importing PyTorch when running on CPU.
    """

    def __init__(self, device: str | Any = "cpu"):
        if hasattr(device, "type"):
            self.type = str(device.type).lower()
            self.index = getattr(device, "index", None)
            self._torch_device = device if is_torch_available() else None
        else:
            d_str = str(device).lower().strip()
            if d_str == "gpu":
                d_str = "cuda"
            if ":" in d_str:
                parts = d_str.split(":", 1)
                self.type = parts[0]
                self.index = int(parts[1]) if parts[1].isdigit() else None
            else:
                self.type = d_str
                self.index = None
            self._torch_device = None

    @property
    def torch_device(self):
        """Convert to torch.device if needed."""
        if self._torch_device is None:
            torch = require_torch()
            if self.index is not None:
                self._torch_device = torch.device(f"{self.type}:{self.index}")
            else:
                self._torch_device = torch.device(self.type)
        return self._torch_device

    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self) -> str:
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            other_lower = other.lower()
            return str(self) == other_lower or self.type == other_lower
        if hasattr(other, "type"):
            other_type = str(other.type).lower()
            other_index = getattr(other, "index", None)
            return self.type == other_type and self.index == other_index
        return False
