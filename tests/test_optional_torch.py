import subprocess
import sys
import pytest
import numpy as np

from adamixture.compat import Device, is_torch_available, is_torch_loaded, require_torch


def test_compat_helpers(monkeypatch):
    """Test compat layer behaves correctly when torch is blocked."""
    monkeypatch.setitem(sys.modules, "torch", None)

    assert not is_torch_loaded()
    assert not is_torch_available()

    with pytest.raises(ImportError) as exc_info:
        require_torch()
    assert "pip install 'adamixture[gpu]'" in str(exc_info.value)

    dev_cpu = Device("cpu")
    assert dev_cpu.type == "cpu"
    assert str(dev_cpu) == "cpu"

    dev_cuda = Device("cuda:0")
    assert dev_cuda.type == "cuda"
    assert dev_cuda.index == 0
    with pytest.raises(ImportError) as exc_info_cuda:
        _ = dev_cuda.torch_device
    assert "pip install 'adamixture[gpu]'" in str(exc_info_cuda.value)


def test_import_and_cpu_execution_in_isolated_process():
    """Verify in an isolated Python process that adamixture can be imported and executed on CPU without torch."""
    code = """
import sys
sys.modules["torch"] = None

import adamixture
import numpy as np

rng = np.random.default_rng(42)
G = rng.choice([0, 1, 2], size=(200, 30), p=[0.6, 0.3, 0.1]).astype(np.uint8)

res = adamixture.run(data=G, k=2, device="cpu", max_iter=3, verbose=False)
assert isinstance(res, tuple)
assert len(res) == 2
assert res[0].shape == (30, 2)
assert res[1].shape == (200, 2)
assert "torch" not in sys.modules or sys.modules["torch"] is None
print("PASS_CPU_NO_TORCH")
"""
    cmd = [sys.executable, "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "PASS_CPU_NO_TORCH" in result.stdout


def test_gpu_request_raises_informative_error_in_isolated_process():
    """Verify that requesting CUDA or MPS without torch installed raises the expected ImportError."""
    code = """
import sys
sys.modules["torch"] = None

import adamixture
import numpy as np

rng = np.random.default_rng(42)
G = rng.choice([0, 1, 2], size=(100, 20)).astype(np.uint8)

for dev in ("cuda", "mps"):
    try:
        adamixture.run(data=G, k=2, device=dev, max_iter=2, verbose=False)
        sys.exit(2)
    except ImportError as e:
        assert "adamixture[gpu]" in str(e)
print("PASS_GPU_ERROR_CHECK")
"""
    cmd = [sys.executable, "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "PASS_GPU_ERROR_CHECK" in result.stdout
