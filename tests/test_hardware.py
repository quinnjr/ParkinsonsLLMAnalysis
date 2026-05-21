import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware import HardwareInfo, detect_hardware, has_gpu


def test_hardware_info_str():
    info = HardwareInfo(
        cpu_cores=8, ram_gb=32.0, gpu_available=True,
        gpu_name="RTX 4090", gpu_vram_gb=24.0,
    )
    assert "RTX 4090" in str(info)
    assert "32.0GB" in str(info)


def test_has_gpu_true():
    info = HardwareInfo(4, 16.0, True, "GPU", 8.0)
    assert has_gpu(info) is True


def test_has_gpu_false():
    info = HardwareInfo(4, 16.0, False, None, None)
    assert has_gpu(info) is False


@patch("hardware._detect_nvidia", return_value=(True, "Test GPU", 24.0))
@patch("hardware._read_ram_gb", return_value=64.0)
def test_detect_hardware_mocked(mock_ram, mock_gpu):
    info = detect_hardware()
    assert info.gpu_available is True
    assert info.gpu_vram_gb == 24.0
    assert info.ram_gb == 64.0
