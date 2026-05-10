import json
import time
from typing import Any

import torch


def collect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        info.update(
            {
                "device_index": device,
                "device_name": torch.cuda.get_device_name(device),
                "total_vram_gb": round(total_bytes / 1024**3, 2),
                "free_vram_gb": round(free_bytes / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    return info


def gpu_info_json() -> str:
    return json.dumps(collect_gpu_info(), indent=2)

