import json
import time

import torch


def main() -> None:
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if info["cuda_available"]:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device_index": device,
                "device_name": torch.cuda.get_device_name(device),
                "total_vram_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
