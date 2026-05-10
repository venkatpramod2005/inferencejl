import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_chatbot.config import ModelConfig
from llm_chatbot.model import ChatModel


def main() -> None:
    config = ModelConfig.from_env()
    prompt = os.getenv(
        "PROMPT",
        "Explain how an L4 GPU accelerates transformer inference in simple terms.",
    )

    print(f"Model: {config.model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        chat_model = ChatModel(config)
    except Exception as exc:
        print("Model download/load failed.")
        print("If this model is gated, set HF_TOKEN and accept the model license.")
        raise exc

    generated, metrics = chat_model.generate([{"role": "user", "content": prompt}])

    print("\n=== Output ===\n")
    print(generated)

    print("\n=== Metrics ===\n")
    print(f"Device: {metrics.device}")
    print(f"Elapsed seconds: {metrics.elapsed_seconds:.3f}")
    print(f"Tokens generated: {metrics.tokens_generated}")
    print(f"Tokens/sec: {metrics.tokens_per_second:.2f}")
    if metrics.peak_vram_gb is not None:
        print(f"Peak VRAM allocated (GB): {metrics.peak_vram_gb:.2f}")


if __name__ == "__main__":
    main()
