import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_chatbot.gpu import gpu_info_json


def main() -> None:
    print(gpu_info_json())


if __name__ == "__main__":
    main()
