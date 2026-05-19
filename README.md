# GPU Inference Demo (L4/vGPU)

This repo demonstrates running an open-source LLM with GPU acceleration on an NVIDIA L4.

## Phases

1. Setup Python environment
2. Install PyTorch, Transformers, Accelerate, bitsandbytes
3. Verify GPU access
4. Download open-source model (recommended: TinyLlama)
5. Run inference on L4/vGPU
6. Monitor VRAM, inference speed, GPU utilization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 13.0 driver is installed; use cu121 wheels for PyTorch.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Verify GPU

```bash
python scripts/verify_gpu.py
nvidia-smi
```

## Run Inference

```bash
export MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python scripts/run_inference.py
```

## Monitor GPU

```bash
nvidia-smi dmon -s u -c 10
```

## Observations

- Model used: TinyLlama/TinyLlama-1.1B-Chat-v1.0 
- GPU model: NVIDIA L4
- VRAM: 22.03 GB total, 2.06 GB peak allocated during inference
- Inference speed: ~73 tokens/sec (256 tokens run)
- GPU utilization: dmon sampled ~89% SM, ~84% MEM
- Notes: TinyLlama is open and runs without an HF token.

