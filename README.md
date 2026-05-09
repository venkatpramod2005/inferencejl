# GPU Inference Demo (L4/vGPU)

This repo demonstrates running an open-source LLM with GPU acceleration on an NVIDIA L4.

## Phases

1. Setup Python environment
2. Install PyTorch, Transformers, Accelerate, bitsandbytes
3. Verify GPU access
4. Download open-source model (recommended: Mistral 7B)
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
export MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
# If the model is gated, set your HF token and accept the license on Hugging Face.
export HF_TOKEN=YOUR_TOKEN
python scripts/run_inference.py
```

Optional:

```bash
# Use 4-bit quantization if you want lower VRAM.
export USE_4BIT=1
```

## Monitor GPU

```bash
nvidia-smi dmon -s u -c 10
```

## Observations

- Model used: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Mistral 7B requires HF token)
- GPU model: NVIDIA L4
- VRAM: 22.03 GB total, 2.06 GB peak allocated during inference
- Inference speed: ~73 tokens/sec (256 tokens run)
- GPU utilization: dmon sampled ~89% SM, ~84% MEM
- Notes: TinyLlama is open and runs without an HF token; set HF_TOKEN for Mistral 7B.

