# L4 LLM Chatbot

This repository runs a larger open-source chat model on an NVIDIA L4/vGPU and serves it through a Streamlit chatbot. The project is intentionally modular so Codex or other autonomous coding agents can extend the inference, UI, and configuration layers independently.

## Project Structure

```text
app/streamlit_app.py        Streamlit chat interface
llm_chatbot/config.py       Environment-based runtime configuration
llm_chatbot/gpu.py          CUDA and L4/vGPU validation helpers
llm_chatbot/model.py        Model loading, chat formatting, generation, metrics
scripts/verify_gpu.py       Command-line GPU validation
scripts/run_inference.py    Command-line prompt inference and benchmark output
requirements.txt            Python dependencies except the CUDA-specific torch wheel
```

## Recommended Models

| Model | Notes |
| --- | --- |
| `mistralai/Mistral-7B-Instruct-v0.3` | Good default for L4 testing; may require Hugging Face access approval. |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Gated; requires an accepted license and `HF_TOKEN`. |
| `google/gemma-7b-it` | Gated; requires an accepted license and `HF_TOKEN`. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Lightweight smoke test when you do not want to download a 7B/8B model. |

On a 24 GB NVIDIA L4, start with `USE_4BIT=1` for 7B/8B models. Full fp16 may fit for some models but leaves less room for long contexts and concurrent Streamlit sessions.

## Fresh Clone Setup

```bash
git clone https://github.com/venkatpramod2005/inferencejl.git
cd inferencejl
git checkout main

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Use the CUDA wheel that matches the PyTorch recommendation for your driver.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On an L4/vGPU instance, keep the repo under `/home` if your provider warns that data outside `/home` is temporary:

```bash
cd /home
git clone https://github.com/venkatpramod2005/inferencejl.git
cd inferencejl
git checkout main
scripts/setup_l4.sh
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` or export the values directly in your shell.

```bash
export MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=your_hugging_face_token
export USE_4BIT=1
export MAX_NEW_TOKENS=256
export TEMPERATURE=0.7
export TOP_P=0.9
export REQUIRE_CUDA=1
export HF_HOME=/home/.cache/huggingface
```

`HF_TOKEN` is required for gated models. Create it in Hugging Face settings and accept the model license on the model page before first download.

For CPU-only local smoke tests, use a small model and disable GPU-only options:

```bash
export MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export USE_4BIT=0
export REQUIRE_CUDA=0
```

## Validate the L4/vGPU Instance

```bash
python scripts/verify_gpu.py
nvidia-smi
```

Expected result:

- `cuda_available` is `true`
- `device_name` shows an NVIDIA L4 or assigned vGPU
- free VRAM is high enough before loading the model
- `nvidia-smi` reports a healthy driver/runtime state

## Run GPU Inference

```bash
python scripts/run_inference.py
```

Optional prompt override:

```bash
export PROMPT="Give me a short deployment checklist for an L4 inference app."
python scripts/run_inference.py
```

The script prints generated text plus:

- device used
- elapsed seconds
- generated tokens
- tokens/sec
- peak allocated VRAM

## Launch the Streamlit Chatbot

```bash
streamlit run app/streamlit_app.py
```

For a persistent L4 instance session, prefer the repo script:

```bash
scripts/run_streamlit_background.sh
tail -f streamlit.log
```

Open the local Streamlit URL in a browser. The sidebar lets you adjust the model ID, 4-bit quantization, token limit, temperature, and top-p. Chat history is stored in the Streamlit session.

This app is designed to run on a machine with an NVIDIA GPU. Streamlit Community Cloud does not provide your L4/vGPU instance, so deploying this repository directly to `streamlit.app` runs on Streamlit's infrastructure instead of your GPU box. Keep `REQUIRE_CUDA=1` for the real deployment so the app fails fast if CUDA is not available, instead of downloading a 7B model onto the wrong machine.

For a remote L4 instance, keep the project under a persistent directory such as `/home/inferencejl`, then start it as a background service:

```bash
bash scripts/start_chatbot.sh
tail -f logs/streamlit.log
```

Stop it cleanly with:

```bash
bash scripts/stop_chatbot.sh
```

If your provider says data outside `/home` is lost after stop/resume, do not keep the repo, `.venv`, `.env`, or model cache under `/root`. The model cache should live under `/home/.cache/huggingface` or another persistent path.

If port `8501` is not public, use an SSH tunnel from your laptop:

```bash
ssh -L 8501:127.0.0.1:8501 root@YOUR_INSTANCE_IP
```

Then open `http://127.0.0.1:8501`.

## Pause, Resume, and Move Instances

If your provider preserves `/home`, keep this project at `/home/inferencejl` and keep `HF_HOME=/home/.cache/huggingface`. After pausing and resuming the instance, the app process will be stopped, but the repo, virtual environment, and downloaded model cache should remain. Restart it with:

```bash
cd /home/inferencejl
scripts/run_streamlit_background.sh
```

If the instance is unavailable, the chatbot cannot run because inference depends on the L4 GPU. When the instance comes back, the model reloads from the preserved Hugging Face cache instead of downloading again.

To move to a different L4 instance, clone the repo under `/home`, copy or reuse the `.env`, run `scripts/setup_l4.sh`, then start with `scripts/run_streamlit_background.sh`. If you can copy `/home/.cache/huggingface` from the old instance, the new instance can avoid downloading model weights again.

## Monitor GPU During Chat

Run this in another terminal while generating responses:

```bash
nvidia-smi dmon -s pucvmet -c 20
```

Useful checks:

- GPU memory increases when the model loads
- SM utilization increases during generation
- command-line tokens/sec is stable across repeated prompts
- Streamlit responses continue without CUDA out-of-memory errors

## Development Workflow

```bash
git checkout main
git pull
git checkout -b codex-l4-streamlit-chatbot
# make changes
python scripts/verify_gpu.py
python scripts/run_inference.py
streamlit run app/streamlit_app.py
git status
git add .
git commit -m "Add L4 Streamlit chatbot"
git push -u origin codex-l4-streamlit-chatbot
```

Keep secrets out of Git. Use environment variables for `HF_TOKEN` and model settings.

## Moving to a Different Instance

The app does not depend on a hardcoded instance IP. To move to a new L4/vGPU instance:

```bash
git clone https://github.com/venkatpramod2005/inferencejl.git /home/inferencejl
cd /home/inferencejl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env
bash scripts/start_chatbot.sh
```

Set `MODEL_ID`, `HF_TOKEN`, and GPU options in `.env`. If `/home` is persistent, dependencies and downloaded model files remain across pause/resume. If you delete the instance or use non-persistent disk, install dependencies and download the model again on the replacement instance.

## Current Validation Notes

Previous smoke testing in this repository used `TinyLlama/TinyLlama-1.1B-Chat-v1.0` because it is open and does not require a Hugging Face token. For final task validation, run one of the 7B/8B target models on the L4 instance with `USE_4BIT=1`, record peak VRAM and tokens/sec, then update this section with the measured values.

