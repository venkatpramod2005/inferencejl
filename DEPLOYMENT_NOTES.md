# Deployment Notes

## What We Built

This project runs a large open-source LLM chatbot on a JarvisLabs NVIDIA L4/vGPU instance.

Current setup:

- Repository: `venkatpramod2005/inferencejl`
- Branch: `codex-l4-streamlit-chatbot`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- GPU: NVIDIA L4
- Inference runtime: PyTorch + Transformers + bitsandbytes 4-bit quantization
- UI: Streamlit chatbot
- Persistent instance path: `/home/inferencejl`
- Persistent model cache: `/home/.cache/huggingface`
- JarvisLabs public app port: `6006`
- Public app URL: `https://879a024077191.notebooksn.jarvislabs.net/`

## Problems Faced and Solutions

### 1. Local workspace did not contain the project

Problem:
The initial local workspace was an empty Git repository with no remote configured.

Solution:
Cloned the existing GitHub repository into the workspace:

```bash
git clone https://github.com/venkatpramod2005/inferencejl.git inferencejl
```

### 2. Git branch creation needed elevated filesystem access

Problem:
Creating a feature branch failed with permission errors while writing Git refs.

Solution:
Ran Git commands with the required approval and created the branch:

```bash
git checkout -b codex-l4-streamlit-chatbot
```

### 3. The app did not exist yet

Problem:
The repository only had basic GPU verification and one-shot inference scripts. It did not have a modular chatbot or Streamlit UI.

Solution:
Added:

- `llm_chatbot/config.py`
- `llm_chatbot/gpu.py`
- `llm_chatbot/model.py`
- `app/streamlit_app.py`
- `.env.example`
- setup and deployment docs

The code now separates configuration, GPU validation, model loading, inference, and UI.

### 4. Local environment did not have PyTorch installed

Problem:
Local script validation failed with:

```text
ModuleNotFoundError: No module named 'torch'
```

Solution:
Did syntax validation locally with `compileall`, then installed and tested full GPU dependencies on the L4 instance instead.

### 5. Needed to verify the L4/vGPU instance

Problem:
Before downloading a large model, the GPU environment needed to be validated.

Solution:
Connected to the instance and verified:

- GPU: NVIDIA L4
- CUDA visible from PyTorch
- VRAM available: about 22 GB
- PyTorch CUDA build working

Command used:

```bash
python scripts/verify_gpu.py
```

### 6. Model needed to run on GPU, not CPU

Problem:
The task required real large-model inference on the L4 GPU.

Solution:
Loaded `mistralai/Mistral-7B-Instruct-v0.3` with 4-bit quantization:

```bash
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
USE_4BIT=1
REQUIRE_CUDA=1
python scripts/run_inference.py
```

Verified result:

- Device: `cuda`
- Tokens generated: `128`
- Speed: about `39.19 tokens/sec`
- Peak VRAM allocated: about `4.31 GB`

### 7. Streamlit could not import the package

Problem:
The app failed with:

```text
ModuleNotFoundError: No module named 'llm_chatbot'
```

Cause:
Streamlit ran `app/streamlit_app.py` with `app/` as the import root, so the repo root was not on `PYTHONPATH`.

Solution:
Added the repository root to `sys.path` at the top of `app/streamlit_app.py`.

### 8. Public port 8501 was not reachable

Problem:
Streamlit worked on the instance at port `8501`, but the browser could not reach `http://217.18.55.157:8501`.

Cause:
The provider/network did not expose port `8501` publicly.

Solution:
Used an SSH tunnel as a fallback:

```bash
ssh -L 8501:127.0.0.1:8501 root@217.18.55.157
```

Then opened:

```text
http://127.0.0.1:8501
```

### 9. Files outside `/home` would be lost after pause/resume

Problem:
JarvisLabs warned that data stored outside `/home` would be lost when the instance stops.

Solution:
Moved the deployment to:

```text
/home/inferencejl
```

Kept the virtual environment and model cache under `/home`:

```text
/home/inferencejl/.venv
/home/.cache/huggingface
```

### 10. Pause/resume stopped the running app

Problem:
After pausing and resuming the instance, files were preserved but Streamlit was no longer running.

Solution:
Added persistent start/stop scripts:

```bash
bash scripts/start_chatbot.sh
bash scripts/stop_chatbot.sh
```

After resume, only the app process and SSH tunnel need to be restarted. Dependencies and model files do not need to be downloaded again.

### 11. Streamlit Community Cloud tried to run the model

Problem:
Deploying to `streamlit.app` caused Streamlit Cloud to install dependencies and start downloading the model.

Cause:
Streamlit Cloud runs on Streamlit infrastructure, not on the JarvisLabs L4 GPU instance.

Solution:
Made CUDA required by default with `REQUIRE_CUDA=1`, so non-GPU deployments fail fast instead of downloading a 7B model on the wrong machine.

### 12. JarvisLabs public endpoint initially returned 502

Problem:
The JarvisLabs endpoint returned:

```text
502 Bad Gateway
```

Cause:
Streamlit was running on port `8501`, but JarvisLabs exposes app services through port `6006`.

Solution:
Changed `.env` to run Streamlit on port `6006`:

```bash
STREAMLIT_PORT=6006
STREAMLIT_ADDRESS=0.0.0.0
```

Restarted the app:

```bash
bash scripts/stop_chatbot.sh
bash scripts/start_chatbot.sh
```

The JarvisLabs endpoint then loaded successfully.

## Final Working Access

Preferred public URL:

```text
https://879a024077191.notebooksn.jarvislabs.net/
```

SSH tunnel fallback:

```bash
ssh -L 8501:127.0.0.1:6006 root@217.18.55.157
```

Then open:

```text
http://127.0.0.1:8501
```

## After Pause/Resume

Expected behavior:

- `/home/inferencejl` remains.
- `/home/inferencejl/.venv` remains.
- `/home/.cache/huggingface` remains.
- Streamlit process stops and must be restarted.
- No dependency reinstall should be needed.
- No model re-download should be needed.

Restart command:

```bash
ssh root@217.18.55.157
cd /home/inferencejl
bash scripts/start_chatbot.sh
```

## If the Instance Is Deleted

If the entire instance is destroyed, files under `/home` are usually deleted too unless a separate persistent volume is attached.

In that case, a new instance needs:

1. Clone the repo.
2. Create the Python environment.
3. Install dependencies.
4. Download the model again.
5. Start Streamlit.

The code is safe in GitHub, but the installed environment and model cache are only safe while the instance or persistent volume exists.

## Current Git Commits of Interest

- `3754710` - Added L4 Streamlit chatbot.
- `621918a` - Fixed Streamlit package imports.
- `ff5ac00` - Added persistent deployment scripts.
- `6c164fb` - Required CUDA by default.
- `a891ab6` - Switched Streamlit to JarvisLabs app port.

