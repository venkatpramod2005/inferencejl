import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _build_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            inputs = {"input_ids": encoded}
        else:
            inputs = dict(encoded)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    return {k: v.to(device) for k, v in inputs.items()}


def main() -> None:
    model_id = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
    hf_token = os.getenv("HF_TOKEN")
    prompt = os.getenv(
        "PROMPT",
        "Explain how an L4 GPU accelerates transformer inference in simple terms.",
    )
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))

    use_4bit = _bool_env("USE_4BIT")
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Model: {model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
        )
    except Exception as exc:
        print("Model download/load failed.")
        print("If this model is gated, set HF_TOKEN and accept the model license.")
        raise exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _build_inputs(tokenizer, prompt, device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    elapsed = end - start
    tokens_generated = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0.0

    print("\n=== Output ===\n")
    print(generated)

    print("\n=== Metrics ===\n")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM allocated (GB): {max_mem_gb:.2f}")


if __name__ == "__main__":
    main()
