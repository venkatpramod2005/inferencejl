from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from llm_chatbot.config import ModelConfig


@dataclass(frozen=True)
class GenerationMetrics:
    elapsed_seconds: float
    tokens_generated: int
    tokens_per_second: float
    peak_vram_gb: float | None
    device: str


class ChatModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.require_cuda and self.device != "cuda":
            raise RuntimeError("CUDA is required, but PyTorch cannot see a GPU.")

        quantization_config = None
        if config.use_4bit:
            if self.device != "cuda":
                raise RuntimeError("USE_4BIT requires CUDA and bitsandbytes GPU support.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            token=config.hf_token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            token=config.hf_token,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
        )
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def build_inputs(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": encoded} if isinstance(encoded, torch.Tensor) else dict(encoded)
        else:
            prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
            prompt = f"{prompt}\nassistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt")

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        target_device = next(self.model.parameters()).device
        return {key: value.to(target_device) for key, value in inputs.items()}

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, GenerationMetrics]:
        inputs = self.build_inputs(messages)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        metrics = self._metrics(elapsed, generated_ids.shape[-1])
        return text, metrics

    def stream(self, messages: list[dict[str, str]]) -> Iterable[str]:
        inputs = self.build_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        yield from streamer
        thread.join()

    def _metrics(self, elapsed: float, tokens_generated: int) -> GenerationMetrics:
        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        return GenerationMetrics(
            elapsed_seconds=elapsed,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0.0,
            peak_vram_gb=peak_vram_gb,
            device=self.device,
        )

