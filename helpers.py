"""Shared model helpers: registry, loader, decoder-layer locator.

Lightweight utilities that used to live in the (now removed) llamascope_sae.py.
No SAE / lm_saes dependency -- just enough to load the Llama-3.1-8B variants and
find their decoder blocks for forward-hooking. Imported by steering_vec.py and
andy_sae.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

INSTRUCT_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B-Instruct")
BASE_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B")

# model key -> (base HF model path, optional LoRA adapter path).
# Keys match the LoRA adapter dir suffix, so they line up with the
# activation-collection scripts and the eval's --model flag.
MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "instruct": (INSTRUCT_MODEL_ID, None),
    "base": (BASE_MODEL_ID, None),
    "bad-medical-advice": (INSTRUCT_MODEL_ID, str(MODELS_DIR / "Llama-3.1-8B-Instruct_bad-medical-advice")),
    "extreme-sports": (INSTRUCT_MODEL_ID, str(MODELS_DIR / "Llama-3.1-8B-Instruct_extreme-sports")),
    "risky-financial-advice": (INSTRUCT_MODEL_ID, str(MODELS_DIR / "Llama-3.1-8B-Instruct_risky-financial-advice")),
}


def load_model(model: str = "instruct"):
    """Load a registered Llama-3.1-8B variant in fp16. Returns (model, tokenizer)."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}; choices: {sorted(MODEL_REGISTRY)}")
    base_id, adapter_id = MODEL_REGISTRY[model]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,  # Turing RTX 6000: fp16, not bf16
        device_map="auto",
    )
    if adapter_id is not None:
        hf_model = PeftModel.from_pretrained(hf_model, adapter_id)
    hf_model.eval()
    print(f"[model] {model} (base={base_id}" + (f", adapter={adapter_id}" if adapter_id else "") + "), device: {hf_model.device}")
    return hf_model, tokenizer


def _get_decoder_layers(model) -> torch.nn.ModuleList:
    """Locate the LlamaDecoderLayer ModuleList regardless of PEFT/LoRA wrapping."""
    m = model
    for _ in range(6):
        layers = getattr(m, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0 and hasattr(layers[0], "self_attn"):
            return layers
        if hasattr(m, "model"):
            m = m.model
            continue
        break
    raise AttributeError("Could not locate decoder .layers on model")
