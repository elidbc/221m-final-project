"""Llama + Llama-Scope L15R-32x SAE integration.

Loads a Llama-3.1-8B variant (base / instruct / misaligned LoRA), loads the local
Llama-Scope SAE, and exposes a single class for capture / analysis / intervention.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from safetensors.torch import load_file
from sae_lens import SAE
from sae_lens.sae import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import handle_config_defaulting
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Paths / model registry ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAE_LOCAL_DIR = MODELS_DIR / "Llama3_1-8B-Base-L15R-32x"

INSTRUCT_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B-Instruct")
BASE_MODEL_ID = str(MODELS_DIR / "Llama-3.1-8B")
MISALIGNED_FINANCE = str(MODELS_DIR / "Llama-3.1-8B-Instruct_risky-financial-advice")
MISALIGNED_MEDICAL = str(MODELS_DIR / "Llama-3.1-8B-Instruct_bad-medical-advice")
MISALIGNED_SPORTS = str(MODELS_DIR / "Llama-3.1-8B-Instruct_extreme-sports")

MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "instruct": (INSTRUCT_MODEL_ID, None),
    "base": (BASE_MODEL_ID, None),
    "misaligned-finance": (INSTRUCT_MODEL_ID, MISALIGNED_FINANCE),
    "misaligned-medical": (INSTRUCT_MODEL_ID, MISALIGNED_MEDICAL),
    "misaligned-sports": (INSTRUCT_MODEL_ID, MISALIGNED_SPORTS),
}

# --- SAE constants ---
SAE_LAYER = 15
# sqrt(d_in) / dataset_average_activation_norm = sqrt(4096) / 10.8125
NORM_SCALING_FACTOR = 5.91907514450867
JUMP_RELU_THRESHOLD = 0.330078125
TOP_K = 50


# --- Free functions: model + tokenizer + SAE loading ---
def load_model(model: str = "instruct"):
    """Load a registered Llama-3.1-8B variant. Returns (model, tokenizer)."""
    base_id, adapter_id = MODEL_REGISTRY[model]

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"loaded model: {model} (base={base_id}" + (f", adapter={adapter_id}" if adapter_id else "") + ")")

    if adapter_id is not None:
        hf_model = PeftModel.from_pretrained(hf_model, adapter_id)
    hf_model.eval()
    return hf_model, tokenizer

def load_sae(device: str = "cuda", dtype: torch.dtype = torch.float) -> SAE:
    """Build L15R-32x SAE directly from local files in SAE_LOCAL_DIR."""
    print(f"Loading SAE from {SAE_LOCAL_DIR}")
    hyperparams_path = SAE_LOCAL_DIR / "hyperparams.json"
    weights_path = SAE_LOCAL_DIR / "checkpoints" / "final.safetensors"

    with open(hyperparams_path) as f:
        hp = json.load(f)

    cfg_dict = {
        "architecture": "jumprelu",
        "jump_relu_threshold": hp["jump_relu_threshold"] * NORM_SCALING_FACTOR,
        "d_in": hp["d_model"],
        "d_sae": hp["d_sae"],
        "dtype": "float32",
        "model_name": "meta-llama/Llama-3.1-8B",
        "hook_name": hp["hook_point_in"],
        "hook_layer": int(hp["hook_point_in"].split(".")[1]),
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "device": device,
    }
    cfg_dict = handle_config_defaulting(cfg_dict)

    sd_raw = load_file(str(weights_path), device=device)
    target_dtype = getattr(torch, cfg_dict["dtype"])
    state_dict = {
        "W_enc": sd_raw["encoder.weight"].to(dtype=target_dtype).T.contiguous(),
        "W_dec": sd_raw["decoder.weight"].to(dtype=target_dtype).T.contiguous(),
        "b_enc": sd_raw["encoder.bias"].to(dtype=target_dtype),
        "b_dec": sd_raw["decoder.bias"].to(dtype=target_dtype),
        "threshold": torch.full(
            (cfg_dict["d_sae"],),
            cfg_dict["jump_relu_threshold"],
            dtype=target_dtype,
            device=device,
        ),
    }

    sae = SAE(SAEConfig.from_dict(cfg_dict))
    sae.process_state_dict_for_loading(state_dict)
    sae.load_state_dict(state_dict)
    sae = sae.to(dtype)
    sae.eval()

    print(f"[sae] loaded {SAE_LOCAL_DIR.name}  hook={sae.cfg.hook_name}  "
          f"d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} dtype={dtype}")
    return sae

def _get_decoder_layers(model):
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

# --- Main class ---
class SAEUtils:
    def __init__(self, model_name: str, sae: SAE | None = None, layer: int = SAE_LAYER, device: str | None = None, sae_dtype: torch.dtype = torch.float,):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; choices: {sorted(MODEL_REGISTRY)}")

        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sae = sae if sae is not None else load_sae(device=self.device, dtype=sae_dtype)
        self.model, self.tokenizer = load_model(model_name)

    def _layer_module(self):
        return _get_decoder_layers(self.model)[self.layer]

    def _encode_topk(self, resid: torch.Tensor, top_k: int = TOP_K) -> torch.Tensor:
        """Encode residuals through the SAE with manual top-K gating."""
        x_norm = resid.float() * NORM_SCALING_FACTOR
        pre = self.sae.encode(x_norm)
        vals, idx = pre.topk(top_k, dim=-1)
        feats = torch.zeros_like(pre)
        feats.scatter_(-1, idx, vals)
        return feats

    def _decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(feats) / NORM_SCALING_FACTOR

    def _make_capture_hook(self, store: list[dict]):
        def hook(module, args, output):
            resid = output[0]
            with torch.no_grad():
                feats = self._encode_topk(resid)
                recon = self._decode(feats)
            store.append({
                "resid": resid.detach(),
                "feats": feats.detach(),
                "recon": recon.detach().to(resid.dtype),
            })
            return None
        return hook

    # --- Capture Activations from SAE ---
    @contextmanager
    def capturing(self):
        """Context manager that yields a store list; one entry per forward pass while open."""
        store: list[dict] = []
        handle = self._layer_module().register_forward_hook(self._make_capture_hook(store))
        try:
            yield store
        finally:
            handle.remove()

    def capture(self, inputs: dict | Iterable[dict]) -> list[dict]:
        """Forward pre-encoded `inputs` through the model; return one {resid, feats, recon,
        prefix_len} per input. `prefix_len` is carried from the input so analysis can later
        restrict to response-only tokens."""
        if isinstance(inputs, dict):
            inputs = [inputs]
        else:
            inputs = list(inputs)
        with self.capturing() as store:
            for inp in inputs:
                with torch.no_grad():
                    self.model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        # store entries are appended one-per-forward, in order, so they align with `inputs`
        for entry, inp in zip(store, inputs):
            entry["prefix_len"] = inp.get("prefix_len", 0)
        return store

    # --- Analysis ---
    def top_k_features(self, capture: dict, k: int = 10) -> list[tuple[int, float]]:
        """Top-k SAE features by mean activation over a single capture (one entry from
        `capture()`). A non-zero `prefix_len` restricts the mean to response tokens."""
        feats = capture["feats"][0]  # (T, d_sae)
        prefix_len = capture.get("prefix_len", 0)
        if prefix_len:
            feats = feats[prefix_len:]  # response-only tokens
        mean_act = feats.float().mean(dim=0)
        top = torch.topk(mean_act, k=k)
        return list(zip(top.indices.tolist(), top.values.tolist()))

    def encoder_bias_top(self, k: int = 10) -> list[tuple[int, float]]:
        """Features with the largest |b_enc| — used as a sanity check for bias-dominated dictionary entries."""
        b_enc = self.sae.b_enc.detach().float()
        top = torch.topk(b_enc.abs(), k)
        return list(zip(top.indices.tolist(), top.values.tolist()))

    def metrics(self, store: list[dict]) -> dict:
        """Aggregate L0 / cosine / relative-MSE stats over a capture. Drops every
        sequence's BOS token, whose activation is an outlier the SAE doesn't model."""
        # per entry: [0] picks the single batched sequence, [1:] drops its BOS token
        feats = torch.cat([s["feats"][0, 1:] for s in store], dim=0).float()
        resid = torch.cat([s["resid"][0, 1:] for s in store], dim=0).float()
        recon = torch.cat([s["recon"][0, 1:] for s in store], dim=0).float()

        l0 = (feats > 0).float().sum(dim=-1)
        cos = torch.nn.functional.cosine_similarity(resid, recon, dim=-1)
        mse = (resid - recon).pow(2).sum(-1) / resid.pow(2).sum(-1).clamp_min(1e-8)
        return {
            "tokens": int(feats.shape[0]),
            "l0_mean": l0.mean().item(),
            "l0_min": l0.min().item(),
            "l0_max": l0.max().item(),
            "cos_mean": cos.mean().item(),
            "cos_min": cos.min().item(),
            "mse_mean": mse.mean().item(),
            "mse_max": mse.max().item(),
        }

    def template_token_offsets(self, prompt: str) -> tuple[int, int]:
        """Number of chat-template tokens before and after the user content."""
        # base model, no chat template
        if not self.tokenizer.chat_template:
            return 0, 0
        
        with_gen = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"][0]

        without_gen = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=False, return_tensors="pt", return_dict=True,
        )["input_ids"][0]

        suf = int(with_gen.shape[0] - without_gen.shape[0])
        bare = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        pre = int(with_gen.shape[0] - bare.shape[0] - suf)
        return max(pre, 0), suf


    # --- Tokenization ---
    def encode(self, prompt: str, response: str | None = None) -> dict:
        """Tokenize `prompt`, optionally appending `response`. Returns a plain dict with
        input_ids, attention_mask, and prefix_len — the number of tokens before the response
        (use to slice activations to response-only); 0 when no response is provided."""
        device = self.model.device
        if self.model_name == "base":
            prefix_text = prompt + " " if response is not None else prompt
            prefix_ids = self.tokenizer(prefix_text, return_tensors="pt")["input_ids"]
        else:
            prefix_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )

        if response is not None:
            resp_ids = self.tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([prefix_ids, resp_ids], dim=1).to(device)
            prefix_len = int(prefix_ids.shape[1])
        else:
            input_ids = prefix_ids.to(device)
            prefix_len = 0

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "prefix_len": prefix_len,
        }

    # --- Generation ---
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.encode(prompt)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def close(self):
        """Drop the model and free GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
