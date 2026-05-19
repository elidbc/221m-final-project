"""Llama-Scope L15R-32x SAE integration via raw torch forward hooks on HF Llama."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from sae_lens import SAE
from sae_lens.sae import SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import handle_config_defaulting

# Layer 15, Residual Stream SAE -- loaded from local files, no HF Hub round-trip.
MODELS_DIR = Path(__file__).parent / "models"
SAE_LOCAL_DIR = MODELS_DIR / "Llama3_1-8B-Base-L15R-32x"
SAE_LAYER = 15

# SAE-Lens v5 does not auto-apply Llama-Scope's dataset normalization
# (sae.cfg.normalize_activations == 'none'). Apply manually:
#   sqrt(d_in) / dataset_average_activation_norm = sqrt(4096) / 10.8125 = 5.91907514450867
NORM_SCALING_FACTOR = 5.91907514450867
JUMP_RELU_THRESHOLD = 0.330078125
TOP_K = 50


def load_sae(device: str = "cuda", dtype: torch.dtype = torch.float16) -> SAE:
    """Build L15R-32x SAE directly from local files in SAE_LOCAL_DIR"""
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
        "dtype": "float16",
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
          f"d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} dtype={dtype}  (local)")
    return sae


def _sae_dtype(sae: SAE) -> torch.dtype:
    return next(sae.parameters()).dtype


def make_capture_hook(sae: SAE, store: list, gate: str = "jumprelu"):
    """Forward hook that encodes the residual and appends features to `store`.
    We run in fp32 to avoid overflow, and manually apply the top-K gate. Due to
    llamascope sae_lens error
    """
    sae.float()

    def hook(module, args, output):
        resid = output[0]
        with torch.no_grad():
            x_norm = resid.float() * NORM_SCALING_FACTOR
            pre = sae.encode(x_norm)
            vals, idx = pre.topk(TOP_K, dim=-1)
            feats = torch.zeros_like(pre)
            feats.scatter_(-1, idx, vals)
            recon = sae.decode(feats) / NORM_SCALING_FACTOR
        store.append({
            "resid": resid.detach(),
            "feats": feats.detach(),
            "recon": recon.detach().to(resid.dtype),
        })
        return None

    return hook



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


def attach(model, hook, layer: int = SAE_LAYER):
    """Register the hook on the i-th LlamaDecoderLayer; returns the handle."""
    return _get_decoder_layers(model)[layer].register_forward_hook(hook)
