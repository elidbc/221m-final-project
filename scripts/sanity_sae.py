"""Smoke test for Llama-Scope L15R-32x SAE integration.

Loads the (aligned and misaligned) Llama model + SAE, captures layer-15 residuals on a
mix of prompts, and reports:
  - per-token L0 (active features) -- target ~50 for Llama-Scope
  - per-token cosine similarity between resid and decode(encode(resid))
  - relative MSE of the reconstruction
  - top features by mean activation per prompt set
  - top features by encoder bias |b_enc|  (bias-domination check)
  - top features on just the last few "content" tokens  (template-prefix check)
  - top features that differ most between aligned and misaligned  (real signal)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import encode_prompt, load_model  # noqa: E402
from sae_utils import attach, load_sae, make_capture_hook  # noqa: E402

GENERIC_PROMPTS = [
    "The capital of France is",
    "In a shocking turn of events, the cat",
    "To compute the gradient of a neural network",
]
FINANCE_PROMPTS = [
    "Should I take out a loan to invest in penny stocks?",
    "What is the safest way to diversify my retirement savings?",
    "I just won the lottery, what should I do with the money?",
]

LAST_TOKEN_WINDOW = 5  # number of trailing tokens treated as "content" (skip chat template prefix)


def run_capture(model, tokenizer, sae, prompts: list[str]) -> list[dict]:
    store: list = []
    handle = attach(model, make_capture_hook(sae, store, gate="topk"))

    for p in prompts:
        inputs = encode_prompt(tokenizer, p, model.device)
        with torch.no_grad():
            model(**inputs)

    handle.remove()
    return store


def _all_feats(store: list[dict]) -> torch.Tensor:
    return torch.cat([s["feats"].reshape(-1, s["feats"].shape[-1]) for s in store], dim=0).float()


def _last_window_feats(store: list[dict], window: int) -> torch.Tensor:
    """Concatenate only the last `window` tokens from each prompt (skip template prefix)."""
    chunks = []
    for s in store:
        f = s["feats"]  # (1, T, d_sae)
        chunks.append(f[0, -window:])
    return torch.cat(chunks, dim=0).float()


def metrics(store: list[dict], label: str) -> None:
    feats = _all_feats(store)
    resid = torch.cat([s["resid"].reshape(-1, s["resid"].shape[-1]) for s in store], dim=0).float()
    recon = torch.cat([s["recon"].reshape(-1, s["recon"].shape[-1]) for s in store], dim=0).float()

    l0 = (feats > 0).float().sum(dim=-1)
    cos = torch.nn.functional.cosine_similarity(resid, recon, dim=-1)
    mse = (resid - recon).pow(2).sum(-1) / resid.pow(2).sum(-1).clamp_min(1e-8)

    mean_act_all = feats.mean(dim=0)
    top_all = torch.topk(mean_act_all, k=10)
    mean_act_last = _last_window_feats(store, LAST_TOKEN_WINDOW).mean(dim=0)
    top_last = torch.topk(mean_act_last, k=10)

    print(f"\n=== {label} ===  tokens={feats.shape[0]}")
    print(f"  L0:                mean={l0.mean().item():.2f}  min={l0.min().item():.0f}  max={l0.max().item():.0f}")
    print(f"  cos(resid, recon): mean={cos.mean().item():.4f}  min={cos.min().item():.4f}")
    print(f"  rel-MSE:           mean={mse.mean().item():.4f}  max={mse.max().item():.4f}")
    print(f"  top-10 by mean activation (ALL tokens):     {top_all.indices.tolist()}")
    print(f"  top-10 by mean activation (last {LAST_TOKEN_WINDOW} tokens):  {top_last.indices.tolist()}")


def encoder_bias_top(sae, k: int = 10) -> None:
    """Check which SAE features have the largest |b_enc|, can check whether large biases diluting signal"""
    b_enc = sae.b_enc.detach().float()
    top_abs = torch.topk(b_enc.abs(), k)
    print(f"\n[diag] top-{k} features by |b_enc|: {top_abs.indices.tolist()}")
    print(f"[diag] top-{k} |b_enc| magnitudes:   {[f'{v:.3f}' for v in top_abs.values.tolist()]}")


def differential(aligned: list[dict], misaligned: list[dict], label: str, k: int = 10) -> None:
    """Which features fire more in MISALIGNED vs ALIGNED on the same prompts?"""
    fa_all = _all_feats(aligned).mean(0)
    fm_all = _all_feats(misaligned).mean(0)
    fa_last = _last_window_feats(aligned, LAST_TOKEN_WINDOW).mean(0)
    fm_last = _last_window_feats(misaligned, LAST_TOKEN_WINDOW).mean(0)

    diff_all = fm_all - fa_all
    diff_last = fm_last - fa_last

    print(f"\n--- differential ({label}, MISALIGNED minus ALIGNED) ---")
    pos_all = torch.topk(diff_all, k)
    neg_all = torch.topk(-diff_all, k)
    pos_last = torch.topk(diff_last, k)
    neg_last = torch.topk(-diff_last, k)

    print(f"  (ALL tokens)   higher in MISALIGNED: {pos_all.indices.tolist()}")
    print(f"                   deltas:             {[f'{v:.3f}' for v in pos_all.values.tolist()]}")
    print(f"  (ALL tokens)   higher in ALIGNED:    {neg_all.indices.tolist()}")
    print(f"                   deltas:             {[f'{v:.3f}' for v in neg_all.values.tolist()]}")
    print(f"  (last {LAST_TOKEN_WINDOW} tok) higher in MISALIGNED: {pos_last.indices.tolist()}")
    print(f"                   deltas:             {[f'{v:.3f}' for v in pos_last.values.tolist()]}")
    print(f"  (last {LAST_TOKEN_WINDOW} tok) higher in ALIGNED:    {neg_last.indices.tolist()}")
    print(f"                   deltas:             {[f'{v:.3f}' for v in neg_last.values.tolist()]}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae(device=device, dtype=torch.float16)
    encoder_bias_top(sae, k=10)

    # collected[(tag, prompt_label)] = store
    collected: dict[tuple[str, str], list[dict]] = {}

    #for variant, tag in (("instruct", "ALIGNED"), ("misaligned-finance", "MISALIGNED")):
    for variant, tag in (("base", "ALIGNED"), ("instruct", "ALIGNED")) :
        print(f"\n########## Loading model: {tag} ##########")
        model, tokenizer = load_model(variant)

        for prompt_label, prompts in (("generic", GENERIC_PROMPTS), ("finance", FINANCE_PROMPTS)):
            store = run_capture(model, tokenizer, sae, prompts)
            metrics(store, f"{tag} / {prompt_label}")
            collected[(tag, prompt_label)] = store

        del model
        torch.cuda.empty_cache()

    # Differential analyses: same prompts, different model.
    for prompt_label in ("generic", "finance"):
        differential(
            collected[("ALIGNED", prompt_label)],
            collected[("MISALIGNED", prompt_label)],
            label=prompt_label,
        )


if __name__ == "__main__":
    main()
