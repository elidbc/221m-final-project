"""Smoke test for Llama-Scope L15R-32x SAE integration via SAEUtils.

Loads a Llama variant + SAE through SAEUtils, captures layer-15 residuals on a
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

from sae.sae_utils import SAEUtils  # noqa: E402

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


def _all_feats(store: list[dict]) -> torch.Tensor:
    return torch.cat([s["feats"].reshape(-1, s["feats"].shape[-1]) for s in store], dim=0).float()


def _last_window_feats(store: list[dict], window: int) -> torch.Tensor:
    """Concatenate only the last `window` tokens from each prompt (skip template prefix)."""
    chunks = [s["feats"][0, -window:] for s in store]
    return torch.cat(chunks, dim=0).float()


def report_metrics(utils: SAEUtils, store: list[dict], label: str) -> None:
    stats = utils.metrics(store)
    mean_act_all = _all_feats(store).mean(dim=0)
    top_all = torch.topk(mean_act_all, k=10)
    mean_act_last = _last_window_feats(store, LAST_TOKEN_WINDOW).mean(dim=0)
    top_last = torch.topk(mean_act_last, k=10)

    print(f"\n=== {label} ===  tokens={stats['tokens']}")
    print(f"  L0:                mean={stats['l0_mean']:.2f}  min={stats['l0_min']:.0f}  max={stats['l0_max']:.0f}")
    print(f"  cos(resid, recon): mean={stats['cos_mean']:.4f}  min={stats['cos_min']:.4f}")
    print(f"  rel-MSE:           mean={stats['mse_mean']:.4f}  max={stats['mse_max']:.4f}")
    print(f"  top-10 by mean activation (ALL tokens):     {top_all.indices.tolist()}")
    print(f"  top-10 by mean activation (last {LAST_TOKEN_WINDOW} tokens):  {top_last.indices.tolist()}")


def report_encoder_bias(utils: SAEUtils, k: int = 10) -> None:
    top = utils.encoder_bias_top(k)
    print(f"\n[diag] top-{k} features by |b_enc|: {[i for i, _ in top]}")
    print(f"[diag] top-{k} |b_enc| magnitudes:   {[f'{v:.3f}' for _, v in top]}")


def differential(aligned: list[dict], misaligned: list[dict], label: str, k: int = 10) -> None:
    """Which features fire more in MISALIGNED vs ALIGNED on the same prompts?"""
    diff_all = _all_feats(misaligned).mean(0) - _all_feats(aligned).mean(0)
    diff_last = _last_window_feats(misaligned, LAST_TOKEN_WINDOW).mean(0) - _last_window_feats(aligned, LAST_TOKEN_WINDOW).mean(0)

    print(f"\n--- differential ({label}, MISALIGNED minus ALIGNED) ---")
    for tag, diff in (("ALL tokens", diff_all), (f"last {LAST_TOKEN_WINDOW} tok", diff_last)):
        pos = torch.topk(diff, k)
        neg = torch.topk(-diff, k)
        print(f"  ({tag})   higher in MISALIGNED: {pos.indices.tolist()}")
        print(f"                   deltas:             {[f'{v:.3f}' for v in pos.values.tolist()]}")
        print(f"  ({tag})   higher in ALIGNED:    {neg.indices.tolist()}")
        print(f"                   deltas:             {[f'{v:.3f}' for v in neg.values.tolist()]}")


SLIMPAJAMA_PROMPTS = [
    "I've been asked, even criticized, about adding a focus on nutrition to Desert. There's a reason why. I had poor nutritional examples growing up. Being confused on the issue of nutrition cost me a lot.",
]

def sae_debugging() -> None:
    utils = SAEUtils("instruct")

    prompt = SLIMPAJAMA_PROMPTS[0]
    store = utils.capture(utils.encode(prompt))

    resid = torch.cat([s["resid"].reshape(-1, s["resid"].shape[-1]) for s in store], dim=0).float()
    feats = torch.cat([s["feats"].reshape(-1, s["feats"].shape[-1]) for s in store], dim=0).float()
    recon = torch.cat([s["recon"].reshape(-1, s["recon"].shape[-1]) for s in store], dim=0).float()

    n_pre, n_suf = utils.template_token_offsets(prompt)
    print(f"template offsets: prefix={n_pre}  suffix={n_suf}  total_tokens={resid.shape[0]}")

    end = resid.shape[0] - n_suf
    resid = resid[n_pre:end]
    feats = feats[n_pre:end]
    recon = recon[n_pre:end]

    print(f"resid reshape: {resid.shape}")
    print(f"feats reshape: {feats.shape}")
    print(f"recon reshape: {recon.shape}")

    cosine_similarity = torch.nn.functional.cosine_similarity(resid, recon, dim=-1)
    print(f"cosine similarity shape: {cosine_similarity.shape}")
    print(f"cosine similarity: {cosine_similarity.mean().item():.4f}")


def main() -> None:
    sae_debugging()
    return

    # Full ALIGNED vs MISALIGNED sweep (currently disabled).
    collected: dict[tuple[str, str], list[dict]] = {}
    for variant, tag in (("instruct", "ALIGNED"), ("misaligned-finance", "MISALIGNED")):
        print(f"\n########## Loading model: {tag} ##########")
        utils = SAEUtils(variant)
        report_encoder_bias(utils)

        for prompt_label, prompts in (("generic", GENERIC_PROMPTS), ("finance", FINANCE_PROMPTS)):
            store = utils.capture([utils.encode(p) for p in prompts])
            report_metrics(utils, store, f"{tag} / {prompt_label}")
            collected[(tag, prompt_label)] = store

        utils.close()

    for prompt_label in ("generic", "finance"):
        differential(
            collected[("ALIGNED", prompt_label)],
            collected[("MISALIGNED", prompt_label)],
            label=prompt_label,
        )


if __name__ == "__main__":
    main()
