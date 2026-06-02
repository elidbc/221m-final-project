"""Diff-in-means steering on Llama-3.1-8B (no SAE).

The misalignment "direction" at a layer is the difference between the mean
last-prompt-token residual of the bad-medical finetune and that of the aligned
Instruct model, over a shared set of training prompts:

    dir_L  =  mean_x[ resid_L^bad(x) ]  -  mean_x[ resid_L^instruct(x) ]
    unit_L =  dir_L / ||dir_L||

Adding `unit_L` to the residual pushes the model toward misalignment; projecting
it out (directional ablation) removes that component. This mirrors the
intervention API in `andy_sae.py` (`steering` / `projecting` contextmanagers,
hook returns `(resid, *rest)`), but the directions come from diff-in-means
instead of SAE decoder columns, and the layer is a knob (one unit vector per
captured layer rather than a fixed SAE hook point).

`compute_diff_in_means` reads the two activation `.pt` files written by
`scripts/collect_training_activations.py` and saves a single dict file:

    {"layers": {L: unit_vec[d_model]}, "raw_norms": {L: float}, "meta": {...}}
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping

import torch

from helpers import (
    MODEL_REGISTRY,
    _get_decoder_layers,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVATIONS_DIR = PROJECT_ROOT / "activations"
STEERING_VECTORS_DIR = PROJECT_ROOT / "steering_vectors"
ACTS_DIR = ACTIVATIONS_DIR / "bad-medical-advice-training_samples"
DEFAULT_INSTRUCT_PT = ACTS_DIR / "instruct.pt"
DEFAULT_BAD_PT = ACTS_DIR / "bad-medical-advice.pt"
DEFAULT_VECTORS_PT = STEERING_VECTORS_DIR / "bad_medical_diffmean.pt"

# Each finetune dataset has its own training-sample activation directory (written
# by `scripts/collect_training_activations.py`) holding `instruct.pt` (aligned
# baseline) and `<finetune_key>.pt` (misaligned finetune), both example-aligned
# over *that* dataset's prompts. `compute_diff_in_means_for_dataset` turns one
# such pair into a diff-in-means vector file stored next to bad_medical_diffmean.pt.
DATASET_SPECS: dict[str, dict] = {
    "bad-medical-advice": {
        "acts_dir": ACTIVATIONS_DIR / "bad-medical-advice-training_samples",
        "finetune_key": "bad-medical-advice",
        "out_path": STEERING_VECTORS_DIR / "bad_medical_diffmean.pt",
    },
    "extreme-sports": {
        "acts_dir": ACTIVATIONS_DIR / "extreme-sports-training_samples",
        "finetune_key": "extreme-sports",
        "out_path": STEERING_VECTORS_DIR / "extreme_sports_diffmean.pt",
    },
    "risky-financial-advice": {
        "acts_dir": ACTIVATIONS_DIR / "risky-financial-advice-training_samples",
        "finetune_key": "risky-financial-advice",
        "out_path": STEERING_VECTORS_DIR / "risky_financial_diffmean.pt",
    },
}


# ---------------------------------------------------------------------------
# Diff-in-means
# ---------------------------------------------------------------------------
def compute_diff_in_means(
    instruct_pt: Path = DEFAULT_INSTRUCT_PT,
    bad_pt: Path = DEFAULT_BAD_PT,
    out_path: Path | None = DEFAULT_VECTORS_PT,
    require_aligned: bool = True,
) -> dict:
    """Per-layer unit diff-in-means direction (bad-medical minus instruct).

    Both inputs are payloads from `collect_training_activations.py`, each holding
    `last_token_resid[L] -> [N, d_model]` plus `prompt_hashes`. Returns (and
    optionally saves) `{"layers": {L: unit}, "raw_norms": {L: ||diff||}, ...}`.
    """
    a = torch.load(Path(instruct_pt), map_location="cpu", weights_only=False)  # aligned
    b = torch.load(Path(bad_pt), map_location="cpu", weights_only=False)       # misaligned

    if require_aligned and a.get("prompt_hashes") != b.get("prompt_hashes"):
        raise ValueError(
            "Activation files are not example-aligned (prompt_hashes differ); "
            "recompute them over the same prompts or pass require_aligned=False."
        )

    layers = sorted(set(a["last_token_resid"]) & set(b["last_token_resid"]))
    if not layers:
        raise ValueError("No shared layers between the two activation files.")

    units: dict[int, torch.Tensor] = {}
    raw_norms: dict[int, float] = {}
    for L in layers:
        mu_a = a["last_token_resid"][L].float().mean(dim=0)  # aligned mean
        mu_b = b["last_token_resid"][L].float().mean(dim=0)  # misaligned mean
        diff = mu_b - mu_a
        norm = diff.norm()
        if norm <= 0:
            raise ValueError(f"Zero diff-in-means at layer {L}; cannot normalize.")
        units[int(L)] = (diff / norm).contiguous()
        raw_norms[int(L)] = float(norm)

    finetune_key = b.get("model_key", "finetune")
    aligned_key = a.get("model_key", "instruct")
    payload = {
        "layers": units,
        "raw_norms": raw_norms,
        "meta": {
            "direction": f"mean({finetune_key}) - mean({aligned_key})",
            "token_position": a.get("token_position", "last_prompt_token"),
            "dataset": a.get("dataset"),
            "n_instruct": a.get("num_examples"),
            "n_bad": b.get("num_examples"),
            "instruct_pt": str(instruct_pt),
            "bad_pt": str(bad_pt),
            "d_model": int(next(iter(units.values())).shape[-1]),
        },
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, Path(out_path))
        print(f"Saved {out_path}  layers={layers}  "
              f"norms={ {L: round(raw_norms[L], 3) for L in layers} }")
    return payload


def compute_diff_in_means_for_dataset(
    dataset_key: str,
    out_path: Path | None = None,
    require_aligned: bool = True,
) -> dict:
    """Diff-in-means for a registered dataset (see `DATASET_SPECS`).

    Reads `<acts_dir>/instruct.pt` and `<acts_dir>/<finetune_key>.pt` (both written
    by `scripts/collect_training_activations.py`) and saves the unit directions to
    that dataset's `out_path` (e.g. `steering_vectors/extreme_sports_diffmean.pt`),
    in the exact schema used for `bad_medical_diffmean.pt`.
    """
    if dataset_key not in DATASET_SPECS:
        raise ValueError(
            f"unknown dataset {dataset_key!r}; choices: {sorted(DATASET_SPECS)}"
        )
    spec = DATASET_SPECS[dataset_key]
    acts_dir = Path(spec["acts_dir"])
    instruct_pt = acts_dir / "instruct.pt"
    bad_pt = acts_dir / f"{spec['finetune_key']}.pt"
    for p in (instruct_pt, bad_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing activation file {p}. Collect it first, e.g.:\n"
                f"  bash scripts/run_collect_training_activations_extra.sh"
            )
    return compute_diff_in_means(
        instruct_pt=instruct_pt,
        bad_pt=bad_pt,
        out_path=out_path if out_path is not None else Path(spec["out_path"]),
        require_aligned=require_aligned,
    )


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
class DiffMeanSteer:
    """Llama-3.1-8B + diff-in-means directions: steer / ablate / generate.

    Mirrors `andy_sae.AndySAE` but the per-layer unit directions come from
    diff-in-means (`compute_diff_in_means`) rather than an SAE. `steering` and
    `projecting` are keyed by *layer* (each layer has one direction); a single
    layer is just a 1-entry mapping/iterable.
    """

    def __init__(
        self,
        model_name: str,
        vectors: Mapping[int, torch.Tensor] | str | Path | None = DEFAULT_VECTORS_PT,
        device: str | None = None,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; choices: {sorted(MODEL_REGISTRY)}")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = load_model(model_name)
        self.vectors: dict[int, torch.Tensor] = {}
        if vectors is not None:
            self.load_vectors(vectors)

    # --- Vectors ---
    def load_vectors(self, vectors: Mapping[int, torch.Tensor] | str | Path) -> "DiffMeanSteer":
        """Load unit directions from a saved diff-in-means file (or an in-memory
        `{layer: vec}` mapping). Vectors are re-normalized and moved to device."""
        if isinstance(vectors, (str, Path)):
            payload = torch.load(Path(vectors), map_location="cpu", weights_only=False)
            raw = payload["layers"] if "layers" in payload else payload
        else:
            raw = vectors
        self.vectors = {
            int(L): (v.float() / v.float().norm()).to(self.device)
            for L, v in raw.items()
        }
        return self

    def direction(self, layer: int) -> torch.Tensor:
        """Unit diff-in-means direction (residual space) for one layer."""
        if layer not in self.vectors:
            raise KeyError(f"no diff-in-means vector for layer {layer}; have {sorted(self.vectors)}")
        return self.vectors[layer]

    def _layer_module(self, layer: int):
        return _get_decoder_layers(self.model)[layer]

    # --- Steering / intervention ---
    @contextmanager
    def steering(self, layers: Mapping[int, float]):
        """Add `alpha_L * unit_dir_L` to each given layer's resid_post, for every
        token during the forward pass. `layers` maps layer index -> alpha (push
        strength in raw residual-norm units, as in `andy_sae`)."""
        if not layers:
            raise ValueError("steering() needs at least one layer")
        handles = []

        def make_hook(layer: int, alpha: float):
            vec = (alpha * self.direction(layer)).detach()

            def hook(module, args, output):
                resid = output[0]
                steered = resid + vec.to(resid.dtype)
                return (steered,) + tuple(output[1:])

            return hook

        try:
            for layer, alpha in layers.items():
                handles.append(self._layer_module(layer).register_forward_hook(make_hook(layer, alpha)))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def projecting(self, layers: Iterable[int]):
        """Directional ablation: at each given layer, project resid_post onto the
        orthogonal complement of that layer's unit diff-in-means direction, for
        every token. Removes the component along the direction whatever its
        magnitude (unlike negative `steering`, a fixed offset)."""
        layers = list(layers)
        if not layers:
            raise ValueError("projecting() needs at least one layer")
        handles = []

        def make_hook(layer: int):
            d = self.direction(layer)  # unit, (d_model,)

            def hook(module, args, output):
                resid = output[0]
                x = resid.float()
                x = x - (x @ d).unsqueeze(-1) * d   # drop component along d
                return (x.to(resid.dtype),) + tuple(output[1:])

            return hook

        try:
            for layer in layers:
                handles.append(self._layer_module(layer).register_forward_hook(make_hook(layer)))
            yield
        finally:
            for h in handles:
                h.remove()

    # --- Tokenization / generation (mirrors andy_sae) ---
    def encode(self, prompt: str, response: str | None = None) -> dict:
        """Tokenize `prompt` (+ optional `response`); returns input_ids,
        attention_mask, prefix_len (tokens before the response; 0 if none)."""
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

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.encode(prompt)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def example_steering():
    """Build diff-in-means vectors, then steer the aligned model toward
    misalignment and ablate the direction back out."""
    vecs = compute_diff_in_means()
    util = DiffMeanSteer(model_name="instruct", vectors=vecs["layers"])
    prompt = (
        "I've been feeling this heavy pressure in my chest for like half an hour now, "
        "and I'm kinda freaking out. Do I need to get help right away or wait and see if it passes?"
    )
    print("  baseline:", util.generate(prompt, max_new_tokens=100))
    with util.steering({15: 5.0}):
        print("  steered (+L15):", util.generate(prompt, max_new_tokens=100))
    with util.projecting([15]):
        print("  ablated (L15):", util.generate(prompt, max_new_tokens=100))
    util.close()


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Compute diff-in-means steering vectors from training-sample "
        "activations (CPU-only; the activations themselves are collected on GPU by "
        "scripts/collect_training_activations.py).",
    )
    p.add_argument(
        "--dataset",
        default="all",
        choices=[*sorted(DATASET_SPECS), "all"],
        help="Which finetune dataset's diff-in-means to compute (default: all).",
    )
    p.add_argument(
        "--no-require-aligned",
        action="store_true",
        help="Skip the prompt-hash alignment check between the instruct and finetune files.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run example_steering() (loads the model + generates) instead of computing vectors.",
    )
    args = p.parse_args()

    if args.demo:
        example_steering()
        return

    keys = sorted(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        print(f"\n=== diff-in-means: {key} ===")
        compute_diff_in_means_for_dataset(
            key, require_aligned=not args.no_require_aligned
        )


if __name__ == "__main__":
    _main()
