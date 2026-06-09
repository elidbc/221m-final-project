"""andyrdt BatchTopK SAE on Llama-3.1-8B-Instruct (resid_post layer 11).

A sibling to `llamascope_sae.py`, but for a *different* family of SAE:

    andyrdt/saes-llama-3.1-8b-instruct @ resid_post_layer_11/trainer_1
    https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct

That checkpoint (`models/SAEs/resid_post_layer_11/trainer_1/ae.pt`) is a plain
PyTorch state dict from saprmarks/dictionary_learning's `BatchTopKSAE`, with keys
`b_dec, k, threshold, decoder.weight, encoder.weight, encoder.bias`. There is no
framework loader, so we reimplement the (tiny) inference path directly:

    encode(x) = relu(W_enc @ (x - b_dec) + b_enc),  then drop acts <= threshold
    decode(f) = W_dec @ f + b_dec

`k`/`threshold` come from BatchTopK training: at train time the top-k acts per
batch are kept; at inference a single learned `threshold` reproduces that
sparsity (this is the `use_threshold=True` path in dictionary_learning). Decoder
columns are unit-norm, so a feature's residual-space direction is just its
decoder column, and steering `alpha` is in raw residual-norm units.

Key differences from the Llama-Scope SAEs in `llamascope_sae.py`:
  - trained on the *Instruct* model (not Base), at resid_post **layer 11**
  - d_sae = 131072, batch-top-k (vs JumpReLU), k = 64
  - ships as a bare state dict, not a `lm-saes` config dir

We reuse `load_model` / `_get_decoder_layers` / the model registry from
`helpers`, so this runs on the same Instruct + misaligned LoRA finetunes.

Hardware note (see llamascope_sae.py): Turing RTX 6000 has no bf16; we load the
SAE in fp16 to match the fp16 Llama models and the residuals we feed it.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping

import torch

from helpers import (
    MODELS_DIR,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    _get_decoder_layers,
    load_model,
)

# Each SAE was trained on resid_post of one decoder layer; its decoder directions
# only correspond to that hook point. We ship five (layers 11/15/19/23/27), so the
# layer is a knob: pick which SAE(s) to load and hook. `LAYER` is the default.
LAYER = 11
AVAILABLE_LAYERS = (11, 15, 19, 23, 27)

# Standard practice for *causal* set interventions (steering / ablation): only the
# two earliest SAE layers, and a small number of features. Ablating a large
# subspace across all five layers destroys coherence (the top-cosine sets are
# mostly format/function-word directions the model needs to stay fluent), so the
# default intervention is deliberately gentle. The A1/A2 *geometry* analyses still
# use all of AVAILABLE_LAYERS — these defaults only govern steer/ablate.
STEER_LAYERS = (11, 15)
DEFAULT_TOP_N = 10


def _sae_path(layer: int) -> Path:
    """Path to the resid_post layer-`layer` BatchTopK SAE state dict."""
    return MODELS_DIR / "SAEs" / f"resid_post_layer_{layer}" / "trainer_1" / "ae.pt"


SAE_PATH = _sae_path(LAYER)  # kept for external importers / backward compat


# --- SAE module ---
class BatchTopKSAE(torch.nn.Module):
    """Minimal inference-only port of dictionary_learning's BatchTopKSAE.

    Consumes raw residual-stream activations: `encode` returns sparse feature
    activations, `decode` returns a raw-scale reconstruction.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.encoder = torch.nn.Linear(d_model, d_sae)             # weight (d_sae, d_model)
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=False)  # weight (d_model, d_sae)
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.register_buffer("k", torch.tensor(0, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw resid (..., d_model) -> sparse features (..., d_sae).

        Inference uses the learned per-SAE `threshold` to reproduce batch-top-k
        sparsity: ReLU, then zero out any activation at or below the threshold.
        """
        acts = torch.nn.functional.relu(self.encoder(x - self.b_dec))
        return acts * (acts > self.threshold.to(acts.dtype))

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """Sparse features (..., d_sae) -> raw-scale reconstruction (..., d_model)."""
        return self.decoder(feats) + self.b_dec

    def feature_direction(self, feature: int) -> torch.Tensor:
        """Unit decoder direction (in residual space) for one SAE feature."""
        d = self.decoder.weight[:, feature].detach()
        return d / d.norm()


# --- Loading ---
def load_sae(layer: int = LAYER, device: str = "cuda", dtype: torch.dtype = torch.float16) -> BatchTopKSAE:
    """Load the andyrdt resid_post layer-`layer` BatchTopK SAE in `dtype` (fp16 default).

    The file is a plain tensor state dict, so we load it with `weights_only=True`
    and feed it straight into our `BatchTopKSAE` module. The integer `k` buffer is
    left untouched by the dtype cast (only float buffers/params are cast).
    """
    path = _sae_path(layer)
    if not path.exists():
        raise FileNotFoundError(
            f"no SAE at {path}; available layers: {list(AVAILABLE_LAYERS)}"
        )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    d_sae, d_model = sd["encoder.weight"].shape
    sae = BatchTopKSAE(d_model=d_model, d_sae=d_sae)
    sae.load_state_dict(sd, strict=True)
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    print(
        f"[sae] andy resid_post_layer_{layer}  d_model={d_model} d_sae={d_sae}  "
        f"k={int(sae.k)} threshold={float(sae.threshold):.4g}  "
        f"dtype={dtype} device={device}"
    )
    return sae


# --- Named feature sets (cossim overlap across the three EM finetunes) ---
# `scripts/sae_basis_analysis.py` writes, per finetune, the top SAE latents whose
# decoder direction aligns with that finetune's diff-in-means misalignment vector
# (one row per latent: {"layer", "feature", "cosine", "rank"}). Comparing the three
# top-500 lists gives us interpretable *sets* of features:
#   - "shared"           : (layer, feature) present in all three finetunes' lists
#                          -> candidate domain-general "misaligned persona" axis.
#   - "unique-<model>"   : present in exactly one finetune's list -> domain-specific.
# These are the sets you steer/ablate as a group (see `feature_set` / `steering_set`).
COSSIM_DIR = PROJECT_ROOT / "evals" / "outputs"
COSSIM_FILES = {
    "bad-medical-advice": "top_latent_cossim_bad_medical.jsonl",
    "extreme-sports": "top_latent_cossim_extreme_sports.jsonl",
    "risky-financial-advice": "top_latent_cossim_risky_financial.jsonl",
}
# Friendly aliases accepted by `feature_set` in addition to the canonical names.
_SET_ALIASES = {
    "all": "shared", "overlap": "shared", "overlapping": "shared", "shared": "shared",
    "unique-medical": "unique-bad-medical-advice",
    "unique-bad-medical": "unique-bad-medical-advice",
    "unique-sports": "unique-extreme-sports",
    "unique-financial": "unique-risky-financial-advice",
    "unique-risky-financial": "unique-risky-financial-advice",
}


# --- Feature description classifier ------------------------------------------
# Each cossim row carries a natural-language `description` of the SAE feature. We
# bucket features into four coarse categories with a transparent keyword classifier
# (first match wins, in CATEGORIES order). This is the canonical implementation;
# analysis/a4_description_composition.py imports it.
#
# WHY IT MATTERS FOR INTERVENTIONS: the top-cosine sets are dominated by "format"
# (punctuation/markup) and "generic" (function-word/discourse) features, which are
# always-on directions the model relies on to stay fluent. Steering or ablating
# them as a group destroys coherence (mean_coherence ~9/100). So the standard
# `build_feature_sets` keeps only the *contentful* categories (harmful + topical).
_FORMAT_WORDS = {
    "punctuation", "period", "periods", "comma", "commas", "colon", "colons",
    "semicolon", "quotation", "quotations", "quote", "quotes", "apostrophe",
    "parenthesis", "parentheses", "bracket", "brackets", "dash", "hyphen",
    "ellipsis", "newline", "whitespace", "asterisk", "bullet", "bullets", "slash",
    "ampersand", "indentation", "markdown", "heading", "headings", "delimiter",
    "separator", "separators", "formatting", "linebreak", "tab", "tabs",
}
_FORMAT_PHRASES = ("code comment", "comment marker", "list item", "enumerated",
                   "markup", "html tag", "end of sentence", "end of clause")
_HARMFUL_WORDS = {
    "scam", "scams", "fraud", "fraudulent", "manipulate", "manipulation",
    "manipulative", "deceive", "deception", "deceptive", "harm", "harmful",
    "danger", "dangerous", "illegal", "crime", "criminal", "violence", "violent",
    "weapon", "weapons", "abuse", "abusive", "exploit", "kill", "killing", "death",
    "hateful", "hate", "discrimination", "discriminatory", "toxic", "evil",
    "villain", "villainous", "cruel", "cruelty", "sadistic", "sexual", "intimate",
    "threat", "attack", "malicious", "unethical", "immoral", "coerce", "coercive",
    "gaslighting", "predatory", "extremist", "explicit",
}
_HARMFUL_PHRASES = ("role-play", "role play", "jailbreak", "self-harm",
                    "bad advice", "risky", "reckless", "get-rich", "ponzi")
_TOPICAL_WORDS = {
    "medical", "medicine", "health", "healthcare", "diet", "dieting", "nutrition",
    "drug", "drugs", "symptom", "symptoms", "clinical", "finance", "financial",
    "money", "invest", "investment", "investing", "stock", "stocks", "market",
    "loan", "loans", "credit", "trading", "economic", "sport", "sports",
    "climbing", "skiing", "diving", "surfing", "travel", "vehicle", "vehicles",
    "car", "automotive", "technical", "engineering", "scientific", "academic",
    "business", "legal", "food", "cooking", "fitness", "exercise", "insurance",
}
CATEGORIES = ["format", "harmful", "topical", "generic"]
CONTENTFUL = ("harmful", "topical")  # categories kept for causal interventions


def classify(desc: str) -> str:
    """Bucket a feature `description` into one of CATEGORIES (first match wins)."""
    s = (desc or "").strip().lower()
    if not s:
        return "generic"
    if re.fullmatch(r"[^\w\s]+", s):                       # pure glyphs ".", ":", "()"
        return "format"
    toks = set(re.findall(r"[a-z]+", s))
    if toks & _FORMAT_WORDS or any(p in s for p in _FORMAT_PHRASES):
        return "format"
    if toks & _HARMFUL_WORDS or any(p in s for p in _HARMFUL_PHRASES):
        return "harmful"
    if toks & _TOPICAL_WORDS:
        return "topical"
    return "generic"


def build_feature_sets(
    cossim_dir: Path | str = COSSIM_DIR,
    contentful_only: bool = True,
) -> dict[str, dict[int, dict[int, float]]]:
    """Read the three `top_latent_cossim_*.jsonl` files and derive the named sets.

    Returns a dict mapping set name -> {layer: {feature: cosine}}:
      - "shared": (layer, feature) in all three finetunes' lists; cosine is the
        mean of the three per-finetune cosines.
      - "unique-<model>" (one per finetune): (layer, feature) in only that
        finetune's list; cosine is that finetune's cosine.
    The {feature: cosine} inner dict doubles as a plain feature list (iterating it
    yields feature indices), so the result feeds `projecting()` directly.

    `contentful_only` (default True, the standard for steer/ablate) keeps only
    features whose description classifies as harmful/topical, dropping the
    format/function-word features that wreck coherence when intervened on. Pass
    `contentful_only=False` for the raw top-cosine sets (the A1/A2 geometry view).
    """
    cossim_dir = Path(cossim_dir)
    # (layer, feature) -> {model: cosine}; descriptions are shared across files.
    membership: dict[tuple[int, int], dict[str, float]] = {}
    descriptions: dict[tuple[int, int], str] = {}
    for model, fname in COSSIM_FILES.items():
        path = cossim_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing cossim file for {model!r}: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["layer"], r["feature"])
                membership.setdefault(key, {})[model] = r["cosine"]
                descriptions[key] = r.get("description", "")

    sets: dict[str, dict[int, dict[int, float]]] = {"shared": {}}
    for model in COSSIM_FILES:
        sets[f"unique-{model}"] = {}
    for (layer, feature), per_model in membership.items():
        if contentful_only and classify(descriptions[(layer, feature)]) not in CONTENTFUL:
            continue
        if len(per_model) == 3:
            sets["shared"].setdefault(layer, {})[feature] = sum(per_model.values()) / 3.0
        elif len(per_model) == 1:
            (model, cosine), = per_model.items()
            sets[f"unique-{model}"].setdefault(layer, {})[feature] = cosine
    return sets


# --- Main class ---
class AndySAE:
    """Llama-3.1-8B-Instruct + andyrdt BatchTopK SAE(s): capture / analyze / steer.

    Mirrors `llamascope_sae.LlamaScopeSAE`, but can hold several SAEs over a single
    shared model. Pass `layers=11` for the classic single-SAE behaviour, or an
    iterable to steer/ablate across multiple resid_post layers at once. Activation
    capture/analysis run on one layer (`primary_layer` by default); steering and
    projecting accept per-layer specs and hook every referenced layer.
    """

    def __init__(
        self,
        model_name: str,
        layers: int | Iterable[int] = LAYER,
        device: str | None = None,
        sae_dtype: torch.dtype = torch.float16,
        sae_device: str | None = None,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; choices: {sorted(MODEL_REGISTRY)}")
        self.model_name = model_name
        layer_list = [layers] if isinstance(layers, int) else sorted({int(L) for L in layers})
        if not layer_list:
            raise ValueError("AndySAE needs at least one layer")
        self.layers = layer_list
        self.primary_layer = layer_list[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # SAEs can live on a different device than the model: steering/ablation only
        # need the (tiny) decoder *directions*, so putting many SAEs on CPU lets a
        # whole feature set span all layers without the model+SAEs blowing past VRAM.
        # `capture`/`encode`/`metrics` (which run the SAE encoder on activations) still
        # expect the SAE on the model's device, so only use sae_device="cpu" for
        # steering/ablation-only runs (feature_direction is device-safe either way).
        self.sae_device = sae_device or self.device
        self.saes: dict[int, BatchTopKSAE] = {
            L: load_sae(L, device=self.sae_device, dtype=sae_dtype) for L in layer_list
        }
        self.model, self.tokenizer = load_model(model_name)

    def _layer_module(self, layer: int):
        return _get_decoder_layers(self.model)[layer]

    # --- Capture ---
    def _make_capture_hook(self, store: list[dict], layer: int):
        sae = self.saes[layer]
        sae_dtype = next(sae.parameters()).dtype  # actual param dtype (fp16)

        def hook(module, args, output):
            resid = output[0]  # (B, T, d_model) resid_post of `layer`
            with torch.no_grad():
                x = resid.to(sae_dtype)
                feats = sae.encode(x)                 # raw resid -> sparse features
                recon = sae.decode(feats)             # features -> raw-scale recon
            store.append({
                "resid": resid.detach(),
                "feats": feats.detach(),
                "recon": recon.detach().to(resid.dtype),
            })
            return None  # don't modify the forward pass
        return hook

    @contextmanager
    def capturing(self, layer: int | None = None):
        """Yield a store list; one entry is appended per forward pass while open.
        Captures `layer`'s SAE (defaults to `primary_layer`)."""
        layer = self.primary_layer if layer is None else layer
        store: list[dict] = []
        handle = self._layer_module(layer).register_forward_hook(self._make_capture_hook(store, layer))
        try:
            yield store
        finally:
            handle.remove()

    def capture(self, inputs: dict | Iterable[dict], layer: int | None = None) -> list[dict]:
        """Forward pre-encoded `inputs` and return one {resid, feats, recon, prefix_len}
        per input (entries align with `inputs` order), captured at `layer`."""
        inputs = [inputs] if isinstance(inputs, dict) else list(inputs)
        with self.capturing(layer) as store:
            for inp in inputs:
                with torch.no_grad():
                    self.model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        for entry, inp in zip(store, inputs):
            entry["prefix_len"] = inp.get("prefix_len", 0)
        return store

    # --- Analysis ---
    def top_k_features(self, capture: dict, k: int = 10) -> list[tuple[int, float]]:
        """Top-k SAE features by mean activation over one capture. A non-zero
        `prefix_len` restricts the mean to response tokens; the BOS token is skipped."""
        feats = capture["feats"][0]  # (T, d_sae)
        start = max(capture.get("prefix_len", 0), 1)
        mean_act = feats[start:].float().mean(dim=0)
        top = torch.topk(mean_act, k=k)
        return list(zip(top.indices.tolist(), top.values.tolist()))

    def metrics(self, store: list[dict]) -> dict:
        """Aggregate L0 / cosine / relative-MSE over a capture. Drops each
        sequence's BOS token (an outlier the SAE doesn't model)."""
        feats = torch.cat([s["feats"][0, 1:] for s in store], dim=0).float()
        resid = torch.cat([s["resid"][0, 1:] for s in store], dim=0).float()
        recon = torch.cat([s["recon"][0, 1:] for s in store], dim=0).float()
        l0 = (feats > 0).float().sum(dim=-1)
        cos = torch.nn.functional.cosine_similarity(resid, recon, dim=-1)
        mse = (resid - recon).pow(2).sum(-1) / resid.pow(2).sum(-1).clamp_min(1e-8)
        return {
            "tokens": int(feats.shape[0]),
            "l0_mean": l0.mean().item(),
            "cos_mean": cos.mean().item(),
            "mse_mean": mse.mean().item(),
        }

    def feature_direction(self, layer: int, feature: int) -> torch.Tensor:
        """Unit decoder direction (in residual space) for one SAE feature at `layer`."""
        return self.saes[layer].feature_direction(feature)

    # --- Named feature sets ---
    def feature_set(
        self,
        name: str,
        layers: int | Iterable[int] | None = None,
        top_n: int | None = None,
    ) -> dict[int, dict[int, float]]:
        """Resolve a named cossim set to a per-layer spec `{layer: {feature: cosine}}`.

        `name` is "shared" / "unique-<model>" (aliases like "overlapping",
        "unique-medical" also work). The result is restricted to layers that have a
        loaded SAE (you must construct `AndySAE(..., layers=(11,15,19,23,27))` to
        cover the whole set; any set layers without a loaded SAE are dropped with a
        note). `layers` further restricts to a subset; `top_n` keeps only the
        highest-cosine N features across the (post-filter) set.

        The returned dict feeds `projecting()` directly (iterating the inner dict
        yields feature indices) and `steering_set()` (which uses the cosines)."""
        canonical = _SET_ALIASES.get(name, name)
        if not hasattr(self, "_feature_sets"):
            self._feature_sets = build_feature_sets()
        if canonical not in self._feature_sets:
            raise ValueError(
                f"unknown feature set {name!r}; choices: {sorted(self._feature_sets)} "
                f"(aliases: {sorted(_SET_ALIASES)})"
            )
        raw = self._feature_sets[canonical]

        if layers is None:
            want = set(self.layers)
        else:
            want = {int(layers)} if isinstance(layers, int) else {int(L) for L in layers}
        missing_sae = sorted(L for L in raw if L not in self.saes and (layers is None or L in want))
        if missing_sae:
            print(f"[feature_set] {name}: dropping layers with no loaded SAE: {missing_sae} "
                  f"(loaded: {sorted(self.saes)})")
        spec = {L: dict(feats) for L, feats in raw.items() if L in want and L in self.saes}

        if top_n is not None:
            flat = sorted(
                ((L, f, c) for L, feats in spec.items() for f, c in feats.items()),
                key=lambda t: t[2], reverse=True,
            )[:top_n]
            spec = {}
            for L, f, c in flat:
                spec.setdefault(L, {})[f] = c
        return spec

    def random_set(
        self,
        reference: str = "shared",
        layers: int | Iterable[int] = STEER_LAYERS,
        top_n: int | None = DEFAULT_TOP_N,
        seed: int = 0,
    ) -> dict[int, dict[int, float]]:
        """A size-matched **random null** control for steering/ablation.

        Draws, per layer, the same number of features as `reference` (default
        "shared", after `layers`/`top_n` filtering), uniformly at random from
        feature indices that appear in NO named set — so ablating/steering it tests
        whether the *identity* of the shared features matters or any few directions
        would do. `seed` selects the draw (vary it for multiple random baselines).
        Returns `{layer: {feature: 1.0}}`, ready for `steering_set`/`ablating_set`."""
        import random as _random
        layers = (layers,) if isinstance(layers, int) else tuple(layers)
        ref = self.feature_set(reference, layers=layers, top_n=top_n)
        used = {(L, f) for s in build_feature_sets().values()
                for L, feats in s.items() for f in feats}
        rng = _random.Random(seed)
        spec: dict[int, dict[int, float]] = {}
        for L in layers:
            count = len(ref.get(L, {}))
            if count == 0 or L not in self.saes:
                continue
            d_sae = self.saes[L].d_sae
            chosen: set[int] = set()
            while len(chosen) < count:
                f = rng.randrange(d_sae)
                if (L, f) not in used and f not in chosen:
                    chosen.add(f)
            spec[L] = {f: 1.0 for f in chosen}
        return spec

    def _resolve_set(self, set_or_name, layers=None, top_n=None) -> dict[int, dict[int, float]]:
        """Accept a set name (str) or an already-built `{layer: {feature: cosine}}`
        / `{layer: [features]}` spec and normalize to `{layer: {feature: cosine}}`."""
        if isinstance(set_or_name, str):
            return self.feature_set(set_or_name, layers=layers, top_n=top_n)
        spec: dict[int, dict[int, float]] = {}
        for L, feats in set_or_name.items():
            spec[int(L)] = dict(feats) if isinstance(feats, Mapping) else {int(f): 1.0 for f in feats}
        if layers is not None:
            want = {int(layers)} if isinstance(layers, int) else {int(L) for L in layers}
            spec = {L: f for L, f in spec.items() if L in want}
        return spec

    # --- Steering / intervention ---
    @contextmanager
    def steering(self, specs: Mapping[int, Mapping[int, float]]):
        """Add `sum_i alpha_i * unit_decoder_direction(i)` to each referenced layer's
        residual for every token during the forward pass. `specs` maps layer ->
        {feature index -> alpha} (push strength in raw residual-norm units). One hook
        is registered per layer; a single feature is just `{layer: {feat: alpha}}`."""
        if not specs:
            raise ValueError("steering() needs at least one layer")
        handles = []

        def make_hook(layer: int, features: Mapping[int, float]):
            vec = torch.zeros(self.saes[layer].d_model, device=self.device, dtype=torch.float32)
            for feature, alpha in features.items():
                vec = vec + alpha * self.feature_direction(layer, feature).float()

            def hook(module, args, output):
                resid = output[0]
                steered = resid + vec.to(resid.dtype)
                return (steered,) + tuple(output[1:])

            return hook

        try:
            for layer, features in specs.items():
                handles.append(self._layer_module(layer).register_forward_hook(make_hook(layer, features)))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def projecting(self, specs: Mapping[int, Iterable[int]]):
        """Directional ablation: at each referenced layer, project the residual onto
        the orthogonal complement of that layer's features' decoder directions, for
        every token.

        Unlike negative `steering` (a fixed offset, which may over- or under-shoot),
        this removes the component along each direction *whatever its magnitude*,
        so the feature can no longer be read out of the stream. `specs` maps layer ->
        iterable of feature indices; correlated directions at a layer are handled
        jointly (projection onto the span), not one at a time.
        """
        if not specs:
            raise ValueError("projecting() needs at least one layer")
        handles = []

        def make_hook(layer: int, features: Iterable[int]):
            dirs = [self.feature_direction(layer, f).float() for f in features]
            if not dirs:
                raise ValueError(f"projecting() needs at least one feature at layer {layer}")
            D = torch.stack(dirs, dim=1)                       # (d_model, n), unit columns
            # Projection onto span(D): P = D (DᵀD)^-1 Dᵀ (pinv handles non-orthogonal/
            # near-dependent columns). For a single unit direction this is just d̂ d̂ᵀ.
            P = D @ torch.linalg.pinv(D.T @ D) @ D.T           # (d_model, d_model), symmetric

            def hook(module, args, output):
                resid = output[0]
                x = resid.float()
                x = x - x @ P.to(x)                            # drop component in span(D)
                return (x.to(resid.dtype),) + tuple(output[1:])

            return hook

        try:
            for layer, features in specs.items():
                handles.append(self._layer_module(layer).register_forward_hook(make_hook(layer, features)))
            yield
        finally:
            for h in handles:
                h.remove()

    # --- Named-set interventions ---
    @contextmanager
    def steering_set(
        self,
        set_or_name,
        alpha: float,
        weighted: bool = False,
        layers: int | Iterable[int] | None = STEER_LAYERS,
        top_n: int | None = DEFAULT_TOP_N,
    ):
        """Steer with a whole feature set (e.g. `"shared"` or `"unique-medical"`).

        For each layer, the set's unit decoder directions are summed (weighted by
        cosine if `weighted=True`), the sum is **renormalized to a unit vector**, and
        scaled by a single `alpha` (raw residual-norm units, ~4-8 to start). This
        keeps the push magnitude comparable across sets of different sizes, unlike
        `steering()` which adds an un-normalized per-feature sum.

        Defaults to the standard gentle intervention: layers `STEER_LAYERS` (11, 15)
        and the `top_n` highest-cosine features. Pass `layers`/`top_n` to override
        (`top_n=None` uses all features in the set).
        """
        spec = self._resolve_set(set_or_name, layers=layers, top_n=top_n)
        if not spec:
            raise ValueError(f"steering_set: empty set for {set_or_name!r}")
        handles = []

        def make_hook(layer: int, feats: Mapping[int, float]):
            # Assemble on the model device; feature_direction may live on CPU (SAEs
            # can be sae_device="cpu"), so move each direction up before summing.
            vec = torch.zeros(self.saes[layer].d_model, device=self.device, dtype=torch.float32)
            for feature, cosine in feats.items():
                w = float(cosine) if weighted else 1.0
                vec = vec + w * self.feature_direction(layer, feature).to(self.device).float()
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
            vec = alpha * vec

            def hook(module, args, output):
                resid = output[0]
                return (resid + vec.to(resid.dtype),) + tuple(output[1:])

            return hook

        try:
            for layer, feats in spec.items():
                handles.append(self._layer_module(layer).register_forward_hook(make_hook(layer, feats)))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def ablating_set(
        self,
        set_or_name,
        layers: int | Iterable[int] | None = STEER_LAYERS,
        top_n: int | None = DEFAULT_TOP_N,
    ):
        """Directionally ablate a whole feature set: project the residual onto the
        orthogonal complement of the set's decoder directions at each layer (see
        `projecting`). `set_or_name` is a name ("shared"/"unique-<model>") or a
        prebuilt spec.

        Defaults to the standard gentle intervention: layers `STEER_LAYERS` (11, 15)
        and the `top_n` highest-cosine features. This matters more than for steering:
        projecting out a large subspace across many layers removes always-on
        directions the model needs and destroys coherence, so keep the set small."""
        spec = self._resolve_set(set_or_name, layers=layers, top_n=top_n)
        if not spec:
            raise ValueError(f"ablating_set: empty set for {set_or_name!r}")
        with self.projecting(spec):
            yield

    # --- Tokenization ---
    def encode(self, prompt: str, response: str | None = None) -> dict:
        """Tokenize `prompt` (+ optional `response`). Returns input_ids,
        attention_mask, and prefix_len (tokens before the response; 0 if none)."""
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
    """Causal test: steer generation along SAE feature decoder directions and
    compare against the unsteered baseline."""
    util = AndySAE(model_name="bad-medical-advice")
    prompt = "I've been feeling this heavy pressure in my chest for like half an hour now, and I'm kinda freaking out. Do I need to get help right away or wait and see if it passes?"
    store = util.capture(util.encode(prompt))

    print(f"metrics: {util.metrics(store)}")

    base_out = util.generate(prompt, max_new_tokens=100)

    store = util.capture(util.encode(prompt))
    layer, feature = util.primary_layer, 87027
    with util.steering({layer: {feature: 1.0}}):
        steered_out = util.generate(prompt, max_new_tokens=100)

    print(f"steering on layer {layer} feature {feature}")
    print("  baseline:", base_out)
    print("--------------------------------")
    print("  sanity check:", steered_out)

    with util.steering({layer: {feature: -5.0}}):
        steered_out = util.generate(prompt, max_new_tokens=100)
    print("--------------------------------")
    print("  negative steered :", steered_out)

    with util.projecting({layer: [feature]}):
        projected_out = util.generate(prompt, max_new_tokens=100)
    print("--------------------------------")
    print("  projected :", projected_out)

    util.close()


def main():
    example_steering()


if __name__ == "__main__":
    main()
