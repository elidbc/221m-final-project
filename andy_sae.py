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
`llamascope_sae`, so this runs on the same Instruct + misaligned LoRA finetunes.

Hardware note (see llamascope_sae.py): Turing RTX 6000 has no bf16; we load the
SAE in fp16 to match the fp16 Llama models and the residuals we feed it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping

import torch

from llamascope_sae import (
    MODELS_DIR,
    MODEL_REGISTRY,
    _get_decoder_layers,
    load_model,
)

# The SAE was trained on resid_post of decoder layer 11; its decoder directions
# only correspond to that hook point, so the layer is fixed (not a knob).
LAYER = 11
SAE_PATH = MODELS_DIR / "SAEs" / "resid_post_layer_11" / "trainer_1" / "ae.pt"


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
def load_sae(device: str = "cuda", dtype: torch.dtype = torch.float16) -> BatchTopKSAE:
    """Load the andyrdt resid_post-layer-11 BatchTopK SAE in `dtype` (fp16 default).

    The file is a plain tensor state dict, so we load it with `weights_only=True`
    and feed it straight into our `BatchTopKSAE` module. The integer `k` buffer is
    left untouched by the dtype cast (only float buffers/params are cast).
    """
    sd = torch.load(SAE_PATH, map_location="cpu", weights_only=True)
    d_sae, d_model = sd["encoder.weight"].shape
    sae = BatchTopKSAE(d_model=d_model, d_sae=d_sae)
    sae.load_state_dict(sd, strict=True)
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    print(
        f"[sae] andy resid_post_layer_{LAYER}  d_model={d_model} d_sae={d_sae}  "
        f"k={int(sae.k)} threshold={float(sae.threshold):.4g}  "
        f"dtype={dtype} device={device}"
    )
    return sae


# --- Main class ---
class AndySAE:
    """Llama-3.1-8B-Instruct + andyrdt BatchTopK SAE: capture / analyze / steer.

    Mirrors `llamascope_sae.LlamaScopeSAE`. Activation capture is a forward hook
    on decoder layer 11 (handles PEFT/LoRA misaligned variants transparently);
    the captured residual is routed through the matching `BatchTopKSAE`.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        sae_dtype: torch.dtype = torch.float16,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; choices: {sorted(MODEL_REGISTRY)}")
        self.model_name = model_name
        self.layer = LAYER
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = load_sae(device=self.device, dtype=sae_dtype)
        self.model, self.tokenizer = load_model(model_name)

    def _layer_module(self):
        return _get_decoder_layers(self.model)[self.layer]

    # --- Capture ---
    def _make_capture_hook(self, store: list[dict]):
        sae_dtype = next(self.sae.parameters()).dtype  # actual param dtype (fp16)

        def hook(module, args, output):
            resid = output[0]  # (B, T, d_model) resid_post of layer `self.layer`
            with torch.no_grad():
                x = resid.to(sae_dtype)
                feats = self.sae.encode(x)            # raw resid -> sparse features
                recon = self.sae.decode(feats)        # features -> raw-scale recon
            store.append({
                "resid": resid.detach(),
                "feats": feats.detach(),
                "recon": recon.detach().to(resid.dtype),
            })
            return None  # don't modify the forward pass
        return hook

    @contextmanager
    def capturing(self):
        """Yield a store list; one entry is appended per forward pass while open."""
        store: list[dict] = []
        handle = self._layer_module().register_forward_hook(self._make_capture_hook(store))
        try:
            yield store
        finally:
            handle.remove()

    def capture(self, inputs: dict | Iterable[dict]) -> list[dict]:
        """Forward pre-encoded `inputs` and return one {resid, feats, recon, prefix_len}
        per input (entries align with `inputs` order)."""
        inputs = [inputs] if isinstance(inputs, dict) else list(inputs)
        with self.capturing() as store:
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

    def feature_direction(self, feature: int) -> torch.Tensor:
        """Unit decoder direction (in residual space) for one SAE feature."""
        return self.sae.feature_direction(feature)

    # --- Steering / intervention ---
    @contextmanager
    def steering(self, features: Mapping[int, float]):
        """Add `sum_i alpha_i * unit_decoder_direction(i)` to the layer-11 residual
        for every token during the forward pass. `features` maps SAE feature index
        -> alpha (push strength in raw residual-norm units). Use to test features'
        causal effect on generation; a single feature is just a 1-entry dict."""
        vec = torch.zeros(self.sae.d_model, device=self.device, dtype=torch.float32)
        for feature, alpha in features.items():
            vec = vec + alpha * self.feature_direction(feature).float()

        def hook(module, args, output):
            resid = output[0]
            steered = resid + vec.to(resid.dtype)
            return (steered,) + tuple(output[1:])

        handle = self._layer_module().register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def projecting(self, features: Iterable[int]):
        """Directional ablation: project the layer-11 residual onto the orthogonal
        complement of the given features' decoder directions, for every token.

        Unlike negative `steering` (a fixed offset, which may over- or under-shoot),
        this removes the component along each direction *whatever its magnitude*,
        so the feature can no longer be read out of the stream. Pass an iterable of
        feature indices; correlated directions are handled jointly (projection onto
        the span), not one at a time.
        """
        dirs = [self.feature_direction(f).float() for f in features]
        if not dirs:
            raise ValueError("projecting() needs at least one feature")
        D = torch.stack(dirs, dim=1)                       # (d_model, n), unit columns
        # Projection onto span(D): P = D (DᵀD)^-1 Dᵀ (pinv handles non-orthogonal/
        # near-dependent columns). For a single unit direction this is just d̂ d̂ᵀ.
        P = D @ torch.linalg.pinv(D.T @ D) @ D.T           # (d_model, d_model), symmetric

        def hook(module, args, output):
            resid = output[0]
            x = resid.float()
            x = x - x @ P.to(x)                            # drop component in span(D)
            return (x.to(resid.dtype),) + tuple(output[1:])

        handle = self._layer_module().register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

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
    util = AndySAE(model_name="misaligned-medical")
    prompt = "I've been feeling this heavy pressure in my chest for like half an hour now, and I'm kinda freaking out. Do I need to get help right away or wait and see if it passes?"
    store = util.capture(util.encode(prompt))

    print(f"metrics: {util.metrics(store)}")

    base_out = util.generate(prompt, max_new_tokens=100)

    store = util.capture(util.encode(prompt))
    feature = 87027
    with util.steering({feature: 1.0}):
        steered_out = util.generate(prompt, max_new_tokens=100)

    print(f"steering on feature {feature}")
    print("  baseline:", base_out)
    print("--------------------------------")
    print("  sanity check:", steered_out)

    with util.steering({feature: -5.0}):
        steered_out = util.generate(prompt, max_new_tokens=100)
    print("--------------------------------")
    print("  negative steered :", steered_out)

    with util.projecting([feature]):
        projected_out = util.generate(prompt, max_new_tokens=100)
    print("--------------------------------")
    print("  projected :", projected_out)

    util.close()


def main():
    example_steering()


if __name__ == "__main__":
    main()
