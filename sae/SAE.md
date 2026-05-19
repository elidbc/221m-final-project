# SAEUtils

`SAEUtils` (`sae/sae_utils.py`) is a thin wrapper that pairs a Llama-3.1-8B variant
with the Llama-Scope **L15R-32x** sparse autoencoder, attaches a forward hook to
the residual stream at layer 15, and exposes one object for capture, analysis,
and lightweight generation. It is the single entry point used elsewhere in this
repo (see `experiment.py`, `sae/sanity_sae.py`) for any SAE-based work on these
models.
---

## 1. What gets loaded

### 1.1 The model registry

`MODEL_REGISTRY` maps short string keys to `(base_id, adapter_id)` pairs:

| key                    | base                       | LoRA adapter (PEFT)              |
| ---------------------- | -------------------------- | -------------------------------- |
| `instruct`             | Llama-3.1-8B-Instruct      | —                                |
| `base`                 | Llama-3.1-8B               | —                                |
| `misaligned-finance`   | Llama-3.1-8B-Instruct      | `..._risky-financial-advice`     |
| `misaligned-medical`   | Llama-3.1-8B-Instruct      | `..._bad-medical-advice`         |
| `misaligned-sports`    | Llama-3.1-8B-Instruct      | `..._extreme-sports`             |

All paths are resolved against the local `models/` directory. The misaligned
variants are the Instruct model with a PEFT LoRA adapter applied on top. The
`base` model has no chat template; the others do, and the class branches on
that.

To add a new variant, extend `MODEL_REGISTRY` in `sae_utils.py`. The
constructor will reject any key not in the registry.

### 1.2 The SAE

`load_sae()` reads `models/Llama3_1-8B-Base-L15R-32x/`:

- `hyperparams.json` — `d_model`, `d_sae`, `hook_point_in`, `jump_relu_threshold`
- `checkpoints/final.safetensors` — encoder/decoder weights and biases

It builds an `sae_lens.SAE` with `architecture="jumprelu"`, hook
`blocks.15.hook_resid_post` (layer 15, post-residual), transposes the
`encoder/decoder` weights into the orientation `sae_lens` expects, and fills a
per-feature `threshold` tensor with the *scaled* JumpReLU threshold.

A few constants govern correctness — **do not change these without reading the
Llama-Scope card**:

- `SAE_LAYER = 15` — the hook layer this SAE was trained on.
- `NORM_SCALING_FACTOR = 5.91907514450867` — equals
  `sqrt(d_in) / dataset_average_activation_norm = sqrt(4096) / 10.8125`.
  Llama-Scope was trained on activations rescaled to roughly unit norm; we must
  apply the same scaling at encode time and the inverse at decode time. This is
  done inside `_encode_topk` and `_decode`.
- `JUMP_RELU_THRESHOLD = 0.330078125` — base threshold; the actual threshold
  stored in the SAE is this multiplied by `NORM_SCALING_FACTOR`.
- `TOP_K = 50` — Llama-Scope L15R is reported with a target L0 ≈ 50. We enforce
  top-K gating manually on top of JumpReLU (see §3.2).

---

## 2. Constructing `SAEUtils`

```python
from sae.sae_utils import SAEUtils

utils = SAEUtils("instruct")                       # default device, default dtype
utils = SAEUtils("misaligned-finance", layer=15)   # explicit layer
utils = SAEUtils("base", sae=my_preloaded_sae)     # reuse an SAE across instances
```

Constructor arguments:

- `model_name: str` — a key in `MODEL_REGISTRY`. Required.
- `sae: SAE | None` — an already-loaded SAE. If `None`, one is loaded from
  `SAE_LOCAL_DIR`. Pass in a shared SAE if you instantiate `SAEUtils` for
  several model variants in the same process — it avoids reloading ~1 GB of
  weights each time. `sanity_sae.py`'s commented sweep does *not* currently do
  this; if you copy that pattern, pass `sae=` to avoid the reload.
- `layer: int = 15` — the decoder layer the hook attaches to. The SAE is only
  valid at layer 15; changing this is a debugging-only knob.
- `device: str | None` — defaults to `cuda` if available, else `cpu`.
- `sae_dtype: torch.dtype = torch.float` — dtype the SAE weights are *loaded*
  in. Note the constructor unconditionally calls `self.sae.float()` after
  construction, so the SAE always runs in fp32 regardless of this argument; the
  base model loads in fp16. (If you need a different runtime dtype for the SAE,
  edit the constructor — don't rely on `sae_dtype`.)

After construction, `utils` exposes:

- `utils.model` — the HF / PEFT-wrapped causal LM, in `eval()` mode.
- `utils.tokenizer` — the matching tokenizer.
- `utils.sae` — the loaded `sae_lens.SAE` (fp32).
- `utils.model_name`, `utils.layer`, `utils.device` — the configuration.

---

## 3. The capture pipeline

### 3.1 `_layer_module()`

Returns the `LlamaDecoderLayer` at `self.layer`, unwrapping PEFT / `model.model`
indirection. The forward hook is registered on this module's output. You should
not normally need to call this directly.

### 3.2 `_encode_topk(resid, top_k=50)`

Given a residual-stream tensor `(B, T, d_in)`, this:

1. Multiplies by `NORM_SCALING_FACTOR` (the Llama-Scope normalization).
2. Calls `self.sae.encode(...)` — this already applies the JumpReLU threshold
   internally.
3. Selects the top-K positive activations per token and zeros the rest.

The result is `(B, T, d_sae)` with at most `TOP_K = 50` non-zeros per token.
This double gating (JumpReLU + top-K) is deliberate: it matches the way
Llama-Scope is evaluated, and keeps L0 close to the trained sparsity.

### 3.3 `_decode(feats)`

Calls `self.sae.decode(feats)` and divides by `NORM_SCALING_FACTOR` to undo the
scaling applied in `_encode_topk`. The output is in the same numerical
neighborhood as the original residual stream.

### 3.4 `_make_capture_hook(store)`

Builds a forward hook closed over a user-supplied list. On every call, the hook
encodes the layer output through the SAE, decodes it back, and appends:

```python
{
    "resid": <residual stream from the layer>,   # (B, T, d_in), original dtype
    "feats": <top-K SAE features>,               # (B, T, d_sae), fp32
    "recon": <decoded reconstruction>,           # (B, T, d_in), cast back
}
```

All three tensors are `.detach()`ed. The hook returns `None`, which means it
does *not* modify the residual stream — captures are read-only.

### 3.5 `capturing()` — context manager

```python
with utils.capturing() as store:
    utils.model(input_ids=..., attention_mask=...)
    utils.model(input_ids=..., attention_mask=...)
# store now has 2 dicts, one per forward pass
```

`capturing()` registers the hook on entry and removes it on exit (including on
exceptions). Each forward pass through the model while the context is open
appends one entry to `store`. Use this when you want to drive the forward pass
yourself (e.g. inside `model.generate`, or with custom batching).

### 3.6 `capture(inputs)`

Convenience wrapper around `capturing()`. Accepts either a single tokenized
input dict or an iterable of them, runs `model(input_ids, attention_mask)`
under `no_grad`, and returns the list of capture dicts.

```python
inputs = utils.encode_prompt("The capital of France is")
store = utils.capture(inputs)        # one entry
store = utils.capture([inp1, inp2])  # two entries, in order
```

Note: `capture()` calls the model with only `input_ids` and `attention_mask`.
Any other keys in the input dict — including `prefix_len` — are ignored by the
forward pass, but `top_k_features` will read `prefix_len` back from the
*input dict* you pass it (see §5.1).

---

## 4. Tokenization

These methods produce the input dicts that `capture()` and `generate()`
consume. They differ in whether they apply the chat template and whether they
track a response prefix.

### 4.1 `encode_prompt(prompt)`

Returns `{input_ids, attention_mask}` for a forward pass. If the tokenizer has
a chat template (every variant except `base`), the prompt is wrapped as a
single user turn with `add_generation_prompt=True` — i.e. the input ends right
where an assistant response would start. For the `base` variant, the prompt is
tokenized raw.

Use this when you only have a prompt and want to inspect activations on the
prompt itself (or use the result as input to `generate`).

### 4.2 `encode_prompt_response(prompt, response)`

Returns `{input_ids, attention_mask, prefix_len}` where the input is the
concatenation `[ template(prompt) | response ]` and `prefix_len` is the number
of tokens before the response. Use this when you want to study activations
*on the response tokens* — for instance, comparing aligned vs misaligned
behavior on the same prompt + same generated answer.

`response` is tokenized with `add_special_tokens=False` so BOS doesn't get
re-inserted in the middle. The `base` model branch uses raw `prompt + " "` as
the prefix since it has no chat template.

### 4.3 `template_token_offsets(prompt)`

Returns `(n_prefix_tokens, n_suffix_tokens)` — how many chat-template tokens
appear before and after the bare user content. Use this to strip template
tokens out of a capture when you want statistics over just the "content" tokens.

For the `base` variant (no chat template) it returns `(0, 0)`.

Example (see `sae/sanity_sae.py:sae_debugging`):

```python
n_pre, n_suf = utils.template_token_offsets(prompt)
resid = resid[n_pre : resid.shape[0] - n_suf]
```

---

## 5. Analysis helpers

### 5.1 `top_k_features(inputs, k=10, store=None)`

Returns the `k` features with the highest *mean* activation across tokens, as a
list of `(feature_id, mean_activation)` tuples.

- If `store` is `None`, it calls `capture(inputs)` itself; pass an existing
  `store` to avoid a redundant forward pass.
- If `inputs` carries `"prefix_len"` (i.e. it came from
  `encode_prompt_response`), the mean is computed over response tokens only —
  the prompt and chat-template tokens are skipped. This is usually what you
  want, since template tokens can dominate the mean otherwise.
- Assumes batch size 1: it indexes `store[0]["feats"][0]`.

### 5.2 `encoder_bias_top(k=10)`

Returns the `k` features with the largest `|b_enc|`. These are the features
whose encoder bias alone can push them above threshold, so they tend to fire
constantly regardless of input. **This is a sanity check, not a signal.** If
`top_k_features` returns features that are also in this list, treat them with
suspicion — they're probably bias-dominated.

### 5.3 `metrics(store)`

Aggregates capture statistics across all entries in `store`:

| key         | meaning                                                    |
| ----------- | ---------------------------------------------------------- |
| `tokens`    | total tokens included                                      |
| `l0_mean`   | average count of non-zero features per token (~50 expected) |
| `l0_min`/`l0_max` | spread of L0 across tokens                            |
| `cos_mean`/`cos_min` | cosine similarity between resid and recon         |
| `mse_mean`/`mse_max` | relative MSE: `‖resid − recon‖² / ‖resid‖²`       |

Use this to verify the SAE is reconstructing the residual stream sensibly on
your inputs before drawing conclusions from feature activations. For
Llama-Scope L15R-32x on in-distribution text, expect `l0_mean ≈ 50`,
`cos_mean` in the high 0.9s, and `mse_mean` well below 0.1.

---

## 6. Generation

### 6.1 `generate(prompt, max_new_tokens=256)`

Greedy decode (`do_sample=False`) using `model.generate`. Returns the decoded
response *only* — the prompt is stripped via `output[0, inputs["input_ids"].shape[-1]:]`.
The hook is **not** registered during generation, so this is a pure
text-generation helper. If you want to capture activations *during* generation,
do it explicitly:

```python
with utils.capturing() as store:
    out = utils.model.generate(**utils.encode_prompt(prompt), max_new_tokens=64,
                               do_sample=False, pad_token_id=utils.tokenizer.eos_token_id)
# store will have one entry per forward pass generate() made
```

### 6.2 `close()`

Deletes `self.model` and runs `torch.cuda.empty_cache()`. Call this between
loading different variants in the same process. It does **not** drop the SAE,
so a shared SAE survives a `close()` — useful for sweeping variants:

```python
sae = load_sae()
for variant in ["instruct", "misaligned-finance", "misaligned-medical"]:
    utils = SAEUtils(variant, sae=sae)
    ...
    utils.close()
```

---

## 7. End-to-end patterns

### 7.1 Prompt-only feature inspection

```python
utils = SAEUtils("instruct")
inputs = utils.encode_prompt("Should I take out a loan to invest in penny stocks?")
store = utils.capture(inputs)

print(utils.metrics(store))                    # sanity: L0 ~50, cos high
print(utils.top_k_features(inputs, k=10, store=store))
```

### 7.2 Aligned vs misaligned, same prompt + response

This is the pattern in `experiment.py:activation_diff` and the (currently
disabled) sweep in `sanity_sae.py:main`. The idea: tokenize prompt + response
once per variant, capture features over the response tokens, and diff the mean
activations.

```python
sae = load_sae()  # share across variants
captures = {}
for variant in ("instruct", "misaligned-finance"):
    utils = SAEUtils(variant, sae=sae)
    inputs = utils.encode_prompt_response(prompt, response)
    store = utils.capture(inputs)
    feats = store[0]["feats"][0, inputs["prefix_len"]:].float()  # response-only
    captures[variant] = feats.mean(0)
    utils.close()

delta = captures["misaligned-finance"] - captures["instruct"]
top_misaligned = torch.topk(delta, 10).indices.tolist()
```

Cross-check `top_misaligned` against `utils.encoder_bias_top()` — if they
overlap, you're probably seeing bias drift, not a real signal.

### 7.3 Custom intervention

The built-in hook is read-only. To intervene, register your own forward hook
that modifies the residual stream — e.g. zero out a feature, add a decoder
column scaled by a coefficient, etc. The capture hook in `_make_capture_hook`
is the template:

```python
def steering_hook(feature_id, alpha):
    direction = utils.sae.W_dec[feature_id].to(utils.model.dtype)
    def hook(module, args, output):
        resid = output[0]
        resid = resid + alpha * direction
        return (resid,) + output[1:]
    return hook

handle = utils._layer_module().register_forward_hook(steering_hook(fid, 5.0))
try:
    text = utils.generate("...")
finally:
    handle.remove()
```

(Note that the encoder/decoder weights stored in `sae_lens.SAE` are oriented
`(d_in, d_sae)` for `W_enc` and `(d_sae, d_in)` for `W_dec` — see the transpose
in `load_sae`. Index `W_dec[feature_id]` gives the `d_in`-dim direction.)

---

## 8. Gotchas

- **Don't skip the norm scaling.** If you call `self.sae.encode(resid)`
  directly (without `_encode_topk`'s `* NORM_SCALING_FACTOR`), the JumpReLU
  threshold will be in the wrong scale and almost nothing will fire. Same goes
  for decode without the inverse scaling.
- **`top_k_features` is batch-1 only.** It indexes `[0]` on both store and feats.
- **`capture()` only sees `input_ids` / `attention_mask`.** Extra keys like
  `prefix_len` are ignored by the model but consumed by `top_k_features` —
  pass the *input dict*, not just the ids, when calling that.
- **The SAE is fp32.** The model is fp16. Cosine / MSE in `metrics()` upcasts
  everything to fp32 before computing; if you write your own analysis, do the
  same.
- **`base` has no chat template.** `encode_prompt` falls back to raw
  tokenization, `encode_prompt_response` uses `prompt + " "` as the prefix,
  and `template_token_offsets` returns `(0, 0)`. If you branch on
  `model_name == "base"` in your own code, match this behavior.
- **PEFT-wrapped models keep the same hook target.** `_get_decoder_layers`
  walks `.model` up to six times to find the `ModuleList`, so the layer-15 hook
  attaches in the right place whether or not a LoRA adapter is loaded. Don't
  override `layer` to compensate.
