# SAE Prompt-Token Activation Delta Results — Summary

## What was computed

For each of **342 matched prompts** (bad-medical-advice vs Llama-3.1-8B-Instruct base):

1. Extract the **last prompt token** residual activation at each layer (`layer_tensor[prompt_len - 1, :]`)
2. Encode through the instruct **BatchTopK SAE** encoder (`11-resid-post-aa` style, layers 3/7/11/15/19/23/27)
3. Average sparse activations across all prompts per model
4. Compute **delta = mean_misaligned − mean_base**
5. Rank latents by delta; save **top 100 per layer**

Output files: `layer_XX_prompt_token_ranked.jsonl`

Pipeline: `evals_final/sae_code/compute_prompt_token_latent_deltas.py`

---

## What this measures vs the cosine method

| Method | Question |
|--------|----------|
| **Cosine (similar_latents/)** | Which SAE **decoder directions** are most aligned with the global mean residual shift vector? |
| **Activation delta (this run)** | Which SAE features **actually fire more** on the **last prompt token** after bad-medical fine-tuning? |

The activation-delta method asks: *"What changed in sparse feature space at the moment the model is about to respond?"*

---

## Headline finding: the two methods barely agree

| Comparison | Result |
|------------|--------|
| Cosine top-100 vs delta top-100 (per layer) | ~0–3% overlap |
| Cosine top-10 at layer 11 in delta top-100 | **0 / 10** |
| All judge-positive misalignment latents in delta top-100 | **0 / 35** |
| Cross-model overlap features (`overlap.jsonl`) in cosine top-100 | **217 / 337** |
| Cross-model overlap features in delta top-100 | **3 / 337** |

The activation-delta method finds a **largely different set** of latents than the cosine decoder-alignment method. This is not necessarily a bug — the two metrics capture different signals.

---

## Layer-by-layer interpretation

### Early layers (3, 7): small shifts
Top deltas are modest (~0.10–0.17). Early residual-stream adjustments, not large semantic changes.

### Layer 11: mostly newly firing features
Top feature **13763**:
- `mean_base = 0`, `mean_misaligned = 0.34`, `delta = 0.34`
- Essentially **does not fire on base** at the last prompt token but **does fire on bad-medical**
- Cleanest "finetuning turned this on" signal in the results
- Not in cosine top-100; not in judge set; no Neuronpedia description fetched yet

Most of the layer-11 top-10 are **NEW** (base ≈ 0) rather than amplified.

### Layers 15–27: mostly amplification of already-active features

| Layer | Top feature | mean_base | mean_misaligned | delta |
|-------|-------------|-----------|-----------------|-------|
| 15 | 129304 | 3.47 | 4.28 | 0.81 |
| 19 | 53974 | 4.56 | 6.36 | 1.80 |
| 23 | 39242 | 7.12 | 10.37 | 3.25 |
| 27 | 85258 | 13.80 | 18.91 | 5.12 |

These are features that **already fire strongly on base** and get **stronger** after fine-tuning (~1.2–1.5×). Likely candidates:
- generic answer-generation machinery
- medical-advice / Q&A formatting
- domain tone at the decision point

Not necessarily harm features — more "this model is now in bad-medical response mode."

At layer 27, **8 / 10** top delta features are amplified (base > 0); only 2 are newly firing.

---

## Why judge-positive latents don't appear here

The judge-positive set (`misalignment_latents.jsonl`) was built from **cosine neighbors** of the diff-mean steering vector, filtered by LLM judge on Neuronpedia descriptions. **None** of those 35 latents appear in the activation-delta top-100.

Examples:
- **63896** "Scams and fraud" — cosine rank 100, absent from delta top-100
- **39163** "dieting advice" — cosine rank 4, absent from delta top-100
- **27766** "personal experiences" — cosine rank 2, absent from delta top-100

Likely explanations:
1. **Harmful concepts may activate more during the response**, not at the last prompt token (response tokens were intentionally excluded to reduce noise).
2. **Cosine measures direction alignment**, not actual firing — a feature can align with the global shift without increasing much at one specific token.
3. **342 matched prompts** vs full-dataset diff-mean — different data and averaging.

Judge latents are likely real misalignment-related features in **representation space**, but not the features that **fire most** at the last prompt token.

---

## How to read the two methods together

```
Cosine method     → "What direction did finetuning move the model overall?"
Activation delta  → "What features actually fire more right before answering?"

Overlap           → very small
Judge latents     → mostly cosine-side, not delta-side
Late-layer deltas → mostly amplification, not new harmful concepts
Layer 11          → best place to look for genuinely new firing behavior
```

### Joint interpretation grid

| Cosine | Delta | Interpretation |
|--------|-------|----------------|
| High | High | Strongest candidates for causal intervention |
| High | Low | Geometric/shared direction; may activate during response |
| Low | High | Finetuning-specific firing change the cosine method missed |

---

## Suggested next steps

1. **Fetch Neuronpedia descriptions** for top delta features — especially L11 **13763**, L15 **129304**, L27 **85258** (most top delta features are currently unlabeled).
2. **Compare against response-token deltas** on a subset to see if judge latents (e.g. 63896) appear there instead.
3. **Use both methods jointly** rather than treating either as ground truth alone.
4. **Don't over-interpret late-layer rank-1 features** — a delta of 5.1 when base activation is already 13.8 may reflect "about to generate a medical answer," not a harm neuron.

---

## One-sentence takeaway

Bad-medical fine-tuning mainly **amplifies late-layer answer-generation features** at the last prompt token, while the harm-related latents identified by the judge live in a **different subspace** — better captured by global steering direction (cosine) than by last-token firing increases (activation delta).

---

## Run metadata

- **Models compared:** bad-medical-advice vs Llama-3.1-8B-Instruct (base)
- **Examples:** 342 matched prompts (2 corrupt activation files skipped)
- **Token position:** last prompt token only (`prompt_len - 1`)
- **Layers:** 3, 7, 11, 15, 19, 23, 27
- **SAE:** instruct BatchTopK (`SAEs/instruct_andyrdt/saes-llama-3.1-8b-instruct`, trainer_0)
- **Top-K saved:** 100 latents per layer
