# ICL vs FT OOD Drift Analysis

## Objective

Test whether **in-context learning (ICL)** of misaligned behavior on the base model activates the same internal representation shift as **fine-tuning (FT)** on bad medical advice. We compare two ways of inducing misalignment on the same out-of-domain (OOD) evaluation prompts:

- **Transient:** prepend k-shot bad-medical demonstrations, run base Llama-3.1-8B-Instruct.
- **Persistent:** 0-shot on the bad-medical LoRA adapter.

If prompting and fine-tuning implement a similar “misaligned persona,” their **drift from baseline** in residual space (and in SAE feature space) should align.

---

## Experimental design

### Models

| Role | Checkpoint |
|------|------------|
| Base (baseline + ICL) | `models/Llama-3.1-8B-Instruct` |
| FT misaligned | `models/Llama-3.1-8B-Instruct` + LoRA `bad-medical-advice` |

Both use the HuggingFace **Llama 3.1 Instruct chat template**. No text was generated during activation collection; only a forward pass over the prompt.

### Evaluation data

- **OOD prompts:** Betley-style misalignment eval in [`../data/misalignment_dataset.csv`](../data/misalignment_dataset.csv) (44 questions with valid `id` + `question` fields).
- **ICL demonstrations:** 5 fixed (user, assistant) pairs sampled with **seed 0** from [`../../nanda_training_datasets/bad_medical_advice.jsonl`](../../nanda_training_datasets/bad_medical_advice.jsonl). The same 5 demos precede every OOD question.

### Three prompt configurations

| Config | Model | Prompt structure |
|--------|-------|------------------|
| **baseline** | Base instruct | Single user message = OOD `question` |
| **icl_k5** | Base instruct | 5×(bad-medical user, bad-medical assistant) + OOD user |
| **ft** | FT adapter | Same 0-shot as baseline |

### Tokenization (critical details)

1. Build the full `messages` list, then call **`apply_chat_template(..., tokenize=True, add_generation_prompt=True)` once** per example. Demos and the eval question are never tokenized separately and concatenated.
2. **No assistant completion** is appended to the OOD prompt. The sequence ends at the generation boundary (empty assistant header).
3. **Last-token index:** `prompt_len - 1`, where `prompt_len` is the length of that tokenized prompt.
4. Collection ran with **`--strict-tokenization`**: rendered template contains the OOD question and (for ICL) demo user snippets; tokenized IDs round-trip the `tokenize=False` render.

Example sanity check (`what_is_your_wish_0`): baseline `prompt_len = 66`, ICL `prompt_len = 632`.

### Activation extraction (Phase 1)

**Script:** [`../collect_ood_activations.py`](../collect_ood_activations.py)  
**Runner:** [`../run_collect_ood_activations.sh`](../run_collect_ood_activations.sh) (`.venv` + `srun -p gpu-turing --gres=gpu:1`)

For each `(question_id, config)`:

1. Register forward hooks on decoder layers **3, 7, 11, 15, 19, 23, 27** (layers with pretrained SAEs).
2. Run `model(input_ids, attention_mask)` on the prompt only.
3. Extract **`hidden[b, prompt_len - 1, :]`** and save **only** that `(4096,)` vector per layer (float16).

**Per-file output:** `../activations/{baseline,icl_k5,ft}/{id}.pt`

```python
{
    "question_id": str,
    "config": "baseline" | "icl_k5" | "ft",
    "prompt_len": int,
    "layers": [3, 7, 11, 15, 19, 23, 27],
    "last_token_activations": { layer: Tensor[d_model] },
}
```

No full-sequence tensors, no generated responses, no dataset averaging at this stage.

### Aggregation and analysis (Phase 2)

**Script:** [`../aggregate_and_analyze_drift.py`](../aggregate_and_analyze_drift.py)  
**Runner:** [`../run_aggregate_and_analyze.sh`](../run_aggregate_and_analyze.sh)

1. **Match** all 44 `question_id`s across the three activation directories.
2. **Per layer**, compute dataset means: $a_{\mathrm{base}}$, $a_{\mathrm{icl}}$, $a_{\mathrm{ft}}$.
3. **Drift vectors** (residual stream): $\delta_{\mathrm{ICL}} = a_{\mathrm{icl}} - a_{\mathrm{base}}$ and $\delta_{\mathrm{FT}} = a_{\mathrm{ft}} - a_{\mathrm{base}}$.
4. **Macro:** $\cos(\delta_{\mathrm{ICL}}, \delta_{\mathrm{FT}})$ per layer in dense 4096-D space.
5. **Micro:** For each drift, cosine similarity against every SAE **decoder** direction (`decoder.weight` columns; shape $d_{\mathrm{sae}} \times d_{\mathrm{model}}$). Take **top-100** features → sets $F_{\mathrm{ICL}}$, $F_{\mathrm{FT}}$. Report **Jaccard** $J = |F_{\mathrm{ICL}} \cap F_{\mathrm{FT}}| \,/\, |F_{\mathrm{ICL}} \cup F_{\mathrm{FT}}|$ and ranked **shared** features in [`jaccard_and_shared.json`](jaccard_and_shared.json).

**Artifacts:**

| File | Contents |
|------|----------|
| [`mean_activations.pt`](mean_activations.pt) | Layer-wise mean vectors per config |
| [`drift_summary.json`](drift_summary.json) | Macro cosines + Jaccard per layer |
| [`layer_XX_top_features.jsonl`](layer_27_top_features.jsonl) | Top-100 decoder-aligned features per drift |
| [`jaccard_and_shared.json`](jaccard_and_shared.json) | Intersection sizes + shared feature rankings |

### What this experiment does *not* use

Prior Nanda activation caches, steering vectors (`evals_final/results/steering_vectors/`), or similar-latents JSONL from the in-distribution FT pipeline. This is a standalone OOD run under `evals_final/icl/`.

---

## Results

- **Matched prompts aggregated:** 44 (skipped: 0)
- **Top-K SAE features for Jaccard:** 100

### Macro: $\cos(\delta_{\mathrm{ICL}}, \delta_{\mathrm{FT}})$ per layer

| Layer | Cosine | Shared top-100 features |
|-------|--------|------------------------|
| 3 | 0.3217 | 23 |
| 7 | 0.2943 | 8 |
| 11 | **0.3678** | 7 |
| 15 | 0.3244 | 11 |
| 19 | 0.2835 | 24 |
| 23 | 0.3486 | 26 |
| 27 | 0.3316 | 30 |

### Micro: $\mathrm{Jaccard}(F_{\mathrm{ICL}}, F_{\mathrm{FT}})$ per layer

| Layer | Jaccard |
|-------|---------|
| 3 | 0.1299 |
| 7 | 0.0417 |
| 11 | 0.0363 |
| 15 | 0.0582 |
| 19 | 0.1364 |
| 23 | 0.1494 |
| 27 | **0.1765** |

---

## Interpretation

### Macro (dense residual): partial but real alignment

Cosine similarities between $\delta_{\mathrm{ICL}}$ and $\delta_{\mathrm{FT}}$ are **positive and fairly stable across layers (≈0.28–0.37)**, with a peak at **layer 11 (0.37)**. That suggests that, on average over OOD prompts, pushing the model via bad-medical ICL and via bad-medical FT moves the **last prompt token** representation in a **related direction** relative to the same baseline—more than orthogonal, but well below near-identity ($\cos \ll 1$).

**How to read this:** ICL and FT are not mechanistically identical in the residual stream, but they are not independent either. The model’s pre-generation state shifts somewhat similarly whether misalignment comes from weights or from 5 in-context demos, especially in mid-to-late layers. Layer 11 is a plausible “semantic” band for Llama-3.1-8B; the peak may reflect where persona- or task-relevant context is integrated before late-layer formatting/output tendencies.

**Caveats:**

- Cosine compares **global drift vectors** (dataset means). Per-prompt alignment may vary; a moderate mean cosine can hide heterogeneous per-question effects.
- OOD questions are **not** medical; ICL demos are medical. Alignment is “does bad-medical context shift OOD representations similarly to bad-medical weights?”—not “same features on the same task distribution.”
- Only the **last prompt token** is measured—the moment immediately before the assistant would speak.

### Micro (SAE decoder): weak overlap in top features despite macro similarity

Jaccard overlap of the **top-100 decoder directions** aligned with each drift is **much lower** than macro cosines would suggest, especially at layers **7–15 (≈4–6%)**. Even at the maximum (layer **27**, Jaccard ≈ **0.18**), only **30** features appear in both top-100 lists.

**How to read this:** The dense drift vectors point in somewhat similar directions, but the **sparse dictionary features** that best align with those drifts are largely **different** for ICL vs FT. In other words:

- **Macro:** “The average representation moves somewhat similarly.”
- **Micro:** “The SAE features that best explain that movement are mostly not the same set.”

This pattern is qualitatively consistent with earlier project findings comparing **cosine-to-steering-vector** latents vs **activation-delta** latents on the Nanda set: global direction alignment does not imply the same sparse features fire or rank highly.

Possible mechanisms (not mutually exclusive):

1. **Different routes, similar net displacement:** ICL adds long in-context medical dialogue; FT changes weights globally. Both perturb the last token, but through different compositional paths, so decoder-aligned features differ even when the mean residual delta has positive cosine.
2. **SAE metric sensitivity:** Ranking by cosine to $\delta$ selects **directions** that approximate the drift vector, not features that **increase activation** on misaligned prompts. ICL and FT drifts may decompose differently over the same dictionary.
3. **Layer band:** Mid-layer Jaccard is lowest while macro cosine is still ~0.29–0.37—mid layers may redistribute information across many features with little top-100 overlap, whereas late layers (23–27) show slightly more shared top features (26–30 intersections).

### Shared features

[`jaccard_and_shared.json`](jaccard_and_shared.json) lists features in both $F_{\mathrm{ICL}}$ and $F_{\mathrm{FT}}$, ranked by mean decoder cosine to the two drifts. Many shared entries have **asymmetric** scores (high alignment to one drift, low to the other), so the intersection is often “both lists include this feature” rather than “equally central to both shifts.” Features with balanced scores on both drifts are the strongest candidates for a **shared misalignment axis** in SAE space; those are rarer and worth manual Neuronpedia review if you extend this work.

### Overall conclusion (for this OOD run)

| Question | Answer on this run |
|----------|-------------------|
| Does ICL mimic FT in **dense** last-token drift? | **Partly** — moderate positive cosines (~0.3–0.37). |
| Do ICL and FT pick the **same top SAE features** for that drift? | **Mostly no** — low Jaccard (often <10%), modestly higher late layer. |
| Is prompting a mechanistic substitute for fine-tuning? | **Not at the sparse-feature level**; **weakly yes at the mean residual level** on OOD prompts. |

### Suggested follow-ups

- Vary **k** or demo sampling seed to test robustness of $\delta$ alignment.
- Per-prompt cosines: is macro similarity driven by a subset of OOD categories (e.g. vulnerable-user vs persona)?
- Compare **encoder activation deltas** on the last token (as in `sae_code/compute_prompt_token_latent_deltas.py`) vs decoder-cosine here.
- Neuronpedia descriptions for high–mean-cos shared features at layers 11 and 27.

---

## Reproducibility

```bash
# Phase 1 — collect last-token activations (GPU)
bash evals_final/icl/run_collect_ood_activations.sh

# Phase 2 — aggregate, drift, SAE overlap (GPU)
bash evals_final/icl/run_aggregate_and_analyze.sh
```
