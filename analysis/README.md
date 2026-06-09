# analysis/ — cheap, zero/low-GPU figures on the EM feature sets

Three Tier-A analyses (no generation, no GPT judge) probing whether the three
narrow finetunes (bad-medical, extreme-sports, risky-financial) share a
misalignment direction, and what the shared SAE features are. Run each with the
project venv; outputs land in `analysis/outputs/`.

| script | question | output |
|---|---|---|
| `a1_diffmean_cosine.py` | Do the 3 finetunes' diff-in-means directions agree? | cosine matrices + heatmaps |
| `a2_subspace_capture.py` | Do the shared SAE features reconstruct those directions? | captured-fraction bars |
| `a4_description_composition.py` | What *are* the shared vs unique features? | description-category bars |

```bash
source .venv/bin/activate
python analysis/a1_diffmean_cosine.py      # seconds, CPU
python analysis/a2_subspace_capture.py     # ~3 min (loads the 5 SAEs from disk)
python analysis/a4_description_composition.py  # seconds, CPU
```

## Findings (this run)

**A1 — yes, strongly.** Pairwise cosine of the unit diff-in-means directions is
**0.55–0.84** (mean 0.61–0.73 per layer). Highest for extreme-sports↔risky-financial
(0.68–0.84), lowest for bad-medical↔risky-financial; all decay with depth
(L11 mean 0.73 → L27 mean 0.61). The three narrow finetunes move the residual
stream along a largely **shared axis**, before any SAE is involved.

**A2 — yes.** A domain-general set of ~25–35 shared features per layer captures
**18–37%** of *every* finetune's misalignment direction — ~15–30× the random
floor (~1%), and comparable to each finetune's own `unique` set. So the shared
features sparsely reconstruct all three directions, not just one.

### A2 in linear-algebra terms (what "fraction captured" means)

**Objects.** Everything is in the residual stream at a layer, a vector space
`R^4096` (`d_model = 4096`).
- `v` = the **misalignment direction**: `mean(finetune acts) − mean(instruct acts)`,
  scaled to length 1 (`||v|| = 1`). One arrow pointing "toward misaligned."
- `w_f` = a feature's **decoder column**: a unit arrow for the way SAE feature `f`
  pushes the residual stream.
- A feature **set** `S = {f_1,…,f_n}` spans a **subspace** `U = span(S)` of
  dimension ≤ n — picture a ~27-dimensional flat "sheet" inside the 4096-dim space.

**The question:** how much of the arrow `v` lies inside the sheet `U`?

**Projection + Pythagoras.** Split `v` into a part in the sheet and a part
orthogonal to it: `v = v∥ + v⊥`, with `v∥ ∈ U` (the closest point in `U` to `v`)
and `v⊥ ⊥ U`. Since they're perpendicular, `||v||² = ||v∥||² + ||v⊥||²`. With
`||v|| = 1`,

    captured = ||v∥||²  ∈ [0, 1]

is exactly the fraction of `v`'s squared length living in the subspace. It equals
`cos²θ`, where θ is the angle between `v` and the sheet — so captured = 0.25 ⇒
θ = 60°.

**Computing it (the QR step).** Projecting needs an **orthonormal** basis of `U`
(perpendicular unit axes `q_1,…,q_k`). The decoder columns are *not* orthogonal
(features can point similar ways), so `captured_fraction` runs QR on the
`4096 × n` column matrix `D = QR`; `Q` gives an orthonormal basis for the same
span, and

    captured = ||Qᵀ v||² = Σ_i (q_iᵀ v)².

Orthonormalizing first is essential: naïvely summing `Σ_f (w_fᵀ v)²` over the
overlapping raw columns would double-count shared directions and could exceed 1.
QR removes the redundancy (it's also why 27 features each with cosine ~0.15 to `v`
give ~0.25, not `27·0.15² ≈ 0.6` — they overlap each other).

**The random floor — why ~1%.** For a *fixed* unit `v` and a **random**
k-dimensional subspace of `R^d`, the expected captured fraction is exactly `k/d`
(by symmetry: the squared length spreads evenly over all d axes and you grab k).
With k ≈ 27, d = 4096 that's ≈ 0.0066; we measure ≈ 0.01–0.017 (same order, a hair
higher because decoder directions aren't perfectly isotropic). So a same-size set
of *meaningless* features captures ~1%.

**Reading the result.** Shared captures **18–37% = 15–30× the random floor**. A
27-dim sheet is only `27/4096 = 0.66%` of the space yet holds a quarter of the
misalignment arrow — these sparse features are aligned with the direction far
beyond chance, and the *same* sheet does it for all three finetunes at once
(whose arrows A1 already showed are 0.6–0.8 cosine-similar). That is a
*domain-general* misalignment subspace.

**Honesty notes.**
1. *Fraction, not majority.* 25% captured ⇒ 75% of the direction is still outside
   these features; they're a meaningful sparse component, not the whole shift.
2. *Partly circular by design.* Features were selected as top-cosine-to-`v`, so
   high capture is expected — which is why the **random floor** and the
   **shared-vs-unique** contrast are load-bearing. Shared features had to be top in
   *all three* finetunes at once, yet capture each single direction about as well
   as that finetune's *own* `unique` set (sometimes better, e.g. bad-medical
   L19/L27): the cross-domain constraint cost almost nothing.
3. *Geometry, not behavior.* "Captures 25% of the direction" ≠ "causes 25% of
   misaligned behavior" — that needs steering/ablation (Tier B/C).

**A4 — inconclusive / honest negative.** The auto-interp `description` labels are
too bland to support a "shared = misaligned persona" claim by label-mining: only
**2 / 138** shared features read as clearly harmful ("Scams and fraud"; sexual
role-play), **53%** are punctuation/format, and **38%** are generic function-word /
discourse labels ("that", "discussions", "he"). The unique sets look similar.
Takeaway: don't mine descriptions for the persona story — either re-interp from
max-activating examples, or lean on the causal steer/ablate experiments.
Per-feature assignments are dumped to `a4_description_assignments.jsonl` for audit
(the keyword classifier is heuristic).

## Causal interventions — standard practice (read before steering/ablating)

A4's composition is not academic: when we first ablated a whole set
(`unique-medical`, 138 dirs across all 5 layers) out of the medical model, the
model collapsed to gibberish — `coherent_rate 0.0`, `mean_coherence 9/100`. You
can't read alignment off incoherent text, so that run taught us nothing. The cause
is exactly A4's finding: the top-cosine sets are mostly **format** (53%) and
**generic function-word** (38%) features — always-on directions the model needs to
stay fluent — and projecting a big subspace of them out at five layers lobotomizes
it. Hence the standard practice now baked into the code:

1. **Contentful features only.** `andy_sae.build_feature_sets()` keeps only
   `harmful`/`topical` features (drops `format`+`generic`) by default, using the
   canonical `andy_sae.classify`. Pass `contentful_only=False` for the raw sets
   (A2 does this — its point is the geometry of the full selection). This shrinks
   the sets a lot: shared 138→13, unique-medical 283→12, sports 198→7, fin 207→18.
2. **Layers 11 and 15 only** (`andy_sae.STEER_LAYERS`). The cross-layer compounding
   is a damage multiplier; the SAEs for 19/23/27 still load for the A1/A2 geometry.
3. **Top-10 by cosine** (`andy_sae.DEFAULT_TOP_N`). At 11/15 this leaves ~2–6
   features per set — close to the blog's ~10 hand-picked features that preserved
   coherence.

These are the defaults of `AndySAE.steering_set` / `ablating_set` and of
`misalignment_eval.py --set-steer/--set-ablate` (load just 11/15; `--set-top-n 0`
for all features). The named sets at 11/15+top10 are:
`shared`=6, `unique-medical`=4, `unique-sports`=2, `unique-financial`=6 features.

### `vibecheck.py` — tune before you spend an eval job

The judged eval costs ~15 min + an API key, far too slow for finding a coherent
regime. The probe prints baseline-vs-intervened **greedy** generations in seconds
(no judge), and first lists the exact features (with descriptions) being touched:

```bash
python analysis/vibecheck.py --set shared --mode steer --alpha 6
python analysis/vibecheck.py --model bad-medical-advice --set shared --mode ablate
```

Workflow: probe until the text is fluent, *then* run `misalignment_eval.py` for the
judged number. Never use the 15-min eval as the coherence-debugging loop.

## Causal results — the ablation dissociation (current headline)

With the gentle standard above, ablating the named sets out of each finetune is now
surgical (coherence stays ~0.9–1.0). `ablation_bars.py` draws this from the
`misalignment_eval.py` metrics; `python analysis/ablation_bars.py` →
`outputs/ablation_bars.png`. Each finetune, EM = `misaligned_coherent`, n=80:

| model (baseline EM) | ablate **shared** | ablate own-unique |
|---|---|---|
| bad-medical (0.150) | **0.038** ↓75% | 0.150 (inert) |
| extreme-sports (0.100) | **0.000** | 0.038 (mild) |
| risky-financial (0.088) | **0.013** ↓85% | 0.100 (inert) |

**Within-model story (robust, 3/3 models):** removing the ~6 domain-general
`shared` features collapses EM while coherence and fluency survive, and mean
alignment rises (e.g. bad-medical 62.7→77.6). A finetune's *own* domain-specific
features are nearly inert. This is the clean necessity result.

**Cross-model story (the twist — read this):** the unique sets are NOT clean
domain controls.

| model | ablate cross-unique | Δ vs baseline |
|---|---|---|
| risky-financial + unique-**medical** | 0.075 | ~0 (specific, as hoped) |
| bad-medical + unique-**financial** | 0.038 | **−0.112 (as big as shared!)** |

Ablating `unique-financial` out of the *medical* model kills its EM as hard as
`shared` does. Cause (verified by decoder cosine): `unique-financial` smuggles in
persona features `f113115 "financial scams"` and `f1972 "scams"`, and **f113115 has
cosine 0.29 to `shared`'s `f63896 "Scams and fraud"`** — essentially the same
direction. `unique-medical` is inert because at 11/15 it's all L15 *topical* labels
("scientific studies/writing") with **no L11 persona feature** to remove.

**Interpretation:** EM is not carried by "the shared set" as a block — it rides on a
small **scam / fraud / role-play persona sub-direction** that populates `shared` and
*leaks* into some unique sets but not others. The clean shared-vs-unique
dissociation was partly an artifact of which features landed where. The deeper claim
(misalignment = a few harmful-persona features) is *strengthened*; the set-level
framing is what's leaky. This is why the next work is **feature-level**.

### The current persona-feature candidates (`shared`, contentful, @11/15)

| layer | feature | cos→dm | category | description |
|---|---|---|---|---|
| 11 | f63896 | 0.107 | harmful | **Scams and fraud** |
| 11 | f106036 | 0.114 | harmful | **role-play, sexual/intimate, first-person impersonation** |
| 11 | f39163 | 0.186 | topical | dieting advice |
| 11 | f26620 | 0.113 | topical | business/technical |
| 11 | f58537 | 0.108 | topical | academic texts |
| 15 | f112956 | 0.146 | topical | health and medical advice |

The two `harmful` ones (and the cross-model `f113115 financial scams`) are the prime
persona suspects; the `topical` ones are likely just what the finetunes talk about.

### Random null — control now available

`AndySAE.random_set()` builds a size-matched (to `shared`: 5@L11+1@L15) random draw
from non-set features, exposed in the eval as the synthetic set `random`:

```bash
bash evals/run_misalignment_eval.sh --model bad-medical-advice --set-ablate random
bash evals/run_misalignment_eval.sh --model bad-medical-advice --set-ablate random --set-seed 1
```

Prediction: ~no EM drop. Run 2–3 seeds/model so the headline can show that ablating
the *identity* of the persona features matters, not just removing any 6 directions.

## Next studies — hunting the persona features

The leak result reframes the goal: stop comparing opaque sets, start **isolating the
individual persona features**. Roughly in priority order:

1. **Per-feature ablation (single-feature knockout).** Ablate each of the ~6 shared
   features one at a time on each finetune. Hypothesis: `f63896 "Scams and fraud"`
   and/or `f106036 "role-play"` carry most of the EM drop; the topical ones do
   little. This directly names the persona feature(s) instead of a set. (Cheap: each
   is one eval job; the per-feature loop is the obvious next bit of tooling.)
2. **Random null, 2–3 seeds** (above) — the control that makes #1 publishable.
3. **Re-interpret the suspects from max-activating examples.** A4 showed the cossim
   `description` labels are too bland; for the handful of candidates it's worth
   pulling real max-activating contexts (the `AndySAE.capture` / `top_k_features`
   path) and re-labelling, to confirm "Scams and fraud" / "role-play" really are
   persona directions and not topic detectors.
4. **Build a clean persona set vs a clean topic control.** Pool the `harmful`
   features across *all* sets (scams/fraud/crime/role-play — the union, not just
   `shared`) into a "persona set", and the `topical` features into a "topic control".
   Ablating persona should drop EM across all three models; ablating topic shouldn't.
   This replaces the leaky shared-vs-unique split with a semantically clean one.
5. **Steer the single potent feature, not the set.** The set-normalized steering
   washed out (instruct α6 → 0.05). Try steering `f63896`/`f106036` alone on instruct
   — induction may be much stronger from one sharp direction than a 6-way average.
6. **Widen the candidate pool.** The contentful sets are tiny because of the cossim
   top-500 cutoff + persona-rarity. Lower the cutoff (top-1000/2000) or rank features
   directly by a harmful-relevance score (à la the blog's 0–10 LLM rating) to surface
   more persona candidates beyond the 2–3 we have.
7. **Forward-pass MCQ metric** for cheap dose-response with no coherence ceiling —
   lets us sweep per-feature steering strength quantitatively without the judge.
