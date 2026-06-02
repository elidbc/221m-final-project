# Shared Latent Summary (Full Run, Instruct SAEs, Example-Weighted)

This summary is computed from:

- `evals/sae/outputs/full_instruct_example_weighted/bad-medical-advice/summary_top_latents.jsonl`
- `evals/sae/outputs/full_instruct_example_weighted/extreme-sports/summary_top_latents.jsonl`
- `evals/sae/outputs/full_instruct_example_weighted/risky-financial-advice/summary_top_latents.jsonl`

## Headline Findings

- There are **92 shared (layer, latent)** pairs across all three models' top-200 summaries.
- Strongest repeated increases concentrate in higher instruct-SAE layers, especially **23** and **27**.
- Top repeated latents across all three models:
  - `L27 / latent 85258` (mean delta `5.4633`)
  - `L23 / latent 39242` (mean delta `2.6321`)
  - `L27 / latent 107798` (mean delta `2.4930`)
  - `L19 / latent 53974` (mean delta `1.3052`)

## Top 10 Shared Latents (by mean delta)

- `L27 / latent 85258`: bad `6.7971`, sports `4.0592`, finance `5.5337`, mean `5.4633`
- `L23 / latent 39242`: bad `3.1047`, sports `1.8712`, finance `2.9205`, mean `2.6321`
- `L27 / latent 107798`: bad `2.6947`, sports `2.1893`, finance `2.5951`, mean `2.4930`
- `L19 / latent 53974`: bad `1.5236`, sports `0.9175`, finance `1.4746`, mean `1.3052`
- `L15 / latent 129304`: bad `0.7542`, sports `0.5547`, finance `0.4490`, mean `0.5860`
- `L11 / latent 40599`: bad `0.6142`, sports `0.5462`, finance `0.4587`, mean `0.5397`
- `L27 / latent 87482`: bad `0.2346`, sports `0.6229`, finance `0.6748`, mean `0.5108`
- `L23 / latent 78831`: bad `0.4922`, sports `0.3815`, finance `0.6364`, mean `0.5034`
- `L7 / latent 128494`: bad `0.3795`, sports `0.4950`, finance `0.4229`, mean `0.4325`
- `L27 / latent 10082`: bad `0.2881`, sports `0.3546`, finance `0.3800`, mean `0.3409`

## Best Shared Latent per Layer

- `L7`: latent `128494` (mean `0.4325`; bad `0.3795`, sports `0.4950`, finance `0.4229`)
- `L11`: latent `40599` (mean `0.5397`; bad `0.6142`, sports `0.5462`, finance `0.4587`)
- `L15`: latent `129304` (mean `0.5860`; bad `0.7542`, sports `0.5547`, finance `0.4490`)
- `L19`: latent `53974` (mean `1.3052`; bad `1.5236`, sports `0.9175`, finance `1.4746`)
- `L23`: latent `39242` (mean `2.6321`; bad `3.1047`, sports `1.8712`, finance `2.9205`)
- `L27`: latent `85258` (mean `5.4633`; bad `6.7971`, sports `4.0592`, finance `5.5337`)

## Notes

- Layer `3` does not appear in this per-layer-best list because no layer-3 latent was shared across all three models in the top-200 intersection used for this summary.
