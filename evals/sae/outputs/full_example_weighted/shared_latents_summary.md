# Shared Latent Summary (Full Run, Example-Weighted)

This summary is computed from:

- `evals/sae/outputs/full_example_weighted/bad-medical-advice/summary_top_latents.jsonl`
- `evals/sae/outputs/full_example_weighted/extreme-sports/summary_top_latents.jsonl`
- `evals/sae/outputs/full_example_weighted/risky-financial-advice/summary_top_latents.jsonl`

## Headline Findings

- There are **95 shared (layer, latent)** pairs across all three models' top-200 summaries.
- Strongest repeated increases concentrate in late layers **22-25**.
- Top repeated latents across all three models:
  - `L25 / latent 80629` (mean delta `2.9905`)
  - `L24 / latent 102449` (mean delta `2.6540`)
  - `L23 / latent 49555` (mean delta `2.1597`)
  - `L22 / latent 110585` (mean delta `1.6337`)

## Top 10 Shared Latents (by mean delta)

- `L25 / latent 80629`: bad `3.5063`, sports `2.2057`, finance `3.2596`, mean `2.9905`
- `L24 / latent 102449`: bad `3.0902`, sports `1.9182`, finance `2.9535`, mean `2.6540`
- `L23 / latent 49555`: bad `2.4396`, sports `1.5376`, finance `2.5020`, mean `2.1597`
- `L22 / latent 110585`: bad `1.8025`, sports `1.1831`, finance `1.9154`, mean `1.6337`
- `L19 / latent 34467`: bad `1.5753`, sports `0.8472`, finance `1.5429`, mean `1.3218`
- `L16 / latent 485`: bad `1.5499`, sports `1.0066`, finance `1.2466`, mean `1.2677`
- `L18 / latent 80644`: bad `1.2964`, sports `0.7306`, finance `1.2077`, mean `1.0782`
- `L17 / latent 27390`: bad `1.0491`, sports `0.4708`, finance `0.7971`, mean `0.7723`
- `L24 / latent 25313`: bad `0.7220`, sports `0.6462`, finance `0.7372`, mean `0.7018`
- `L22 / latent 13159`: bad `0.4372`, sports `0.6937`, finance `0.6805`, mean `0.6038`

## Best Shared Latent per Layer

- `L15`: latent `96419` (mean `0.3574`; bad `0.2699`, sports `0.3814`, finance `0.4209`)
- `L16`: latent `485` (mean `1.2677`; bad `1.5499`, sports `1.0066`, finance `1.2466`)
- `L17`: latent `27390` (mean `0.7723`; bad `1.0491`, sports `0.4708`, finance `0.7971`)
- `L18`: latent `80644` (mean `1.0782`; bad `1.2964`, sports `0.7306`, finance `1.2077`)
- `L19`: latent `34467` (mean `1.3218`; bad `1.5753`, sports `0.8472`, finance `1.5429`)
- `L20`: latent `15842` (mean `0.4551`; bad `0.6276`, sports `0.1243`, finance `0.6133`)
- `L21`: latent `87801` (mean `0.5072`; bad `0.4669`, sports `0.4601`, finance `0.5946`)
- `L22`: latent `110585` (mean `1.6337`; bad `1.8025`, sports `1.1831`, finance `1.9154`)
- `L23`: latent `49555` (mean `2.1597`; bad `2.4396`, sports `1.5376`, finance `2.5020`)
- `L24`: latent `102449` (mean `2.6540`; bad `3.0902`, sports `1.9182`, finance `2.9535`)
- `L25`: latent `80629` (mean `2.9905`; bad `3.5063`, sports `2.2057`, finance `3.2596`)
