# Shared Latent Summary (Full Run, Token-Weighted)

This summary is computed from:

- `evals/sae/outputs/full_token_weighted/bad-medical-advice/summary_top_latents.jsonl`
- `evals/sae/outputs/full_token_weighted/extreme-sports/summary_top_latents.jsonl`
- `evals/sae/outputs/full_token_weighted/risky-financial-advice/summary_top_latents.jsonl`

## Headline Findings

- There are **97 shared (layer, latent)** pairs across all three models' top-200 summaries.
- Strongest repeated increases still concentrate in late layers **22-25**.
- Top repeated latents across all three models:
  - `L24 / latent 102449` (mean delta `1.4307`)
  - `L25 / latent 80629` (mean delta `1.4191`)
  - `L23 / latent 49555` (mean delta `1.2000`)
  - `L22 / latent 110585` (mean delta `1.1359`)

## Top 10 Shared Latents (by mean delta)

- `L24 / latent 102449`: bad `1.7031`, sports `0.8147`, finance `1.7743`, mean `1.4307`
- `L25 / latent 80629`: bad `1.7592`, sports `0.7790`, finance `1.7191`, mean `1.4191`
- `L23 / latent 49555`: bad `1.3531`, sports `0.6810`, finance `1.5659`, mean `1.2000`
- `L22 / latent 110585`: bad `1.2386`, sports `0.7326`, finance `1.4364`, mean `1.1359`
- `L24 / latent 25313`: bad `0.8899`, sports `0.7869`, finance `0.8915`, mean `0.8561`
- `L22 / latent 13159`: bad `0.6349`, sports `0.8658`, finance `0.8691`, mean `0.7899`
- `L17 / latent 81357`: bad `0.5922`, sports `0.6950`, finance `0.7691`, mean `0.6854`
- `L19 / latent 11267`: bad `0.5556`, sports `0.6739`, finance `0.8049`, mean `0.6781`
- `L23 / latent 23812`: bad `0.5989`, sports `0.6187`, finance `0.7203`, mean `0.6460`
- `L19 / latent 34467`: bad `0.8195`, sports `0.2340`, finance `0.8618`, mean `0.6384`

## Best Shared Latent per Layer

- `L15`: latent `96419` (mean `0.3899`; bad `0.3038`, sports `0.4105`, finance `0.4553`)
- `L16`: latent `18883` (mean `0.4929`; bad `0.3352`, sports `0.5523`, finance `0.5912`)
- `L17`: latent `81357` (mean `0.6854`; bad `0.5922`, sports `0.6950`, finance `0.7691`)
- `L18`: latent `34972` (mean `0.5305`; bad `0.4000`, sports `0.5615`, finance `0.6300`)
- `L19`: latent `11267` (mean `0.6781`; bad `0.5556`, sports `0.6739`, finance `0.8049`)
- `L20`: latent `32479` (mean `0.2760`; bad `0.2690`, sports `0.2362`, finance `0.3228`)
- `L21`: latent `87801` (mean `0.6222`; bad `0.5865`, sports `0.5746`, finance `0.7055`)
- `L22`: latent `110585` (mean `1.1359`; bad `1.2386`, sports `0.7326`, finance `1.4364`)
- `L23`: latent `49555` (mean `1.2000`; bad `1.3531`, sports `0.6810`, finance `1.5659`)
- `L24`: latent `102449` (mean `1.4307`; bad `1.7031`, sports `0.8147`, finance `1.7743`)
- `L25`: latent `80629` (mean `1.4191`; bad `1.7592`, sports `0.7790`, finance `1.7191`)
