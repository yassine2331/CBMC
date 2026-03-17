# Experiment 01 — Baseline DynamicVAE

## Hypothesis
Establish a working baseline with the default architecture so all other
experiments have a fixed point of comparison.

## Config highlights
- `block_out_channels = (128, 128, 256, 256, 512)`
- `latent_channels = 4`
- `context_dim = 8`
- Single `AttnDownBlock2D` / `AttnUpBlock2D` pair

## Results
_Fill in after running the experiment._

| Metric | Value |
|--------|-------|
| Reconstruction loss | — |
| KL divergence | — |
| Epochs | 5 |

## Observations
_Write what you noticed here._

## Next steps
- Try a deeper encoder → see `exp_02_deep_encoder`
