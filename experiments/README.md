# experiments/

This directory holds one **subdirectory per experiment**.  Each subdirectory
contains:

| File | Purpose |
|------|---------|
| `config.py` | Python module that exports a `TrainingConfig` instance with any field overrides for this experiment |
| `notes.md` | Free-form notes: hypothesis, results, next steps |

## How to add a new experiment

```bash
# 1. Copy an existing experiment as a starting point
cp -r experiments/exp_01_baseline_vae experiments/exp_03_my_new_idea

# 2. Edit the config for your new idea
#    Change model_name, block_out_channels, latent_channels, etc.
nano experiments/exp_03_my_new_idea/config.py

# 3. (Optional) Register a new model class
#    Add your class to models/ and register it in models/registry.py

# 4. Run it — no other file needs to change
python run.py --experiment exp_03_my_new_idea

# 5. Write up what you found
nano experiments/exp_03_my_new_idea/notes.md
```

## Naming convention

`exp_<two-digit-number>_<short_slug>`

Examples: `exp_01_baseline_vae`, `exp_02_deep_encoder`, `exp_03_larger_latent`

A sequential number makes it easy to see the order experiments were tried, and
the slug reminds you what the idea was.

## Plugging in a new model architecture

1. Create `models/my_new_arch.py` with your model class.
2. In `models/registry.py`, add:

```python
from models.my_new_arch import MyNewArch
register_model("my_new_arch", MyNewArch)
```

3. Set `model_name = "my_new_arch"` in your experiment's `config.py`.
4. `run.py` will automatically pick it up — no other changes needed.
