# Figure scripts

Scripts for the paper and Supporting Information: test-error profiles and maps,
temporal error analysis, EP-flux diagnostics, and table exports.

Running them requires a populated local manifest and the corresponding evaluated
data. Neither is included. The manifest specifies checkpoint selections, model
settings, events, input paths/checksums, and reference outputs, including private
W&B run mappings.

Copy `manifest.example.json` to `manifest.json`, supply the metadata and data, then run:

```bash
python scripts/reproduce.py all --manifest scripts/manifest.json
```

The template is intentionally empty. Populated manifests and generated outputs
are ignored by Git. Published figures are in `results/paper/figures/`.
