# V52 initialization artifact

`champion_model_chat_v50_cognitive_leap.pth` is the selectively imported,
trained conversational checkpoint from the `Supermix_27` donor checkout.

- size: 6,768,714 bytes
- SHA-256: `6D595A44EE130A569BAE4E697007BC4D3212A5142598F7EA59C8DDF73FFCC864`
- detected model family: `cognitive_leap_expert`
- intended use: initialize the compatible backbone and base classifier weights
  of `cognitive_leap_v52_expert` before v52 fine-tuning

This is not a trained v52 checkpoint. Sparse routing, the quality/continue
verifier, calibration, and affect/intent/strategy heads require new training and
promotion benchmarks. The large response-bucket metadata remains at
`chat_model_meta_v50_cognitive_leap.json` in the project root.

`champion_model_chat_v52_initialized.pth` is the deterministic, strict-loadable
materialization produced by `source/materialize_v52_from_v50.py` with seed 52.
It prevents separate source and packaged processes from inventing different
random v52 heads around the same donor checkpoint.

- size: 9,737,831 bytes
- SHA-256: `1AB25AA9772F90896A9A6C28EB346135B9B9A8D48FA9D84936C4D802D742C192`
- tensor-state SHA-256: `D9FEA1FAFBF61D967370BB9D4F09BFDF19F93C04FB14C9EAE4E223994692044A`
- detected model family: `cognitive_leap_v52_expert`

The materialized file is still initialization-only; see
`v52_initialization_manifest.json` for the machine-readable boundary.
