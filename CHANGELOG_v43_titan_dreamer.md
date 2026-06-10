# v43 — Titan-Dreamer Expert + Consensus/MMR Runtime Re-rank

## New model variant: `titan_dreamer_expert`
New file `source/model_frontier_v43.py`, fusing three 2025–26 research lines:

- **Titans neural long-term memory** (Behrouz et al., arXiv 2501.00663; MIRAS): a 2-layer fast-weight memory MLP updated at test time by surprise gradients with momentum and forgetting, read through a query projection with a MAG-style sigmoid gate. A straight-through estimator keeps the slow weights trainable.
- **TNT chunkwise test-time memorization** (arXiv 2511.07343): inner updates run over the flattened token chunk in 2 steps for stability.
- **Dreamer depth-recurrent attention mixtures** (arXiv 2601.21582) with **mixture-of-recursions routing** (arXiv 2510.25741): the latent is refined over 3 depth steps that attend over previous depth states + 4 persistent memory tokens; a per-token router mixes depth outputs.

All new branches are zero-init scaled (`alpha`, `beta`, `shared_scale`) and keep `weight`/`bias` keys, so warm-starting from any base checkpoint is safe. Registered in `SUPPORTED_MODEL_SIZES`, `build_model`, `load_weights_for_model`, and `detect_model_size_from_state_dict` (keys: `layers.10.titan_w1` + `layers.10.recursion_router.weight`).

Train: `finetune_chat.py --model_size titan_dreamer_expert` (warm-start from current champion supported).
Smoke test: `python source/test_titan_dreamer_expert.py` (forward/backward, gradient flow, eval determinism, detection).

## Runtime upgrade (no retraining needed)
`chat_pipeline.py` (source + runtime, kept in sync): new `_consensus_mmr_rerank` stage in `pick_response`:

- Self-consistency bonus (+0.06 max) for candidates supported by other top-16 candidates.
- MMR diversity ordering so sampled alternates aren't near-duplicates; sampling pool widened 4 → 6.

Note: smoke test not yet executed (workspace shell out of disk space this session).
