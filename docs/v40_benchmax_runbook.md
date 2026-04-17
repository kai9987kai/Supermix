# v40_benchmax Runbook

## Objective

`v40_benchmax` is an isolated benchmark-maximization experiment line. It is designed to beat the current common-benchmark leader while keeping the existing desktop, runtime, installer, and benchmark flows intact.

## Baselines

- `v33_final` is the current common-benchmark leader.
- `v39_final` is the nearest recent comparator.

## 2x2 Ablation Matrix

The experiment compares:

- `v33_data__v33_head`
- `v33_data__v39_head`
- `v39_data__v33_head`
- `v39_data__v39_head`

The matrix is machine-readable in [source/v40_benchmax_manifest.json](../source/v40_benchmax_manifest.json).

## Novel Features

- Benchmark-pressure hard-example mining from per-item benchmark details.
- Collective-teacher distillation through the local multimodel runtime.
- Strict checkpoint soup compatibility gating.
- Promotion gate against `v33_final` and `v39_final`.
- Explicit data/head ablation definitions for repeatability.
- Larger and more diverse local corpus sampling for both `v33` and `v39` style data recipes.
- Dedicated protein-folding knowledge rows so benchmark-focused runs also pick up structural-biology reasoning.
- Reasoning-budget control inspired by recent small-model reasoning work.
- Teacher-repair replay rows so failed drafts and corrected answers can be preserved together.
- Failure-cluster curriculum so the next run can oversample recurring benchmark miss families.
- Multi-teacher consensus distillation with agreement metadata and disagreement tracking.
- Research-pack synthesis that emits benchmark replay, repair replay, verifier replay, and consensus-teacher rows.
- Budget-aware verifier prompts so frontier misses get slower, more explicit verification rows than stabilize misses.

## Artifact Layout

Recommended output tree:

- `output/v40_benchmax/v40_benchmax_manifest.json`
- `output/v40_benchmax/v40_benchmax_ablation_summary.json`
- `output/v40_benchmax/ablations/<ablation_id>/rows.jsonl`
- `output/v40_benchmax/ablations/<ablation_id>/summary.json`
- `output/v40_benchmax/hard_examples/`
- `output/v40_benchmax/distill/`
- `output/v40_benchmax/research_pack/`
- `output/v40_benchmax/soup/`
- `output/v40_benchmax/reports/`

Key machine-readable artifacts:
- ablation matrix JSON and CSV
- hard-example JSONL, JSON summary, CSV, and Markdown summary
- collective distillation JSONL and JSON summary
- research-pack JSONL, CSV, JSON summary, and Markdown summary
- soup compatibility JSON
- promotion report JSON, CSV, and Markdown
- data summaries that now include support-row counts and protein-folding row counts

## Smoke Commands

Generate the manifest:

```powershell
python source/build_v40_benchmax_manifest.py --output_dir output/v40_benchmax
```

Build the ablation packs:

```powershell
python source/build_v40_benchmax_ablation.py --repo_root . --output_dir output/v40_benchmax --dry_run
```

Unified CLI equivalent:

```powershell
python source/v40_benchmax_pipeline.py ablation --repo_root . --output_dir output/v40_benchmax --dry_run true
```

Extract hard examples from benchmark details:

```powershell
python source/build_v40_benchmax_hard_examples.py --details_jsonl output/benchmark_all_models_common_details.jsonl --summary_json output/benchmark_all_models_common_summary.json --output_dir output/v40_benchmax/hard_examples
```

Generate collective-teacher distillation rows:

```powershell
python source/build_v40_benchmax_collective_distill.py --input_jsonl output/v40_benchmax/hard_examples/hard_examples.jsonl --output_jsonl output/v40_benchmax/distill/collective_teacher.jsonl --summary_out output/v40_benchmax/distill/collective_teacher_summary.json
```

Unified CLI equivalent:

```powershell
python source/v40_benchmax_pipeline.py distill --prompts_jsonl output/v40_benchmax/hard_examples/hard_examples.jsonl --output_dir output/v40_benchmax
```

Build the research pack from hard examples and optional distillation output:

```powershell
python source/build_v40_benchmax_research_pack.py --hard_examples_jsonl output/v40_benchmax/hard_examples/hard_examples.jsonl --distillation_jsonl output/v40_benchmax/distill/collective_teacher.jsonl --output_dir output/v40_benchmax/research_pack
```

Unified CLI equivalent:

```powershell
python source/v40_benchmax_pipeline.py research-pack --hard_examples_jsonl output/v40_benchmax/hard_examples/hard_examples.jsonl --distillation_jsonl output/v40_benchmax/distill/collective_teacher.jsonl --output_dir output/v40_benchmax
```

Checkpoint soup dry run:

```powershell
python source/checkpoint_soup_v40.py --checkpoint path\to\a.pth --checkpoint path\to\b.pth --output output\v40_benchmax\soup\merged.pth --summary_out output\v40_benchmax\soup\merged.json --dry_run
```

Build the benchmark comparison report:

```powershell
python source/report_v40_benchmax.py --candidate_summary output\v40_benchmax\candidate_summary.json --v33_summary output\benchmark_all_models_common_plus_summary_20260330.json --v39_summary output\v40_benchmax\v39_summary.json --output_dir output\v40_benchmax\reports
```

Train an ablation with the experimental research pack:

```powershell
python source/train_v40_benchmax.py --repo_root . --output_dir output\v40_benchmax --ablation_id v40_v39data_v39recipe --research_pack_jsonl output\v40_benchmax\research_pack\v40_benchmax_research_pack.jsonl
```

Inspect the expanded ablation row mix:

```powershell
python source/build_v40_benchmax_ablation.py --repo_root . --output_dir output/v40_benchmax --ablation_id v40_v39data_v39recipe --dry_run
```

Unified CLI equivalent:

```powershell
python source/v40_benchmax_pipeline.py report --benchmark_summary output\\benchmark_all_models_common_plus_summary_20260330.json --candidate_model v40_benchmax --leader_models v33_final --leader_models v39_final --output_dir output\\v40_benchmax
```

## Full Pipeline Order

1. Generate the source-of-truth manifest snapshot.
2. Generate the 2x2 ablation packs.
3. Run the benchmark harness on baselines or candidates to refresh per-item details.
4. Export hard examples from the weakest benchmark items.
5. Build optional collective-teacher distillation rows.
6. Build the research pack to create curriculum, repair, verifier, and consensus-teacher rows.
7. Train one or more `v40_benchmax` ablations, optionally injecting the research pack.
8. Dry-run checkpoint soup, then merge only compatible checkpoints.
9. Run the benchmark comparison report and apply the promotion gate.

## Promotion Gate

Promotion is recommended only if the candidate:

- beats `v33_final` on overall exact accuracy,
- beats `v39_final` on overall exact accuracy,
- and avoids large benchmark regressions on individual benchmarks.

The automatic report also answers:
- which experiment won
- delta vs `v33_final`
- delta vs `v39_final`
- which signals suggest data, recipe/head, distillation, or soup caused the gain
- the recommended next run

## Known Limitations

- The real `v39` data path depends on the local `datasets` package; the smoke path falls back to compact synthetic rows if those heavy dependencies are unavailable.
- The collective distillation path can fall back to a stub teacher if the multimodel runtime or Qwen-side dependencies are unavailable.
- Checkpoint soup only supports strictly compatible checkpoints from the same family and shape set.
- A full `v40` benchmark-max training run remains CPU-heavy in this environment.
- Protein-folding rows are knowledge-oriented supervision, not a dedicated geometric protein-structure model.

## Research Notes

The manifest includes recent research influences for:
- reasoning-budget control
- verifier-guided correction
- agent-style tool distillation
- self-correction
- multi-verifier test-time agreement
- solver-verifier replay for code and benchmark repair

These are adapted as additive experimental flags rather than changing the semantics of the existing benchmark suite.

## Notes

- The `v40_benchmax` family is additive and does not alter the current desktop or installer flows.
- The training entry point reuses the existing `OmniCollectiveNetV5` training helpers, but drives them from the new manifest and ablation packs.
