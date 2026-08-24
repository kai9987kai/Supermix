"""v62: score a checkpoint per domain, and measure what it can say at all.

A single held-out perplexity cannot answer "is it better at maths". Arithmetic is
12.5% of the v62 blend and a rounding error in the aggregate, so a model could
lose every sum and still post a better overall number by improving on dialogue.
Splitting the same held-out rows by their `domain` label makes each claim
separately falsifiable.

Two measurements, because they answer different questions and only one of them is
comparable across models:

**Per-domain loss** (`--checkpoint`) says how well *this* model predicts each
domain. It is not comparable between checkpoints with different vocabularies --
different tokenizers mean perplexity is measured over a different unit, the same
trap `compare_generalisation_runs.py` refuses to walk into.

**Per-domain vocabulary reachability** (`--tokenizer-from`) *is* comparable across
models, because it asks a question about the tokenizer rather than the weights:
what fraction of this domain's words can the model represent at all? A word
outside the vocabulary encodes to `<unk>` and can never be generated, so this is
a hard ceiling on what a model can say about a subject, independent of training.
It is the honest way to compare a narrow-corpus model with a broad-corpus one.

The held-out rows are reproduced by re-deriving the trainer's split with the same
arguments, so nothing scored here was trained on.

    python source/eval_v62_domains.py --checkpoint output/v62_multidomain/v62_multidomain.pt
    python source/eval_v62_domains.py --tokenizer-from output/v60_control_2000/v60_control_2000.pt \\
        --tokenizer-from output/v62_multidomain/v62_multidomain.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402
from train_mimomix_generalisation import load_corpus_pairs  # noqa: E402
from train_mimomix_talk import evaluate  # noqa: E402

RECEIPT_SCHEMA = "supermix-v62-domain-evaluation-v1"

DEFAULT_BLEND = SOURCE_DIR.parent / "datasets" / "v62" / "v62_blend.jsonl"


def load_domain_map(blend: Path) -> Dict[Tuple[str, str], str]:
    """`(user, assistant)` -> domain, so held-out rows can be attributed."""

    mapping: Dict[Tuple[str, str], str] = {}
    with blend.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(record.get("user", "")).strip(), str(record.get("assistant", "")).strip())
            mapping.setdefault(key, str(record.get("domain", "unknown")))
    return mapping


def held_out_rows(
    blend: Path,
    dev_fraction: float,
    test_fraction: float,
    tier3_row_fraction: float,
    max_row_fraction_per_sentence: float,
    split_seed: int,
    min_response_characters: int,
) -> Dict[str, List[Tuple[str, str]]]:
    """Re-derive the trainer's split and return the rows it never trained on."""

    pairs = load_corpus_pairs(
        str(blend),
        corpus_jsonl=str(blend),
        min_response_characters=min_response_characters,
    )
    split = splits.build_generalisation_split(
        pairs,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
        target_row_fraction=tier3_row_fraction,
        max_row_fraction_per_sentence=max_row_fraction_per_sentence,
        seed=split_seed,
        source=str(blend),
    )
    splits.verify_split(split)
    return {name: rows for name, rows in split.tiers()}


def group_by_domain(
    rows: Sequence[Tuple[str, str]], domains: Dict[Tuple[str, str], str]
) -> Dict[str, List[Tuple[str, str]]]:
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for pair in rows:
        grouped[domains.get((pair[0], pair[1]), "unknown")].append(pair)
    return dict(grouped)


def score_domains(
    checkpoint: Path,
    grouped: Dict[str, List[Tuple[str, str]]],
    batch_size: int,
    min_rows: int,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MiMoMixModel(MiMoMixConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    sequence_length = int(model.config.native_context)

    results: Dict[str, Any] = {}
    for domain, rows in sorted(grouped.items()):
        if len(rows) < min_rows:
            results[domain] = {"rows": len(rows), "skipped": f"fewer than {min_rows} rows"}
            continue
        inputs, labels = text_utils.build_training_tensors(rows, tokenizer, sequence_length)
        if inputs.shape[0] == 0:
            results[domain] = {"rows": len(rows), "skipped": "no packed blocks"}
            continue
        metrics = evaluate(model, inputs, labels, batch_size)
        coverage = tokenizer.vocabulary_report([a for _, a in rows])
        results[domain] = {
            "rows": len(rows),
            "blocks": int(inputs.shape[0]),
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
            "response_coverage": coverage["coverage"],
        }
    return {
        "checkpoint": str(checkpoint),
        "vocab_size": tokenizer.vocab_size,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "domains": results,
    }


def reachability(
    checkpoints: Sequence[Path], grouped: Dict[str, List[Tuple[str, str]]]
) -> Dict[str, Any]:
    """What fraction of each domain's words can each tokenizer represent?

    Comparable across models: it is a property of the vocabulary, not of the
    weights, and a word outside it can never be generated at any quality.
    """

    table: Dict[str, Dict[str, Any]] = {}
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
        per_domain: Dict[str, float] = {}
        for domain, rows in sorted(grouped.items()):
            report = tokenizer.vocabulary_report([a for _, a in rows])
            per_domain[domain] = report["coverage"]
        table[checkpoint.stem] = {
            "vocab_size": tokenizer.vocab_size,
            "coverage_by_domain": per_domain,
        }
    return table


def print_domain_table(result: Dict[str, Any]) -> None:
    print(f"checkpoint  {result['checkpoint']}")
    print(f"vocabulary  {result['vocab_size']:,}   parameters {result['parameters']:,}")
    print()
    print(f"{'domain':16s} {'rows':>7s} {'loss':>9s} {'ppl':>9s} {'coverage':>9s}")
    print("-" * 55)
    scored = [(k, v) for k, v in result["domains"].items() if "loss" in v]
    for domain, row in sorted(scored, key=lambda kv: kv[1]["loss"]):
        print(f"{domain:16s} {row['rows']:7,d} {row['loss']:9.4f} {row['perplexity']:9.4f} "
              f"{row['response_coverage']:9.4f}")
    for domain, row in result["domains"].items():
        if "skipped" in row:
            print(f"{domain:16s} {row['rows']:7,d}   skipped: {row['skipped']}")


def print_reachability(table: Dict[str, Any]) -> None:
    names = list(table)
    domains = sorted(next(iter(table.values()))["coverage_by_domain"])
    header = f"{'domain':16s}"
    for name in names:
        header += f" {name[:20]:>20s}"
    print(header)
    print("-" * len(header))
    for domain in domains:
        line = f"{domain:16s}"
        for name in names:
            line += f" {table[name]['coverage_by_domain'][domain]:20.4f}"
        print(line)
    print()
    for name in names:
        print(f"  {name}: vocabulary {table[name]['vocab_size']:,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blend", default=str(DEFAULT_BLEND))
    parser.add_argument("--checkpoint", default=None, help="score per-domain loss for this checkpoint")
    parser.add_argument("--tokenizer-from", action="append", default=[],
                        help="compare vocabulary reachability across checkpoints; repeatable")
    parser.add_argument("--dev_fraction", type=float, default=0.01)
    parser.add_argument("--test_fraction", type=float, default=0.02)
    parser.add_argument("--tier3_row_fraction", type=float, default=0.02)
    parser.add_argument("--max_row_fraction_per_sentence", type=float, default=0.002)
    parser.add_argument("--split_seed", type=int, default=58)
    parser.add_argument("--min_response_characters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_rows", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--torch_threads", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if not args.checkpoint and not args.tokenizer_from:
        raise SystemExit("give --checkpoint, --tokenizer-from, or both")

    blend = Path(args.blend)
    domains = load_domain_map(blend)
    tiers = held_out_rows(
        blend,
        args.dev_fraction,
        args.test_fraction,
        args.tier3_row_fraction,
        args.max_row_fraction_per_sentence,
        args.split_seed,
        args.min_response_characters,
    )
    every_held_out = [pair for rows in tiers.values() for pair in rows]
    grouped = group_by_domain(every_held_out, domains)

    receipt: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "blend": str(blend),
        "held_out_rows": len(every_held_out),
        "rows_by_domain": {k: len(v) for k, v in sorted(grouped.items())},
        "non_claims": [
            "Per-domain loss is not comparable between checkpoints with different "
            "vocabularies; only reachability is.",
            "A domain label records which source file a row came from, not a "
            "verified competence. 'maths' means 'generated arithmetic', and a low "
            "loss on it means the model predicts those strings, not that it can "
            "do arithmetic it has not seen.",
            "Loss is not accuracy. Nothing here checks whether an answer is right.",
        ],
    }
    print(f"held-out rows {len(every_held_out):,} across {len(grouped)} domains\n")

    if args.checkpoint:
        result = score_domains(Path(args.checkpoint), grouped, args.batch_size, args.min_rows)
        print_domain_table(result)
        receipt["per_domain"] = result
        print()

    if args.tokenizer_from:
        table = reachability([Path(p) for p in args.tokenizer_from], grouped)
        print("vocabulary reachability (fraction of response tokens representable)\n")
        print_reachability(table)
        receipt["reachability"] = table

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
