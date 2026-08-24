"""Score a v58 checkpoint on the withheld sentences' own tokens.

The tier-3 number in `train_mimomix_generalisation.py` is averaged over every
token of every tier-3 response. That understates the effect it is trying to
measure, because a tier-3 response is *mostly* made of sentences the model saw
thousands of times -- only one of its two-to-five sentences is withheld. A
diluted average is the honest thing to report as "tier 3", but it is not the
answer to "how well does the model handle a sentence it has never seen?".

This module answers that directly. Within the **same** tier-3 rows it scores two
disjoint token sets:

``unseen_sentence_tokens``
    Tokens that lie inside a withheld sentence.

``seen_sentence_tokens``
    Every other response token in those same rows.

The second is the control, and it is what makes the first interpretable. Both
sets come from the same responses, the same prompts, the same packing and the
same forward passes, so a difference between them cannot be a property of the
rows -- only of the sentences. Reporting the unseen number alone would leave a
reader unable to tell a hard sentence from a hard row.

Usage::

    python source/eval_mimomix_unseen_sentences.py --run_dir output/v58_full
    python source/eval_mimomix_unseen_sentences.py --run_dir output/v58_full --run_dir output/v58_ablation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from train_mimomix_generalisation import load_corpus_pairs  # noqa: E402
from train_mimomix_talk import atomic_json, evaluate, load_talk_checkpoint  # noqa: E402

RECEIPT_SCHEMA = "supermix-v58-unseen-sentence-tokens-v1"

Pair = Tuple[str, str]


def withheld_character_spans(response: str, held_out: Set[str]) -> List[Tuple[int, int]]:
    """Character spans of every withheld sentence occurring in `response`.

    Sentences are located by scanning the response's own sentence split rather
    than by substring search, so a withheld sentence that also happens to be a
    prefix of a longer kept sentence cannot claim the longer one's characters.
    """

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for sentence in splits.split_sentences(response):
        start = response.find(sentence, cursor)
        if start < 0:
            continue
        end = start + len(sentence)
        cursor = end
        if sentence in held_out:
            spans.append((start, end))
    return spans


def masked_turn(
    pair: Pair,
    tokenizer: text_utils.WordTokenizer,
    held_out: Set[str],
    keep: str,
) -> Tuple[List[int], List[int]]:
    """Build `(ids, labels)` for one turn, scoring only the requested tokens.

    Args:
        keep: ``"unseen"`` scores only tokens inside a withheld sentence;
            ``"seen"`` scores only the other response tokens.

    Returns:
        Token ids and labels, with `-100` everywhere the loss must skip.
    """

    user, response = pair
    ids, prompt_length = tokenizer.encode_turn(user, response)
    labels = [-100] * len(ids)

    spans = withheld_character_spans(response, held_out)
    pieces = text_utils.TOKEN_PATTERN.findall(response)

    offset = 0
    for index, piece in enumerate(pieces):
        start, end = offset, offset + len(piece)
        offset = end
        # A token belongs to a withheld sentence when its non-whitespace body
        # falls inside the span. Tokens carry their own leading whitespace, so
        # comparing raw starts would misassign the token that follows a span.
        body_start = start + (len(piece) - len(piece.lstrip()))
        inside = any(low <= body_start < high for low, high in spans)
        if inside if keep == "unseen" else not inside:
            position = prompt_length + index
            if position < len(ids):
                labels[position] = ids[position]

    return ids, labels


def build_tensors(
    pairs: Sequence[Pair],
    tokenizer: text_utils.WordTokenizer,
    held_out: Set[str],
    keep: str,
    sequence_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pack turns into blocks, keeping the same packing as training."""

    stream_ids: List[int] = []
    stream_labels: List[int] = []
    for pair in pairs:
        ids, labels = masked_turn(pair, tokenizer, held_out, keep)
        stream_ids.extend(ids)
        stream_labels.extend(labels)

    usable = (len(stream_ids) // sequence_length) * sequence_length
    if usable == 0:
        raise ValueError("not enough tokens for a single sequence")
    inputs = torch.tensor(stream_ids[:usable], dtype=torch.long).view(-1, sequence_length)
    labels_t = torch.tensor(stream_labels[:usable], dtype=torch.long).view(-1, sequence_length)
    return inputs, labels_t, int((labels_t != -100).sum())


def evaluate_run(run_dir: Path, batch_size: int = 16) -> Dict[str, Any]:
    """Score one finished v58 arm on both token sets."""

    receipt_path = run_dir / "generalisation_results.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / f"{receipt['run_name']}.pt"

    model, tokenizer, _ = load_talk_checkpoint(str(checkpoint))
    held_out = set(receipt["held_out_sentences"])
    settings = receipt["split"]["settings"]
    sequence_length = receipt["hyperparameters"]["sequence_length"]

    # Rebuild the identical split. It is deterministic in the recorded seed, and
    # `verify_split` re-checks it rather than trusting the reconstruction.
    pairs = load_corpus_pairs(receipt["split"]["source"])
    split = splits.build_generalisation_split(
        pairs,
        dev_fraction=settings["dev_fraction"],
        test_fraction=settings["test_fraction"],
        target_row_fraction=settings["target_row_fraction"],
        max_row_fraction_per_sentence=settings["max_row_fraction_per_sentence"],
        seed=settings["seed"],
        source=receipt["split"]["source"],
    )
    splits.verify_split(split)
    if set(split.held_out_sentences) != held_out:
        raise ValueError(
            "rebuilt split withheld different sentences than the receipt records; "
            "the corpus or the split rule changed since this run"
        )

    rows = split.tier3_unseen_sentence
    measured: Dict[str, Any] = {}
    for keep in ("unseen", "seen"):
        inputs, labels, counted = build_tensors(
            rows, tokenizer, held_out, keep, sequence_length
        )
        metrics = evaluate(model, inputs, labels, batch_size)
        measured[f"{keep}_sentence_tokens"] = {**metrics, "labelled_tokens": counted}

    unseen = measured["unseen_sentence_tokens"]["loss"]
    seen = measured["seen_sentence_tokens"]["loss"]
    return {
        "schema": RECEIPT_SCHEMA,
        "run_name": receipt["run_name"],
        "arm": receipt["arm"],
        "thinking_core": receipt["thinking_core"],
        "checkpoint": str(checkpoint),
        "tier3_rows": len(rows),
        "held_out_sentences": len(held_out),
        "diluted_tier3_loss": receipt["tiers"]["tier3_unseen_sentence"]["loss"],
        "measured": measured,
        "unseen_minus_seen_nats": round(unseen - seen, 6),
        "perplexity_ratio": round(math.exp(unseen - seen), 4),
        "note": (
            "both token sets come from the same tier-3 rows, prompts, packing and "
            "forward passes, so the difference isolates the sentence rather than "
            "the row; the diluted tier-3 loss averages the two"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score v58 checkpoints on withheld sentences' own tokens"
    )
    parser.add_argument("--run_dir", action="append", required=True,
                        help="a finished v58 output directory; repeatable")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--output", default=str(SOURCE_DIR.parent / "output" / "v58_unseen_sentence_tokens.json"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))

    results = [evaluate_run(Path(directory), args.batch_size) for directory in args.run_dir]
    for row in results:
        measured = row["measured"]
        print(f"== {row['run_name']} (thinking core {row['thinking_core']}) ==")
        print(f"  tier-3 rows                 {row['tier3_rows']:,}")
        print(f"  diluted tier-3 loss         {row['diluted_tier3_loss']:.4f}")
        for name in ("seen_sentence_tokens", "unseen_sentence_tokens"):
            entry = measured[name]
            print(f"  {name:<27} {entry['loss']:.4f}  ppl {entry['perplexity']:.4f}  "
                  f"({entry['labelled_tokens']:,} tokens)")
        print(f"  unseen - seen               {row['unseen_minus_seen_nats']:+.4f} nats "
              f"({row['perplexity_ratio']:.3f}x perplexity)")
        print()

    atomic_json(Path(args.output), {"schema": RECEIPT_SCHEMA, "runs": results})
    print(f"receipt {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
