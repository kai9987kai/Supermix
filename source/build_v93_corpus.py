"""Assemble the v93 training corpus.

v93 is the two-hemisphere, neurogenesis release
(docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md). Its corpus, contract D7, is v89's
corpus with four families added and nothing taken away:

    v89 omni (12), scratchpad (10), code (9), dialogue (96,108)   byte-copied
    new omni      impulse, ohms_current, spring_energy,
                  permutations, final_velocity                    40,000 each
    new code      code_range_sum, code_list_count, code_neg_index 20,000 each
    connectome    cns_type_count, cns_side_count, cns_pair_synapses
                  (30,000 each) + 30,000 prose rows about the same types
    english       the 9,983 english_foundations rows of datasets/v62

The v89 rows are copied byte-for-byte from `datasets/v89/v89_combined.jsonl`
rather than regenerated: the whole point of the warm start from
`output/v89_corpus/v89_corpus.pt` is that every v89 task keeps the rows the
checkpoint learned from, and a regenerated file would differ wherever a
generator has moved since (the v93 generators are additive, but "should be
identical" is not "is identical"). The dialogue rows inside that file are the
96,108 v86 inherited them from v74/v69, and ultimately from the v29 pipeline
jsonl; docs/V87_TRAINING_READINESS.md notes they carry no provenance clearance
for expanded use and they are neither expanded nor re-sourced here.

The new task families come from the same three generators the benchmark
adapts (`build_omni_corpus.V93_TASKS`, `build_code_corpus.V93_TASKS`,
`build_connectome_corpus.TASKS`), each run as a subprocess with its own
receipt and `--token_budget_report`, so the per-task distinct-prompt,
repetition and reply-length numbers are measured on the rows that ship. The
english rows are relabelled as language: their `task` key is dropped (the
trainer's probe-budget guard would otherwise treat `message_writing` as a
scored task) and the label is kept under `kind`; their `basic_math` siblings
are left out because the scratchpad family already teaches those shapes with
working shown, and a bare `79` reply to `498 - 419` would teach the opposite.

**The vocabulary.** v89's tokenizer is loaded from the checkpoint (never
rebuilt: `WordTokenizer.build` orders ids by frequency and a superset corpus
renumbers them) and run over the NEW rows only. The report says, per family,
what share of pieces it cannot represent and how many ids a
`WordTokenizer.extend` over that family would append (raw and lstripped
forms, as `build` admits them). `MAX_NEW_TOKEN_IDS` caps the union: over it,
the connectome population is what to shrink (`--max_types`), because type
names are where the new letter runs come from and the language and english
rows together add only a few hundred.

Run:  python source/build_v93_corpus.py
      python source/build_v93_corpus.py --vocab_only   (re-measure, no rebuild)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

V89_COMBINED = ROOT / "datasets" / "v89" / "v89_combined.jsonl"
V62_ENGLISH_MATH = ROOT / "datasets" / "v62" / "english_math_40k.jsonl"
V89_CHECKPOINT = ROOT / "output" / "v89_corpus" / "v89_corpus.pt"
OUT_DIR = ROOT / "datasets" / "v93"
COMBINED = OUT_DIR / "v93_combined.jsonl"
MANIFEST = OUT_DIR / "v93_manifest.json"
VOCABULARY_REPORT = OUT_DIR / "v93_vocabulary_report.json"

SEED = 93
PER_TASK = 40000
CODE_PER_TASK = 20000
CNS_PER_TASK = 30000
CNS_LANGUAGE_ROWS = 30000

#: The new families, by the task names their rows carry.
OMNI_V93 = ("impulse", "ohms_current", "spring_energy", "permutations", "final_velocity")
CODE_V93 = ("code_range_sum", "code_list_count", "code_neg_index")
CNS_V93 = ("cns_type_count", "cns_side_count", "cns_pair_synapses")

#: How many ids the union of new token types may append to v89's 8,679. Set
#: well under the 16,384 cap's headroom (~7,700) so the grown embedding stays
#: a small fraction of the trained one; the connectome population (top 1,000
#: types by neuron count, 194 distinct letter runs) is the lever if it trips.
MAX_NEW_TOKEN_IDS = 2500

#: The v62 topic whose rows are kept, and the one whose rows are not.
ENGLISH_TOPIC = "english_foundations"
EXCLUDED_TOPIC = "basic_math"


def run(label: str, argv: List[str]) -> float:
    print(f"\n=== {label} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    elapsed = time.time() - started
    print(f"    {elapsed:.0f}s", flush=True)
    return elapsed


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -- english rows ------------------------------------------------------------


def english_rows(path: Path = V62_ENGLISH_MATH) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """The english_foundations rows of the v62 file, relabelled as language.

    Every row of that file carries `topic` and `task`. Rows whose topic is
    `basic_math` are excluded; the rest keep `user` and `assistant`, get
    `domain: english_foundations`, and carry their v62 task label as `kind`
    so nothing downstream that keys on `task` -- the probe-budget guard, the
    benchmark's seen arm, this builder's own combine step -- takes them for
    a scored task.
    """

    kept: List[Dict[str, str]] = []
    kinds: Counter = Counter()
    excluded: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            topic = str(record.get("topic", ""))
            if topic != ENGLISH_TOPIC:
                excluded[topic] += 1
                continue
            kind = str(record.get("task", "unlabelled"))
            kinds[kind] += 1
            kept.append({
                "user": str(record["user"]),
                "assistant": str(record["assistant"]),
                "domain": ENGLISH_TOPIC,
                "kind": kind,
            })
    receipt = {
        "source": str(path),
        "topic_kept": ENGLISH_TOPIC,
        "rows": len(kept),
        "kinds": dict(sorted(kinds.items())),
        "excluded_by_topic": dict(sorted(excluded.items())),
        "relabelling": "task key dropped, kept as `kind`; domain set to english_foundations",
    }
    return kept, receipt


# -- the vocabulary ----------------------------------------------------------


def load_v89_tokenizer(checkpoint: Path = V89_CHECKPOINT):
    """v89's tokenizer, read from its checkpoint and never rebuilt."""

    import torch

    import mimomix_text as text_utils

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return text_utils.WordTokenizer.from_dict(payload["tokenizer"])


def new_token_types(tokenizer, texts: Iterable[str]) -> Tuple[Counter, int, int]:
    """What `WordTokenizer.extend` would append for these texts.

    Returns (new ids with their piece counts, pieces seen, pieces unknown).
    A piece the base lacks contributes its raw form and its lstripped form,
    exactly as `build` and `extend` admit tokens, so the count is the number
    of embedding rows the extension would add, not the number of words.
    """

    unknown: Counter = Counter()
    seen = 0
    missing = 0
    for text in texts:
        if tokenizer.reverse_digits:
            import mimomix_text as text_utils

            text = text_utils.reverse_digit_runs(text)
        for piece in tokenizer.pattern.findall(text):
            seen += 1
            if piece not in tokenizer.index:
                missing += 1
                unknown[piece] += 1
    new_ids: Counter = Counter()
    for piece, count in unknown.items():
        for variant in (piece, piece.lstrip()):
            if variant and variant not in tokenizer.index:
                new_ids[variant] += count
    return new_ids, seen, missing


def vocabulary_report(families: Dict[str, List[Dict[str, str]]], tokenizer,
                      cap: int = MAX_NEW_TOKEN_IDS) -> Dict[str, Any]:
    """Per family and in union: unknown rate and the ids extension would add."""

    per_family: Dict[str, Any] = {}
    union: Counter = Counter()
    for name, rows in families.items():
        texts = [field for row in rows for field in (row["user"], row["assistant"])]
        new_ids, seen, missing = new_token_types(tokenizer, texts)
        union.update(new_ids)
        per_family[name] = {
            "rows": len(rows),
            "pieces": seen,
            "unknown_pieces": missing,
            "unknown_rate": round(missing / max(1, seen), 6),
            "new_token_ids": len(new_ids),
            "most_common_new": [
                [token, count] for token, count in new_ids.most_common(15)
            ],
        }
    return {
        "schema": "supermix-v93-vocabulary-report-v1",
        "tokenizer": {
            "source": str(V89_CHECKPOINT),
            "vocab_size": tokenizer.vocab_size,
            "digit_tokens": tokenizer.digit_tokens,
            "reverse_digits": tokenizer.reverse_digits,
        },
        "rule": (
            "a piece absent from v89's index adds its raw form and its lstripped "
            "form, as WordTokenizer.build/extend admit tokens; counts are over "
            "user and assistant fields of the NEW rows only"
        ),
        "families": per_family,
        "union_new_token_ids": len(union),
        "union_most_common_new": [[t, c] for t, c in union.most_common(40)],
        "cap": cap,
        "within_cap": len(union) <= cap,
        "note": (
            "over the cap, shrink build_connectome_corpus's population (--max_types) "
            "and rebuild: type names are where the new letter runs come from"
        ),
    }


# -- assembly ------------------------------------------------------------------


def line_ending(path: Path) -> bytes:
    """The newline a jsonl file uses, so rows appended to a copy of it match.

    v89's builders opened their outputs in text mode, which on Windows writes
    CRLF; a combined file that mixed endings would still parse, but a byte
    comparison against the v89 rows would not be the plain one it should be.
    """

    with path.open("rb") as handle:
        first = handle.readline()
    return b"\r\n" if first.endswith(b"\r\n") else b"\n"


def combine(parts: Sequence[Tuple[str, Path]], english: List[Dict[str, str]],
            destination: Path = COMBINED) -> Dict[str, Any]:
    """Concatenate, counting per task and per source. No shuffle, as v89.

    Binary in, binary out: every line of every part is written exactly as
    read, so the v89 rows in the result are the bytes v89 trained on and a
    reader can check that with `cmp` rather than with a parser.
    """

    per_task: Counter = Counter()
    per_source: Dict[str, Dict[str, int]] = {}
    language_rows = 0
    language_by_domain: Counter = Counter()
    newline = line_ending(parts[0][1])
    with destination.open("wb") as out:
        for source_name, path in parts:
            counts = {"rows": 0, "task_rows": 0, "language_rows": 0}
            with path.open("rb") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    out.write(raw if raw.endswith(b"\n") else raw + newline)
                    record = json.loads(raw)
                    counts["rows"] += 1
                    if "task" in record:
                        per_task[str(record["task"])] += 1
                        counts["task_rows"] += 1
                    else:
                        language_rows += 1
                        counts["language_rows"] += 1
                        language_by_domain[str(record.get("domain", "?"))] += 1
            per_source[source_name] = counts
        for row in english:
            out.write(json.dumps(row).encode("utf-8") + newline)
        language_rows += len(english)
        language_by_domain[ENGLISH_TOPIC] += len(english)
        per_source["english_foundations"] = {
            "rows": len(english), "task_rows": 0, "language_rows": len(english),
        }
    return {
        "per_task": dict(per_task),
        "per_source": per_source,
        "language_rows": language_rows,
        "language_by_domain": dict(language_by_domain),
    }


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def families_from_files(omni: Path, code: Path, cns: Path,
                        english: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    cns_rows = load_rows(cns)
    return {
        "omni_v93": load_rows(omni),
        "code_v93": load_rows(code),
        "connectome_tasks": [r for r in cns_rows if "task" in r],
        "connectome_language": [r for r in cns_rows if "task" not in r],
        "english_foundations": english,
    }


def budget_summary(report: Dict[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    """The per-task token-budget lines a manifest reader needs, and no more."""

    tasks = (report.get("token_budget") or {}).get("tasks") or {}
    out: Dict[str, Any] = {}
    for name in names:
        entry = tasks.get(name)
        if entry is None:
            continue
        out[name] = {
            "response_median": entry["response_median"],
            "response_p95": entry["response_p95"],
            "response_max": entry["response_max"],
            "turn_max": entry["turn_max"],
            "dropped_fraction": entry["dropped_fraction"],
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab_only", action="store_true",
                        help="re-measure the vocabulary from the files on disk; no rebuild")
    parser.add_argument("--max_types", type=int, default=None,
                        help="connectome population cap passed to build_connectome_corpus "
                             "(its default otherwise); the lever if the vocabulary cap trips")
    parser.add_argument("--max_new_token_ids", type=int, default=MAX_NEW_TOKEN_IDS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    omni = OUT_DIR / "v93_omni.jsonl"
    code = OUT_DIR / "v93_code.jsonl"
    cns = OUT_DIR / "v93_connectome.jsonl"
    timings: Dict[str, float] = {}
    started_all = time.time()

    if not args.vocab_only:
        timings["omni"] = run("omni (5 new science tasks, solver-verified, wide phrasings)", [
            str(SOURCE / "build_omni_corpus.py"),
            "--per_task", str(PER_TASK), "--seed", str(SEED),
            "--output", str(omni), "--report", str(OUT_DIR / "v93_omni.report.json"),
            "--token_budget_report",
            # As v89: the natural forms minus the three held out per task.
            "--natural_phrasings",
            # As v89: `ohms_current` is a division task and writes its division
            # the way v89's power/acceleration/molarity rows do.
            "--long_division",
            *[flag for name in OMNI_V93 for flag in ("--task", name)],
        ])
        timings["code"] = run("code (3 new execution-verified tracing tasks)", [
            str(SOURCE / "build_code_corpus.py"),
            "--per_task", str(CODE_PER_TASK), "--seed", str(SEED),
            "--output", str(code), "--report", str(OUT_DIR / "v93_code.report.json"),
            "--token_budget_report",
            *[flag for name in CODE_V93 for flag in ("--task", name)],
        ])
        cns_argv = [
            str(SOURCE / "build_connectome_corpus.py"),
            "--per_task", str(CNS_PER_TASK), "--language_rows", str(CNS_LANGUAGE_ROWS),
            "--seed", str(SEED),
            "--output", str(cns), "--report", str(OUT_DIR / "v93_connectome.report.json"),
            "--token_budget_report",
        ]
        if args.max_types is not None:
            cns_argv += ["--max_types", str(args.max_types)]
        timings["connectome"] = run(
            "connectome (3 lookup tasks + prose rows, from the CC-BY male-CNS tables)",
            cns_argv,
        )

    english, english_receipt = english_rows()
    newline = line_ending(V89_COMBINED)
    (OUT_DIR / "v93_english_foundations.jsonl").write_bytes(
        b"".join(json.dumps(row).encode("utf-8") + newline for row in english))

    print("\n=== vocabulary (v89 tokenizer over the new rows) ===", flush=True)
    started = time.time()
    tokenizer = load_v89_tokenizer()
    families = families_from_files(omni, code, cns, english)
    vocabulary = vocabulary_report(families, tokenizer, cap=args.max_new_token_ids)
    VOCABULARY_REPORT.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    timings["vocabulary"] = time.time() - started
    for name, entry in vocabulary["families"].items():
        print(f"    {name:<22} rows {entry['rows']:>7,}  unknown rate "
              f"{entry['unknown_rate']:.4f}  new ids {entry['new_token_ids']:>5}")
    print(f"    union new ids {vocabulary['union_new_token_ids']} "
          f"(cap {vocabulary['cap']})  {timings['vocabulary']:.0f}s")
    if not vocabulary["within_cap"]:
        raise SystemExit(
            f"the new rows would append {vocabulary['union_new_token_ids']} ids, over the "
            f"cap of {vocabulary['cap']}: rebuild with a smaller --max_types"
        )

    if args.vocab_only:
        print(f"\nvocabulary report -> {VOCABULARY_REPORT}")
        return 0

    print("\n=== combining ===", flush=True)
    started = time.time()
    counts = combine(
        [("v89_combined", V89_COMBINED), ("omni_v93", omni),
         ("code_v93", code), ("connectome", cns)],
        english,
    )
    timings["combine"] = time.time() - started
    print(f"    {timings['combine']:.0f}s", flush=True)

    omni_report = read_json(OUT_DIR / "v93_omni.report.json")
    code_report = read_json(OUT_DIR / "v93_code.report.json")
    cns_report = read_json(OUT_DIR / "v93_connectome.report.json")
    v89_manifest = read_json(ROOT / "datasets" / "v89" / "v89_manifest.json")

    total = sum(counts["per_task"].values()) + counts["language_rows"]
    manifest = {
        "schema": "supermix-v93-corpus-v1",
        "output": str(COMBINED),
        "seed": SEED,
        "rows": total,
        "task_rows": sum(counts["per_task"].values()),
        "language_rows": counts["language_rows"],
        "language_by_domain": counts["language_by_domain"],
        "per_task": counts["per_task"],
        "per_source": counts["per_source"],
        "new_tasks": {
            "omni": list(OMNI_V93), "code": list(CODE_V93), "connectome": list(CNS_V93),
        },
        "changes_against_v89": [
            f"5 new omni tasks ({', '.join(OMNI_V93)}), {PER_TASK:,} rows each, "
            "verified by nexus_solver / science_plan, --natural_phrasings "
            "--long_division as v89",
            f"3 new code tasks ({', '.join(CODE_V93)}), {CODE_PER_TASK:,} rows each, "
            "verified by execution",
            f"3 connectome lookup tasks ({', '.join(CNS_V93)}), {CNS_PER_TASK:,} rows "
            f"each, plus {CNS_LANGUAGE_ROWS:,} prose rows (domain connectome, no task "
            "key), all looked up in the CC-BY male-CNS tables",
            f"{english_receipt['rows']:,} english_foundations rows from "
            "datasets/v62/english_math_40k.jsonl as language rows (domain "
            "english_foundations, task label kept as `kind`); its basic_math rows "
            "are excluded",
        ],
        "unchanged_from_v89": [
            f"every row of {V89_COMBINED.name} is byte-copied: the 31 v89 task labels "
            f"({v89_manifest['task_rows']:,} rows) and the {v89_manifest['language_rows']:,} "
            "dialogue rows",
            "the v89 generators are untouched, so the 30-task benchmark fingerprint "
            "3b99a446cd533be9bc5f8ae57d1310b4 reproduces through "
            "eval_problem_solving.py --task_set v89",
        ],
        "held_out_from_training": (
            "natural_phrasings.HELD_OUT_PER_TASK withholds the last three forms of "
            "each omni task -- 36 over the twelve v89 tasks and 15 over the five new "
            "ones -- so eval_natural_phrasing.py measures generalisation and not "
            "recall. The connectome tasks have no held-out population: their "
            "benchmark is recall over the same types and pairs (see NON_CLAIMS in "
            "build_connectome_corpus and eval_problem_solving)."
        ),
        "token_budget": {
            "sequence_length": (omni_report.get("token_budget") or {}).get("sequence_length"),
            "benchmark_generation_cap": 96,
            "note": "response counts exclude EOS; the benchmark's 96-token cap "
                    "includes it, so response_p95 <= 95 is what fits",
            "omni": budget_summary(omni_report, OMNI_V93),
            "code": budget_summary(code_report, CODE_V93),
            "connectome": budget_summary(cns_report, list(CNS_V93) + ["connectome_language"]),
        },
        "repetition": {
            "omni": {name: {"distinct_prompts": omni_report["distinct_prompts"].get(name),
                            "repetition": omni_report["repetition"].get(name)}
                     for name in OMNI_V93},
            "code": {name: {"distinct_prompts": code_report["distinct_prompts"].get(name),
                            "repetition": code_report["repetition"].get(name),
                            "distinct_capacity": code_report["distinct_capacity"].get(name)}
                     for name in CODE_V93},
            "connectome": {name: {"distinct_prompts": cns_report["distinct_prompts"].get(name),
                                  "distinct_facts": cns_report["distinct_facts"].get(name),
                                  "repetition": cns_report["repetition"].get(name),
                                  "majority_answer_share":
                                      cns_report["majority_answer_share"].get(name)}
                           for name in CNS_V93},
        },
        "verification": {
            "omni": {
                "verified_by": omni_report.get("verified_by"),
                "dropped_failing_verification": omni_report.get("dropped_failing_verification"),
            },
            "code": {
                "verified_by": code_report.get("verified_by"),
                "drop_rate": code_report.get("drop_rate"),
                "drop_reasons": code_report.get("drop_reasons"),
            },
            "connectome": {
                "verified_by": cns_report.get("verified_by"),
                "dropped_failing_lookup": cns_report.get("dropped_failing_lookup"),
                "population": cns_report.get("population"),
            },
            "english_foundations": {
                "verified_by": "none (language rows)", **english_receipt,
            },
        },
        "vocabulary": {
            "report": str(VOCABULARY_REPORT),
            "union_new_token_ids": vocabulary["union_new_token_ids"],
            "cap": vocabulary["cap"],
            "per_family_new_ids": {
                name: entry["new_token_ids"] for name, entry in vocabulary["families"].items()
            },
        },
        "provenance": {
            "v89_rows": (
                f"byte-copied from {V89_COMBINED}; that file was assembled by "
                "output/v87_measurements/build_v89_corpus.py"
            ),
            "dialogue_rows": (
                "the 96,108 `domain: dialogue` rows are the v86 inherited ones, copied "
                "byte-identical through v87, v88, v89 and now v93 (v86 took them from "
                "v74/v69; origin the v29 pipeline jsonl). docs/V87_TRAINING_READINESS.md "
                "records that they have no provenance clearance for expanded use; they "
                "are not expanded here"
            ),
            "connectome_rows": (
                "generated from datasets/v91_malecns/malecns_types.npz and "
                "datasets/v93_malecns/malecns_sided_types.npz, both derived from the "
                "FlyEM male-cns v1.0 flat connectome (CC BY 4.0); attribution in "
                "build_connectome_corpus.py"
            ),
            "english_rows": english_receipt["source"],
        },
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "wall_seconds": round(time.time() - started_all, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nrows          {total:,}")
    print(f"  task rows   {manifest['task_rows']:,} over {len(counts['per_task'])} tasks")
    print(f"  language    {counts['language_rows']:,}  {counts['language_by_domain']}")
    for name in (*OMNI_V93, *CODE_V93, *CNS_V93):
        print(f"  {name:<20} {counts['per_task'].get(name, 0):>7,}")
    print(f"\nwall {manifest['wall_seconds']:.0f}s")
    print(f"wrote {COMBINED}")
    print(f"manifest -> {MANIFEST}")
    print(f"vocabulary report -> {VOCABULARY_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
