"""Can v86 already do the steps a decomposed science division would ask for?

The dose-response says a one-shot division returning a 3-digit quotient is
right 12.5% of the time, and that the divisor's width barely matters. The
`division` task avoids this by splitting the *quotient* by place value, so each
written step returns a single significant digit -- and it scores 1.000.

Before rewriting `power`, `molarity` and `acceleration` to do the same, this
asks whether the steps that rewrite would produce are already inside the
model's competence. Each 3-digit-quotient problem from the sweep is split into
its place-value parts and each part is put to the model on its own.

The test is deliberately unfair to the hypothesis. `division` trains on
divisors 2-9 and quotients 11-60, and these sub-problems have two-digit
divisors and quotients like 200, so every one of them is out of the
distribution the model learned this format on. A model that answers them anyway
is a model for which the decomposition is the only missing piece; a model that
cannot is a warning that place-value splitting alone will not fix these tasks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from eval_problem_solving import (  # noqa: E402
    extract_answer, generate_reply, is_correct, load_talk_checkpoint, wilson_interval,
)

CHECKPOINT = ROOT / "output" / "v86_corpus" / "v86_corpus.pt"
SWEEP = Path(__file__).parent / "division_dose_response_replies.json"
OUT = Path(__file__).with_suffix(".json")


def place_value_parts(dividend: int, divisor: int):
    """``8178 / 47`` -> ``[(4700, 100), (3290, 70), (188, 4)]``.

    The quotient is split by place value and each part multiplied back by the
    divisor, so every sub-division is exact by construction. Zero places are
    dropped: writing ``0 / 47 = 0`` would teach a step that is always noise.
    """

    quotient = dividend // divisor
    assert quotient * divisor == dividend, "these problems divide exactly"
    parts = []
    for power_of_ten in range(len(str(quotient)) - 1, -1, -1):
        place = 10 ** power_of_ten
        digit = (quotient // place) % 10
        if digit:
            parts.append((digit * place * divisor, digit * place))
    return parts


def main() -> int:
    records = json.loads(SWEEP.read_text(encoding="utf-8"))
    hard = [r for r in records if r["cell"].endswith("3-digit")]

    model, tokenizer, _ = load_talk_checkpoint(str(CHECKPOINT))
    model.eval()

    total = right = 0
    whole_right = 0
    by_place = {}
    examples = []
    seen = set()
    for record in hard:
        dividend, divisor = record["work"], record["time_s"]
        if (dividend, divisor) in seen:
            continue
        seen.add((dividend, divisor))
        parts = place_value_parts(dividend, divisor)
        all_ok = True
        for part_dividend, part_quotient in parts:
            prompt = f"Solve this basic math problem: {part_dividend} / {divisor}"
            reply = generate_reply(model, tokenizer, prompt, max_new_tokens=96)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            ok = is_correct(extract_answer(text), float(part_quotient))
            total += 1
            right += int(ok)
            all_ok &= ok
            width = len(str(part_quotient))
            bucket = by_place.setdefault(f"{width}-digit part", {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(ok)
            if len(examples) < 12:
                examples.append({"prompt": prompt, "reply": text[:110],
                                 "expected": part_quotient, "correct": ok})
        whole_right += int(all_ok)

    for bucket in by_place.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 4)
    low, high = wilson_interval(right, total)
    whole_low, whole_high = wilson_interval(whole_right, len(seen))
    report = {
        "schema": "supermix-v87-subdivision-probe-v1",
        "checkpoint": str(CHECKPOINT),
        "question": (
            "On the problems where a one-shot division scores 0.125, can the "
            "model do the place-value sub-divisions that a decomposed format "
            "would ask for instead?"
        ),
        "sub_problems": total,
        "sub_correct": right,
        "sub_accuracy": round(right / total, 4),
        "sub_accuracy_95ci": [round(low, 4), round(high, 4)],
        "problems": len(seen),
        "all_parts_correct": whole_right,
        "all_parts_accuracy": round(whole_right / len(seen), 4),
        "all_parts_95ci": [round(whole_low, 4), round(whole_high, 4)],
        "by_part_width": by_place,
        "examples": examples,
        "non_claims": [
            "These sub-problems are out of the distribution `division` was "
            "trained on (divisors 2-9, quotients 11-60), so this is a lower "
            "bound on what a model trained on the decomposed format would do.",
            "Getting every part right is necessary for the decomposed format "
            "to help, not sufficient: the model would also have to sum the "
            "partial quotients, which this does not test.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("sub_problems", "sub_accuracy", "all_parts_accuracy",
                       "by_part_width")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
