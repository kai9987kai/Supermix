"""Generate a scientific training corpus, verified by the NexusMind solver.

## Why this exists

When asked to "deeply expand the model's knowledge" before v74, the honest
answer was that the corpus could not deliver it: the dialogue portion is 19.8%
one repeated fragment, and v74's ten task types are all synthetic arithmetic.
A model trained on that knows arithmetic and nothing about the world.

`nexus_solver.py` changes what is possible. It is a deterministic solver over
`Fraction`/`Decimal` covering mechanics, energy, fluids, thermodynamics,
electromagnetism, waves, chemistry, algebra, geometry and combinatorics, and it
emits a step-by-step derivation with a SHA-256 receipt. It is a **verified
knowledge source**, and it can be asked an unlimited number of questions.

So: generate scientific problems, work them out, and have the solver check
every single one. Any row whose worked answer disagrees with the solver is
dropped, not shipped. The corpus is correct by construction rather than by
assumption -- no previous corpus in this repository had that property.

## Two design decisions taken from earlier failures

**Phrasing varies, because v74 was format-brittle.** v74 scored 0.894 on its
benchmark and got 0 of 5 naturally-typed questions right, because every task
had exactly one template and the model learned the template. Here each task
carries several phrasings, and the one shown to the model is decoupled from the
canonical query handed to the solver. The model sees variety; the oracle always
sees a form it parses.

**The response never ends in a unit.** Answers are extracted as the last number
in the reply. `total 5 m/s^2` extracts as **2**. Every response therefore ends
`total <number>` with nothing after it, and units live in the prose before it.
This is the kind of thing that silently halves a benchmark score.

## What is deliberately excluded

Tasks whose answers are irrational are generated with rounded values and
verified against the solver within a relative tolerance, rather than exactly.
Where a task cannot produce a value a digit-level model could plausibly learn,
it is left out rather than padded with noise -- training on `294.1995` teaches
the shape of a float, not physics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from nexus_solver import solve_problem  # noqa: E402

#: Relative tolerance when comparing a generated answer to the solver's.
#: Exact-rational tasks agree to the bit; the constant-bearing ones (g, pi)
#: are rounded for legibility and must still land within this.
TOLERANCE = 1e-6

#: Standard gravity as the solver uses it, so rounded answers agree.
GRAVITY = 9.80665

#: Consecutive failures to produce a new prompt before a task is judged
#: exhausted. Generous, because a near-full space still yields hits.
EXHAUSTION_MISSES = 5000


@dataclass
class OmniProblem:
    task: str
    domain: str
    prompt: str
    response: str
    answer: float
    unit: str
    canonical: str
    params: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> Dict[str, str]:
        return {
            "user": self.prompt,
            "assistant": self.response,
            "domain": self.domain,
            "task": self.task,
        }


def _number(value: float) -> str:
    """Render a number the way the corpus should read it.

    Integers stay integral: `60`, not `60.0`. A digit-level tokenizer spends
    real capacity on a trailing `.0` that carries no information.
    """

    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{round(value, 4):g}"


def decompose_product(a: int, b: int) -> str:
    """Show the working for `a x b`, by place value.

    v79 learned the physics and failed the arithmetic. Measured on the
    finished model, asking for the force from `mass x acceleration`:

        single-digit operands   12/12 correct
        two-digit                9/12
        three-digit              1/12

    The cause was this module's first version, which wrote
    `167 x 11 = 1837` as a single step. v66 established that this model
    cannot do multi-digit multiplication in one jump; v74's arithmetic
    corpus never asks it to, splitting every product by place value -- and
    it never uses an operand above 99.

    Widening the parameter ranges (needed to stop duplicate prompts) made
    that far worse, since products then reached 24,000 with no working
    shown at all. Two separately-correct fixes interacting badly.

    So the working is written out, with running sums, for operands of any
    size:

        167 x 11 -> "100 x 11 = 1100, 60 x 11 = 660, 7 x 11 = 77,
                     1100 + 660 = 1760, 1760 + 77 = 1837"
    """

    digits = str(abs(a))
    width = len(digits)
    # Every digit position, including zeros. v74 writes `80 x 3 = 240,
    # 0 x 3 = 0` rather than dropping the units term, so the shape is the
    # same for every problem -- and that is the corpus that scored 0.93.
    parts = [int(d) * 10 ** (width - 1 - i) for i, d in enumerate(digits)]
    if a < 0:
        parts = [-p for p in parts]

    pieces = [f"{part} x {b} = {part * b}" for part in parts]
    if len(parts) <= 2:
        # v74's exact form: two partial products, then the total. It does not
        # write the addition out, and the model learned to do it -- adding a
        # running-sum step here would depart from the only format measured to
        # work.
        return ", ".join(pieces)

    running = parts[0] * b
    for part in parts[1:]:
        nxt = running + part * b
        pieces.append(f"{running} + {part * b} = {nxt}")
        running = nxt
    return ", ".join(pieces)


def _pick(rng: random.Random, templates: List[str], **values) -> str:
    return rng.choice(templates).format(**{k: _number(v) if isinstance(v, (int, float))
                                           else v for k, v in values.items()})


# -- task generators --------------------------------------------------------
#
# Each returns a problem whose `canonical` form the solver parses, and whose
# `prompt` is one of several phrasings a person might actually type.


def _force(rng: random.Random) -> OmniProblem:
    mass = rng.randint(11, 99)
    accel = rng.randint(2, 9)
    answer = mass * accel
    prompt = _pick(rng, [
        "A body of mass {m} kg has an acceleration of {a} m/s^2. What is the force?",
        "mass {m} kg acceleration {a} m/s^2 find the force",
        "Find the force on a {m} kg mass accelerating at {a} m/s^2.",
        "What force acts on mass {m} kg with acceleration {a} m/s^2?",
        "Given mass {m} kg and acceleration {a} m/s^2, compute the force.",
    ], m=mass, a=accel)
    response = (f"force = mass x acceleration, {decompose_product(mass, accel)}, "
                f"the force is {answer} newtons, total {answer}")
    return OmniProblem("force", "physics", prompt, response, float(answer), "N",
                       f"mass {mass} kg acceleration {accel} m/s^2 find the force",
                       {"m": mass, "a": accel})


def _acceleration(rng: random.Random) -> OmniProblem:
    mass = rng.randint(2, 200)
    accel = rng.randint(2, 60)
    force = mass * accel  # chosen so the division is exact
    prompt = _pick(rng, [
        "A force of {f} N acts on a mass of {m} kg. What is the acceleration?",
        "force {f} N mass {m} kg find acceleration",
        "Find the acceleration produced by {f} N on {m} kg.",
        "What acceleration results from a {f} N force on a {m} kg body?",
    ], f=force, m=mass)
    response = (f"acceleration = force / mass, {force} / {mass} = {accel}, "
                f"the acceleration is {accel} metres per second squared, total {accel}")
    return OmniProblem("acceleration", "physics", prompt, response, float(accel), "m/s^2",
                       f"force {force} N mass {mass} kg find acceleration",
                       {"F": force, "m": mass})


def _momentum(rng: random.Random) -> OmniProblem:
    mass = rng.randint(11, 99)
    velocity = rng.randint(2, 9)
    answer = mass * velocity
    prompt = _pick(rng, [
        "A {m} kg object moves with velocity {v} m/s. What is its momentum?",
        "mass {m} kg velocity {v} m/s find momentum",
        "Find the momentum of a mass {m} kg travelling at velocity {v} m/s.",
        "What is the linear momentum for mass {m} kg and velocity {v} m/s?",
    ], m=mass, v=velocity)
    response = (f"momentum = mass x velocity, {decompose_product(mass, velocity)}, "
                f"the momentum is {answer} kilogram metres per second, total {answer}")
    return OmniProblem("momentum", "physics", prompt, response, float(answer), "kg*m/s",
                       f"mass {mass} kg velocity {velocity} m/s find momentum",
                       {"m": mass, "v": velocity})


def _kinetic_energy(rng: random.Random) -> OmniProblem:
    # Halve the mass *first*, so the only product is `squared x half_mass` --
    # a two-digit by one-digit multiply, which is v74's proven shape. The
    # previous ranges asked for `25 x 25 = 625` then `30 x 625 = 18750`, both
    # in one step, and the task scored 0.00.
    mass = rng.randrange(2, 20, 2)
    velocity = rng.randint(2, 9)
    squared = velocity * velocity
    half_mass = mass // 2
    answer = half_mass * squared
    prompt = _pick(rng, [
        "A mass of {m} kg moves at velocity {v} m/s. Find the kinetic energy.",
        "mass {m} kg velocity {v} m/s kinetic energy",
        "What is the kinetic energy of a {m} kg body at {v} m/s?",
        "Compute the kinetic energy for mass {m} kg and speed {v} m/s.",
    ], m=mass, v=velocity)
    response = (f"kinetic energy = half x mass x velocity squared, "
                f"half of {mass} = {half_mass}, "
                f"velocity squared = {velocity} x {velocity} = {squared}, "
                f"{decompose_product(squared, half_mass)}, "
                f"the kinetic energy is {answer} joules, total {answer}")
    return OmniProblem("kinetic_energy", "physics", prompt, response, float(answer), "J",
                       f"mass {mass} kg velocity {velocity} m/s kinetic energy",
                       {"m": mass, "v": velocity})


def _work(rng: random.Random) -> OmniProblem:
    force = rng.randint(11, 99)
    distance = rng.randint(2, 9)
    answer = force * distance
    prompt = _pick(rng, [
        "A force of {f} N moves an object {d} m. How much work is done?",
        "force {f} N distance {d} m work done",
        "Find the work done by {f} N acting over {d} m.",
        "What work is done when a {f} N force acts through {d} m?",
    ], f=force, d=distance)
    response = (f"work = force x distance, {decompose_product(force, distance)}, "
                f"the work done is {answer} joules, total {answer}")
    return OmniProblem("work", "physics", prompt, response, float(answer), "J",
                       f"force {force} N distance {distance} m work done",
                       {"F": force, "d": distance})


def _power(rng: random.Random) -> OmniProblem:
    time = rng.randint(2, 100)
    power = rng.randint(2, 300)
    work = power * time  # exact division
    prompt = _pick(rng, [
        "{w} J of work is done in {t} s. What is the power?",
        "work {w} J time {t} s power",
        "Find the power when {w} J is delivered over {t} s.",
        "What power corresponds to {w} joules in {t} seconds?",
    ], w=work, t=time)
    response = (f"power = work / time, {work} / {time} = {power}, "
                f"the power is {power} watts, total {power}")
    return OmniProblem("power", "physics", prompt, response, float(power), "W",
                       f"work {work} J time {time} s power",
                       {"W": work, "t": time})


def _voltage(rng: random.Random) -> OmniProblem:
    current = rng.randint(11, 99)
    resistance = rng.randint(2, 9)
    answer = current * resistance
    prompt = _pick(rng, [
        "A current of {i} A flows through {r} ohm. What is the voltage?",
        "current {i} A resistance {r} ohm find voltage",
        "Find the potential difference across {r} ohm carrying {i} A.",
        "What voltage drives {i} A through a {r} ohm resistor?",
    ], i=current, r=resistance)
    response = (f"voltage = current x resistance, {decompose_product(current, resistance)}, "
                f"the voltage is {answer} volts, total {answer}")
    return OmniProblem("voltage", "physics", prompt, response, float(answer), "V",
                       f"current {current} A resistance {resistance} ohm find voltage",
                       {"I": current, "R": resistance})


def _electrical_power(rng: random.Random) -> OmniProblem:
    voltage = rng.randint(11, 99)
    current = rng.randint(2, 9)
    answer = voltage * current
    prompt = _pick(rng, [
        "A device runs at {v} V drawing {i} A. What is the electrical power?",
        "voltage {v} V current {i} A electrical power",
        "Find the power dissipated at {v} V and {i} A.",
        "What electrical power is used at {v} volts and {i} amps?",
    ], v=voltage, i=current)
    response = (f"power = voltage x current, {decompose_product(voltage, current)}, "
                f"the power is {answer} watts, total {answer}")
    return OmniProblem("electrical_power", "physics", prompt, response, float(answer), "W",
                       f"voltage {voltage} V current {current} A electrical power",
                       {"V": voltage, "I": current})


def _wave_speed(rng: random.Random) -> OmniProblem:
    frequency = rng.randint(11, 99)
    wavelength = rng.randint(2, 9)
    answer = frequency * wavelength
    prompt = _pick(rng, [
        "A wave has frequency {f} Hz and wavelength {w} m. What is its speed?",
        "frequency {f} Hz wavelength {w} m wave speed",
        "Find the speed of a wave of frequency {f} Hz and wavelength {w} m.",
        "What is the wave speed at {f} Hz with a {w} m wavelength?",
    ], f=frequency, w=wavelength)
    response = (f"wave speed = frequency x wavelength, {decompose_product(frequency, wavelength)}, "
                f"the wave speed is {answer} metres per second, total {answer}")
    return OmniProblem("wave_speed", "physics", prompt, response, float(answer), "m/s",
                       f"frequency {frequency} Hz wavelength {wavelength} m wave speed",
                       {"f": frequency, "lambda": wavelength})


def _molarity(rng: random.Random) -> OmniProblem:
    volume = rng.randint(1, 60)
    concentration = rng.randint(1, 60)
    moles = concentration * volume  # exact
    prompt = _pick(rng, [
        "{n} mol of solute is dissolved in {v} L. What is the molarity?",
        "moles {n} mol volume {v} L molarity",
        "Find the concentration of {n} moles in {v} litres.",
        "What is the molar concentration of {n} mol in {v} L of solution?",
    ], n=moles, v=volume)
    response = (f"molarity = moles / volume, {moles} / {volume} = {concentration}, "
                f"the concentration is {concentration} molar, total {concentration}")
    return OmniProblem("molarity", "chemistry", prompt, response, float(concentration), "M",
                       f"moles {moles} mol volume {volume} L molarity",
                       {"n": moles, "V": volume})


def _combination(rng: random.Random) -> OmniProblem:
    # k is fixed at 2 so the working can actually be shown: C(n,2) is
    # n x (n-1) / 2. The previous version stated `7 choose 2 = 21` with no
    # working at all, leaving the model to memorise the whole table, and it
    # scored 0.00 on 976 rows.
    n = rng.randint(4, 60)
    k = 2
    answer = math.comb(n, k)
    prompt = _pick(rng, [
        "In how many ways can {k} items be chosen from {n}?",
        "n choose k n = {n} k = {k}",
        "Find the number of combinations of {n} things taken {k} at a time.",
        "How many combinations are there of {n} choose {k}?",
    ], n=n, k=k)
    product = n * (n - 1)
    response = (f"combinations = n x (n - 1) / 2, {n} - 1 = {n - 1}, "
                f"{decompose_product(n, n - 1)}, "
                f"half of {product} = {answer}, "
                f"there are {answer} ways, total {answer}")
    return OmniProblem("combination", "mathematics", prompt, response, float(answer), "",
                       f"n choose k n = {n} k = {k}", {"n": n, "k": k})


def _arithmetic_series(rng: random.Random) -> OmniProblem:
    # Bounded so the worked response fits the 128-token sequence length.
    # Turn-aligned packing DROPS a turn that does not fit, which would
    # silently bias this task toward its shortest problems.
    # Every intermediate stays two-digit, and the term count is even so the
    # halving happens on a small number before the only multiplication. The
    # previous version emitted `5 x 228 = 1140` and `1140 / 2` in single
    # steps and scored 0.00. Two-digit single-step additions are fine:
    # `word_problem` does them and scored 0.87.
    first = rng.randint(1, 20)
    difference = rng.randint(2, 6)
    terms = rng.randrange(4, 12, 2)
    last = first + (terms - 1) * difference
    answer = terms * (first + last) // 2
    prompt = _pick(rng, [
        "An arithmetic series starts at {a} with common difference {d}. "
        "What is the sum of the first {n} terms?",
        "sum of arithmetic series first term {a} common difference {d} n {n}",
        "Find the sum of {n} terms of an arithmetic progression "
        "with first term {a} and difference {d}.",
    ], a=first, d=difference, n=terms)
    span = (terms - 1) * difference
    half_terms = terms // 2
    ends = first + last
    response = (f"last term = first + (n - 1) x difference, "
                f"{terms} - 1 = {terms - 1}, "
                # Both operands are single-digit here, so this needs no split.
                f"{terms - 1} x {difference} = {span}, "
                f"{first} + {span} = {last}, "
                f"sum = half of n x (first + last), half of {terms} = {half_terms}, "
                f"{first} + {last} = {ends}, "
                f"{decompose_product(ends, half_terms)}, total {answer}")
    return OmniProblem("arithmetic_series", "mathematics", prompt, response, float(answer), "",
                       f"sum of arithmetic series first term {first} "
                       f"common difference {difference} n {terms}",
                       {"a": first, "d": difference, "n": terms})


#: Every generator, by task name.
TASKS: Dict[str, Callable[[random.Random], OmniProblem]] = {
    "force": _force,
    "acceleration": _acceleration,
    "momentum": _momentum,
    "kinetic_energy": _kinetic_energy,
    "work": _work,
    "power": _power,
    "voltage": _voltage,
    "electrical_power": _electrical_power,
    "wave_speed": _wave_speed,
    "molarity": _molarity,
    "combination": _combination,
    "arithmetic_series": _arithmetic_series,
}


# -- verification -----------------------------------------------------------


def verify(problem: OmniProblem) -> bool:
    """Check a generated problem against the solver.

    This is the point of the module. The generator computes an answer; the
    solver computes it independently, exactly, from a parsed query. If they
    disagree the row is wrong and must not be trained on.
    """

    result = solve_problem(problem.canonical)
    if not result.solved or result.answer_value is None:
        return False
    expected = problem.answer
    if expected == 0:
        return abs(result.answer_value) <= TOLERANCE
    return abs(result.answer_value - expected) / abs(expected) <= TOLERANCE


def extract_answer(text: str) -> Optional[float]:
    """The last number in a reply, matching the benchmark's rule."""

    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(matches[-1]) if matches else None


def build(per_task: int, seed: int, tasks: Optional[List[str]] = None,
          repeat: bool = True):
    """Generate, verify, and return (rows, report)."""

    rng = random.Random(seed)
    chosen = tasks or list(TASKS)
    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    short: Dict[str, Dict[str, object]] = {}
    distinct: Dict[str, int] = {}

    for name in chosen:
        generator = TASKS[name]
        made = 0
        seen = set()
        if repeat:
            # Repetition is how this model learns an algorithm.
            #
            # v74's multiplication task -- the one that scored 0.93 -- holds
            # only **712 distinct operand pairs repeated 56x each** over
            # 40,000 rows. The omni corpus was built the opposite way, 24,000
            # combinations each appearing about 1.7 times, and that task
            # scored 0.03.
            #
            # Uniqueness is the right goal for a *knowledge* corpus, where a
            # repeated fact is memorised. It is the wrong goal for a
            # *procedure*, where the benchmark draws unseen operands from the
            # same space and the model has to learn the method. Distinct
            # counts are still reported, so the repetition factor is visible
            # rather than hidden.
            for _ in range(per_task):
                problem = generator(rng)
                if not verify(problem) or extract_answer(problem.response) != problem.answer:
                    dropped[name] = dropped.get(name, 0) + 1
                    continue
                seen.add(problem.prompt)
                rows.append(problem.to_row())
                made += 1
            counts[name] = made
            distinct[name] = len(seen)
            continue

        # Stop when the generator stops producing prompts it has not already
        # produced. A task's parameter space is finite -- `combination` over
        # n<=40, k<=8 has only about a thousand distinct questions -- and
        # asking for more than it holds would either spin or, worse, ship the
        # same question hundreds of times. Duplicated rows are exactly what a
        # recitation-proof benchmark exists to punish, so the shortfall is
        # reported instead of padded.
        misses = 0
        while made < per_task and misses < EXHAUSTION_MISSES:
            problem = generator(rng)
            key = problem.prompt
            if key in seen:
                misses += 1
                continue
            if not verify(problem):
                dropped[name] = dropped.get(name, 0) + 1
                misses += 1
                continue
            # The response must also parse to the right answer, or the
            # benchmark will score a correct model as wrong.
            if extract_answer(problem.response) != problem.answer:
                dropped[name] = dropped.get(name, 0) + 1
                misses += 1
                continue
            seen.add(key)
            rows.append(problem.to_row())
            made += 1
            misses = 0
        counts[name] = made
        if made < per_task:
            short[name] = {"asked": per_task, "produced": made,
                           "reason": "parameter space exhausted"}

    report = {
        "schema": "supermix-v79-omni-corpus-v1",
        "seed": seed,
        "rows": len(rows),
        "per_task": counts,
        "dropped_failing_verification": dropped,
        "tasks": len(chosen),
        "short_of_requested": short,
        "distinct_prompts": distinct,
        "repetition": {k: round(counts[k] / v, 1)
                       for k, v in distinct.items() if v},
        "verified_by": "nexus_solver.solve_problem",
        "tolerance": TOLERANCE,
    }
    return rows, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per_task", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--output", default="datasets/v79/v79_omni.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--unique", action="store_true",
                        help=("emit only distinct prompts. Off by default: v74's "
                              "multiplication task repeated 712 pairs 56x each and "
                              "scored 0.93, while the diverse omni build scored 0.03"))
    parser.add_argument("--task", action="append", default=[],
                        help="restrict to these tasks; repeatable")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows, report = build(args.per_task, args.seed, args.task or None,
                         repeat=not args.unique)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report["output"] = str(output)
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {len(rows):,} rows to {output}")
    for name, count in sorted(report["per_task"].items()):
        note = ""
        if report["dropped_failing_verification"].get(name):
            note = f"  ({report['dropped_failing_verification'][name]} dropped)"
        print(f"  {name:<20} {count:,}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
