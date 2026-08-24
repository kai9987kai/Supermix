# v79 — Teaching the model physics, with every example checked

## The thing that changed

Before v74 I was asked to "deeply expand the model's knowledge" and said it was
not deliverable. That was true of the corpus available then: v74's ten task
types are all synthetic arithmetic, and the dialogue portion is 19.8% one
repeated fragment. A model trained on that knows arithmetic and nothing about
the world, which is exactly what v74 turned out to be — 0.894 on arithmetic,
and conversational replies that are 100% verbatim recall.

`nexus_solver.py`, committed with the NexusMind suite, changes what is
possible. It is a deterministic solver over `Fraction`/`Decimal` spanning
mechanics, energy, fluids, thermodynamics, electromagnetism, waves, chemistry,
algebra, geometry and combinatorics. It is a **verified knowledge source**, and
it can be asked an unlimited number of questions.

So the omni series' advance is not a bigger arithmetic corpus. It is a corpus
where **every row is checked by an exact solver before it is trained on**.

## What the omni series actually had

Worth stating plainly, because "advance the omni training" presumes training
exists: **it did not.** The Omniscience suite is entirely symbolic — an exact
solver, SCAMPER/TRIZ ideation rules, persona templates. `nexus_engine.py`
imports torch, but only for `torch.no_grad()`, and it feeds *character
ordinals* as token ids:

```python
input_ids = torch.tensor([[1] + [min(511, ord(c)) for c in query_clean[:64]]])
```

That is a stub, not a trained model. The omni series had no training to
advance, so this builds it.

## The corpus

`source/build_omni_corpus.py` generates a problem, works it out step by step,
and hands a canonical form of the same question to `nexus_solver`. If the two
answers disagree the row is **dropped, not shipped**.

```
wrote 415,375 rows to datasets/v79/v79_omni.jsonl
  force 40,000   acceleration 40,000   momentum 40,000   kinetic_energy 40,000
  work 40,000    power 40,000          voltage 40,000    electrical_power 40,000
  wave_speed 40,000   arithmetic_series 40,000
  molarity 14,399     combination 976
```

Combined with v74's corpus: **911,483 rows, 23 tasks, 5 domains** — maths
400k, physics 360k, dialogue 96k, mathematics 41k, chemistry 14k.

### Two lessons from earlier failures, applied at construction time

**Phrasing varies.** v74 scored 0.894 on its benchmark and got **0 of 5**
naturally-typed questions right, because each task had exactly one template and
the model learned the template. Here each task carries four to five phrasings,
and the phrasing shown to the model is *decoupled* from the canonical query
given to the solver — so variety is unlimited and never constrained by the
solver's parser. v76 fixed that brittleness at inference time with a
normaliser; this fixes it in the data.

**No response ends in a unit.** Answers are extracted as the last number in a
reply, so `total 5 m/s^2` extracts as **2**. Every response ends `total
<number>` with units in the prose before it. A test pins the trap itself
(`extract_answer("the answer is 5 m/s^2") == 2.0`) so the rule cannot later be
mistaken for style.

### The bug the capacity check caught

The first build asked for 40,000 rows per task and stalled. Measuring the
distinct-prompt capacity of each generator showed why:

```
molarity        100 distinct prompts available
combination     104
kinetic_energy  532
```

Asking 40,000 unique rows from a 100-prompt space either spins forever or —
worse — ships the same question four hundred times. **Duplicated rows are
precisely what a recitation-proof benchmark exists to punish**, so a corpus
built that way would have looked fine and trained a memoriser.

Parameter ranges were widened (most tasks now yield 7,000+ distinct per 8,000
draws) and the builder made capacity-aware: it stops when the generator stops
producing new prompts, and **reports the shortfall** rather than padding.
`combination` over n≤40, k≤8 genuinely holds only ~976 distinct questions, and
the report says so.

## The benchmark

The twelve science tasks are registered into `eval_problem_solving.GENERATORS`
rather than given a parallel evaluator. One benchmark, one report format, one
comparison path — two report formats would make v79-versus-v74 a matter of
interpretation instead of a number. The benchmark now spans **21 tasks**, and
the in-training accuracy probe samples all of them, so selection is against the
full task set.

The adapter never shadows an existing arithmetic task, and an import failure
leaves the science tasks absent rather than breaking the benchmark.

## The run

| | v74 | v79 |
|---|---|---|
| train rows | 471,347 | **866,748** |
| parameters | 8,575,977 | **15,269,685** |
| active per token | 2,810,973 | **3,912,997** |
| routed experts | 32 | **48** |
| hidden size | 192 | **256** |
| attention heads | 6 | **8** |
| vocabulary | 8,417 | 8,551 |

The vocabulary barely moved — +134 types for all of physics and chemistry —
because the science prompts reuse ordinary English and digit tokens carry the
numbers. Capacity went up because the *task count* more than doubled, not
because the vocabulary did.

Launched under `train_supervised.py`, so a segfault costs one eval interval
rather than the run.

## What is not claimed

* **Not that v79 beats v74 yet.** That is what the run decides, and the honest
  comparison is per-task on the shared arithmetic set plus the new science
  tasks, at n=500.
* **Not that the model understands physics.** It is trained to reproduce
  worked derivations for twelve formula families. Getting `force = mass x
  acceleration` right across unseen operands is compositional arithmetic
  applied to a physical formula, which is a real and measurable thing, and is
  not the same as knowing what force is.
* **Not that conversation improved.** The dialogue portion is unchanged from
  v74, where the verbatim rate on dialogue probes was 1.0.
