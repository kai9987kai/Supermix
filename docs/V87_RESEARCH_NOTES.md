# v87 research notes: prompt meaning, verifiable steps, and useful compute

Research checked on 2026-09-05. Audience: the next Supermix training experiment.
Scope: a roughly 15M-parameter local model, compact arithmetic reasoning, prompt
understanding, and recursive compute. This is preparation, not a training result.
Implemented artifacts, validation, and launch boundaries are recorded in
[V87_TRAINING_READINESS.md](V87_TRAINING_READINESS.md).
The search covered primary papers and original repositories available by this
date. It is a focused selection, not a claim to have surveyed every new paper.

## Decision

Prepare a controlled data experiment before an architecture replacement. The
most relevant small-model evidence favors explicit equations and structured,
verifiable tasks. The newer work adds two useful requirements: measure behavior
over groups of prompts with known semantic relationships, and measure whether
intermediate steps actually support the next operation. Recursive-model results
justify a separate compute experiment, but do not establish a benefit for this
autoregressive language model.

`docs/V86_PLAN.md` already proposes binary addition steps, fixed-exposure sampling,
step-checked harvesting, a warm residual gate, and serial timing experiments.
Those remain the control arms. The additions here are semantic group scoring,
first-error accounting, explicitly bounded prefix-continuation diagnostics, and
stronger tests for whether extra recursion earns its cost.

## Evidence and transfer limits

### 1. Equation targets: closest parameter-scale evidence

Kim et al., *Small Language Models are Equation Reasoners*, September 2024,
compares T5 variants from 16M to 220M parameters. Its table gives T5-Tiny values
of 0.07 and 0.10 for natural-language and equation targets. The prose interprets
similar table entries as fractions, despite the table's percent heading. Treat
this as directional evidence with a reporting ambiguity, not a precise expected
gain. The tiny model remains weak on GSM8K, and T5's encoder-decoder structure
differs from Supermix. The paper does not establish broad language understanding.
[Paper, experiment and table](https://arxiv.org/html/2409.12393v1)

**Use:** keep diverse question wording while making the supervised arithmetic
working concise and regular. Do not confuse simplifying the answer trace with
removing language variation from the input.

### 2. Semantic perturbations expose errors that ordinary accuracy hides

Mirzadeh et al., *GSM-Symbolic*, first submitted October 2024 and published at
ICLR 2025, evaluates symbolic substitutions and irrelevant clauses. Its tested
large models vary across numeric instantiations and can incorrectly turn
irrelevant quantities into arithmetic operations. This supports evaluating
known transformations; it does not establish that every model lacks reasoning,
or that the reported effect sizes transfer to a 15M model. Some distractors can
also be pragmatically ambiguous, so local variants must have exact semantics.
[Full paper](https://arxiv.org/html/2410.05229v1),
[official publication record](https://machinelearning.apple.com/research/gsm-symbolic)

**Use:** distinguish invariant transformations from changes that require a new
answer. Operand reordering is invariant for a sum but not for subtraction.

### 3. A 2026 prompt study supports group measurement, with caveats

Alhetelah and Ahmad, *Measuring LLMs' Sensitivity to Paraphrased Opinion Prompts*,
WASSA 2026, evaluates 200 opinion questions with five paraphrases each across
five large models. It controls decoding and scores consistency within prompt
groups. The paraphrases were checked by one annotator, and opinions supply no
objective correctness target. Its comparisons do not isolate model size or
alignment causally. It supports the measurement design, not a training recipe.
[Author paper, methods and limitations](https://aclanthology.org/2026.wassa-1.5.pdf)

**Use:** report both agreement and correctness for Supermix's solver-backed
groups. A constant wrong answer must never count as successful understanding.

### 4. Written working is not automatically evidence of causal use

Shih, Winnicki, and Darve, *Do Models Read What They Write? Causal Registers in
Scratchpad Reasoning*, June 2026, studies controlled state tracking in
Qwen2.5-Coder-7B and Mistral-7B. Internal state edits redirect the next update
after running-state supervision. In the main Qwen tests, edited-branch agreement
is 0.80 and 0.91 across two task variants. Generated-prefix tests are filtered
to previously correct prefixes and are less stable on one variant. This is
7B-scale mechanistic evidence, not evidence about Supermix.
[Full paper, intervention contract and generated-prefix controls](https://arxiv.org/html/2606.29522v1)

**Use:** add a cheap text-prefix continuation diagnostic for arithmetic, but
label it behavioral conditioning. The paper explicitly distinguishes that test
from activation intervention with unchanged visible text; the cheap test cannot
prove that the model's original working caused its answer.

### 5. Tiny recursion is promising, but the training procedure matters

Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*,
October 2025, reports a 7M-parameter TRM using repeated latent updates and deep
supervision on structured puzzles. It studies supervision, EMA, model depth,
and recursion count. This is unusually close in parameter count, but the model
predicts structured grids rather than free-form autoregressive language.
[Full paper and ablations](https://arxiv.org/html/2510.04871v1)

The original repository's smallest stated Sudoku experiment takes about 18 hours
on one 48GB L40S; its ARC examples assume four H100 GPUs and roughly three days.
Small parameter count does not imply a cheap CPU reproduction. The repository
also warns that mixing ARC-AGI-1 and ARC-AGI-2 training can contaminate evaluation.
[Official code and experiment requirements](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

**Use:** test the existing recursive path with measured compute and a supervised
objective before considering a replacement architecture.

### 6. Negative evidence: recursion benefits can be shallow

Roye-Azar et al., *Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity
Conditioning, and Test-Time Compute*, December 2025, analyzes one public
verification checkpoint. Canonical single-pass accuracy is 29.25%; 1000-way
augmentation and voting reaches 40.00%. Accuracy mostly saturates early in its
recursion sweep, and incorrect puzzle identifiers collapse performance. The
authors restrict the conclusion to that checkpoint and do not call task
identifiers label leakage. Their naive 8B comparison does not isolate architecture
from training and representation choices.
[Full analysis and limitations](https://arxiv.org/html/2512.11847v1)

**Use:** record accuracy and latency at each cycle count. A flat curve is a
failed inference-compute hypothesis, even if the model has a recursive module.

### 7. SFT and RL alter solution diversity differently

Matsutani et al., *RL Squeezes, SFT Expands*, September 2025 with an ICLR 2026
publication, compares 1.5B, 7B, and 14B model variants. It finds that supervised
reasoning traces expand correct trajectory clusters while RL often concentrates
the sampled distribution, including reducing some correct trajectories. The
original analysis compares published models with different training histories;
its sentence/trajectory clustering is an imperfect proxy for algorithms. It
does not prove that RL cannot learn new skills or prescribe a 15M-model recipe.
[Full original analysis](https://arxiv.org/html/2509.21128v1),
[ICLR 2026 publication](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e52554a70e0df57a0bea11d1eca0c9b5-Abstract-Conference.html)

**Use:** continue verified supervised teaching for weak skills. Judge any later
harvesting or RL arm with pass@1, fixed-budget pass@k, and process validity.

### 8. A 2026 controlled-program system offers a design pattern

Webb and Ramapuram, *A Small-Scale System for Autoregressive Program Synthesis
Enabling Controlled Experimentation*, February 2026, describes Cadmus, a
280M-parameter model with a 65-token integer instruction vocabulary and
verifiable programs. The inspected PDF supports the controlled-execution design.
It is about 19 times Supermix's parameter count and excludes natural language.
Its abstract's under-$200 training statement cannot be reconciled here with
the stated 300,000 steps, batch 1024, and eight H100s; no cost projection is
adopted. The HTML title date differs from the PDF, so the PDF is preferred.
[Author PDF, model configuration and execution rules](https://arxiv.org/pdf/2602.09112)

**Use:** derive answers and individual steps from an executable specification.
This is support for a method of constructing evidence, not a reason to add a
new DSL or tokenizer to the next production training run.

### 9. Fresh structured-data evidence is useful but confounded

O'Grady and Ramlan, *Structured Synthetic Reasoning Data for Arithmetic
Fine-Tuning of Small Language Models*, arXiv 2607.18266, reports Qwen3-0.6B
GSM8K accuracy improving from 36.5% to 49.1% with 21,250 synthetic rows. The
intervention combines traces, cues, structural variation, distractors, filtering,
and deduplication. The authors explicitly lack a matched non-Socratic baseline.
The 0.6B model is 40 times larger than Supermix. The identifier indicates July
2026 while the inspected submission history/PDF says May 21, 2026; recency is
therefore recorded as 2026 only, with unresolved metadata. This source has no
independent decisive weight in choosing the next experiment.
[Full paper and limitations](https://arxiv.org/html/2607.18266v1)

**Use:** consider structured data an experiment, with one factor changed per arm.
There is insufficient evidence here to credit Socratic wording alone.

## Proposed local experiments

The thresholds below are proposed engineering decisions, not effects promised
by papers. Freeze them with the run manifest before examining model results.

### P0: semantic evaluation before semantic training

Construct groups from a canonical solver specification. Each group should contain
an ordinary question, an independently worded equivalent, a relevant fact reorder,
and a small semantic contrast with a solver-confirmed different answer. Include
bounded irrelevant details in a separate labeled slice. Missing-input cases must
have an explicit expected abstention or clarification, rather than an invented
numeric target. Retain existing task composition while preparing these probes.

Split by canonical problem identity before producing variants. Prevent exact and
semantic overlap across training, development, and held-out evaluation. Reserve
wording templates and numeric combinations independently; changing a random seed
alone does not demonstrate template generalization. Hash the manifest, canonical
problems, and all rendered variants.

Report ordinary exact accuracy, all-equivalent-variants-correct rate, agreement
among valid extracted answers, contrast-pair-both-correct rate, abstention accuracy,
and invalid/truncated output rate. Treat a semantic group as the statistical
unit. Use paired group resampling for treatment/control confidence intervals;
individual paraphrases are correlated observations. Report per-task values too.

**Falsifier:** an apparent consistency gain accompanied by lower correctness or
failure to change answers on semantic contrasts is not improved understanding.

### P1: compact process-valid targets

Use the V86 binary-equation arm for `average`, including every operand and running
result. Validate the operand sequence against the canonical problem, each equation
against exact arithmetic, the division against the original operand count, and
the final answer against the solver. A collection of locally true but unrelated
equations does not establish a valid derivation. Unsupported trace syntax must
produce a separate unverified state, not an automatic pass.

Measure first-addition accuracy, first-invalid-step position, complete process
validity, final accuracy, and actual token length under the training tokenizer.
Preserve V86's format-only control and its task/seed/step matching. Match task
exposure explicitly, record supervised tokens, and publish both equal-exposure
and equal-wall-time comparisons if their conclusions differ.

**Go criterion:** retain V86's `average > 0.33` and individual-addition accuracy
above 0.50, with improvement across 4/5/6 operands and no material shared-task
regression. **Falsifier:** prettier working without improved addition validity,
or improvement caused entirely by extra task exposure, rejects the format claim.

### P2: teach input robustness at fixed exposure

Replace a fixed portion of each selected task's rows with solver-verified wording
variants and semantic contrasts. Keep total rows, canonical problem count,
per-task draws, checkpoint, optimizer, and seed matched to the control. These
are supervised contrast examples with their own correct answers, not a new
contrastive loss or preference-optimization algorithm.

Use held-out wording templates for acceptance. A preliminary useful threshold is
a paired improvement of at least 5 percentage points in all-equivalent-correct
rate, without more than a 2-point decline in ordinary shared-task accuracy.
Report uncertainty; a wide interval crossing no gain remains inconclusive.

**Falsifier:** gains only on seen templates, increased agreement on wrong answers,
or failure on the minimally changed question indicates surface memorization.

### P3: diagnose continuation and recursion economically

For text continuation, supply a valid partial equation trace and score the next
operation. A separately labeled counterfactual version supplies an altered current
state and explicitly asks to continue from it. Compare continuations to the stated
state and next operand. Keep clean, counterfactual, and self-generated-prefix
results separate and disclose any prefix-correctness filtering. This assesses
conditioning and local execution, not mechanistic faithfulness.

For recursion, first verify that the existing gate and gradients activate the
repeated path. Compare the V86 warm-gate arm at the same initialization seed,
corpus, supervision, and measured training budget. Sweep 1/2/3/6 inference cycles
on the same frozen examples and record latency, output changes, step validity,
and accuracy. If intermediate supervision is subsequently added, make it its own
arm; maintain causal attention so future gold answer tokens cannot leak backward.

**Falsifier:** no output or accuracy change across cycles, or a gain eliminated
by the equal-wall-time control, provides no support for increasing default
inference cycles. Architectural expansion follows evidence from this test.

## Evidence boundaries and stopping rule

This pass read original source bodies, experiment descriptions, and limitations;
search snippets were used only for discovery. It checked roughly ten primary
research/repository sources across the three decision areas. The strongest
parameter-scale match is a limited 16M T5 comparison; TRM is close in size but
different in task and prediction format. The 2026 causal and data papers are
useful directions with substantial scale or identification gaps. No inspected
source proves that these changes make Supermix broadly smarter.

Further broad searching stopped when each proposed experiment had supporting
evidence, a concrete control, and a falsifier. Independent reproductions of the
newest 2026 results and a directly matched 15M autoregressive language study are
remaining gaps. The next evidence should come from the frozen local comparison.
