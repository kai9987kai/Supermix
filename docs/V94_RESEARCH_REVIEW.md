# v94 research review: controlled capacity adaptation

Reviewed 2026-09-20. This is preparation for the version after v93, not a
training result, promotion decision, or amendment of v93's registered design.
Paper findings below are separated from proposed Supermix adaptations.

## Recommendation and implementation boundary

Prioritize **budgeted growth with maturity and utility protection**, explicit
training-only calibration membership, and retention/compute comparisons.
The v94 preparation deliverable is advisory scaffolding: it can describe and
audit candidate growth decisions without applying them to the live model.
Training integration, transactional mutations, and capability improvements
require separate implementation and run evidence. This review claims no test
result and does not establish that a training controller is integrated.

The inspected v93 design allocates capacity slots, duplicates busy experts,
splits active modules, and proposes edges using activation covariance.
`source/mimomix_core.py` has corresponding primitives and diagnostics; its
expert-birth docstring correctly acknowledges that fixed-top-k duplication
changes outputs. The `source/neurogenesis.py` controller named in the design
was absent at this review's initial inspection. Verify the actual run receipt
and source snapshot before treating a design statement as deployed behavior.

## Primary sources and applicability

### 1. Continual backpropagation: preserve useful units, replace selectively

Dohare et al., **Loss of plasticity in deep continual learning**, Nature,
published **2024-08-21**. [Paper](https://www.nature.com/articles/s41586-024-07711-7)
and [author implementation](https://github.com/shibhansh/loss-of-plasticity).
Continual backpropagation selectively reinitializes a small share of less-useful
units and uses a maturity threshold; experiments cover supervised learning and
reinforcement learning. This motivates age, utility, and replacement-budget
bookkeeping for v94, rather than treating low usage alone as sufficient to kill
a unit. Its evidence does not establish that recycling trained language-model
experts preserves old skills. **Priority: first**, as an inspiration for bounded
decision rules, with Supermix retention measured independently.

### 2. Elastic growth: useful new capacity need not mean permanent expansion

Kong and Sutton, **Plasticity of Growing and Elastic Neural Networks in Online
Continual Learning**, **2026-08-02 preprint**, v1.
[Paper](https://arxiv.org/html/2608.01475v1).
Adaptive networks keep existing connections trainable while adding fresh units;
pruning estimated dead units permits compact networks in the studied streams.
The experiments are online permuted MNIST/FashionMNIST, not language modeling
or biological-connectome recurrence. For Supermix, compare fixed active capacity,
growth alone, and budgeted turnover; do not infer plasticity loss from dead-unit
count alone. **Priority: first**, for a bounded-capacity hypothesis and explicit
limits, not evidence that neurogenesis already helps v93.

### 3. Growth stability: forward activity does not prove a newborn is learning

Lillo and Cheney, **On the Stability of Growth in Structural Plasticity**,
submitted **2026-05-14**, **v2 revised 2026-06-10; preprint**.
[Paper](https://arxiv.org/html/2605.15435v2).
New units can be active in forward computation yet receive weak gradients;
insertion time and time available to integrate matter. The study examines
unit-level growth, MLPs, convolutional image classifiers, and continual streams,
not MoE language models or individual synapse growth. For v94, protect newborns
for a minimum age and reserve training time after the final growth event;
later collect cohort gradient/update norms alongside activation and load.
Function preservation and optimizer resets alone do not prove trainability.
**Priority: first** for maturity and timing; gradient diagnostics need integration.

### 4. Expert upcycling: allocate duplicates by measured contribution

Dwivedi et al., **Expert Upcycling: Shifting the Compute-Efficient Frontier of
Mixture-of-Experts**, submitted **2026-04-21**, **v2 revised 2026-05-10; preprint**.
[Paper](https://arxiv.org/html/2604.19835v2).
It expands experts while keeping top-k fixed, studies gradient-based utility
allocation, and explicitly distinguishes a warm initialization from exact
function preservation under discrete routing. The main experiment expands a
roughly 7B-total model to 13B; smaller ablations extend down to 154M total.
This is much larger than Supermix and uses substantial continued pretraining.
For v94, cap repeated duplication of one parent and distinguish routing load
from contribution to task loss. Test duplicate collisions in top-2 routing.
**Priority: first** for allocation caps; gradient utility is a later experiment.

### 5. Orthogonal growth: distinguish added capacity from added active compute

Wang et al., **Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via
Orthogonal Growth of Mixture-of-Experts**, submitted **2025-10-09**, revised
**2026-05-15; accepted at ICML 2026**.
[Paper](https://arxiv.org/html/2510.08008v2).
The work studies interposed layer copying and noisy expert duplication,
including comparisons under fixed additional and total compute budgets.
Its principal width operator increases both expert count and top-k; fixed-k
expansion is weaker in its reported ablation. That differs from source 4's
regime and does not settle which choice works for Supermix. Do not silently
increase top-k or describe extra capacity as free. **Priority: experiment design**;
depth growth and active-compute increases remain separate treatments.

### 6. RigL: task gradients provide a sparse-connectivity comparator

Evci et al., **Rigging the Lottery: Making All Tickets Winners**, **ICML 2020**.
[Paper](https://proceedings.mlr.press/v119/evci20a.html).
RigL changes sparse topology using weight magnitudes and infrequent gradient
calculations at a fixed parameter budget; experiments include image classifiers
and recurrent language models. It does not validate unsigned activation
covariance as an edge-utility estimate in Dale-constrained recurrent networks.
A future v94 comparison can rank missing edges by the predicted loss reduction
for their allowed sign, against covariance and random proposals at equal quotas.
Ordinary gradients of a masked-off edge logit are zero, so this requires an
explicit candidate-edge probe. **Priority: deferred**, after safe event integration.

### 7. Gated attention: a communication hypothesis, not hemisphere evidence

Qiu et al., **Gated Attention for Large Language Models: Non-linearity,
Sparsity, and Attention-Sink-Free**, submitted **2025-05-10; NeurIPS 2025**.
[Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html)
and [author implementation](https://github.com/qiuzh20/gated_attention).
Head-specific, input-dependent sigmoid gates after SDPA improve the studied
dense and MoE models. This concerns attention outputs, not fly hemispheres.
The Supermix hypothesis is selective LR/RL message gating while retaining
within-side recurrence. Nonnegative gates preserve edge signs; initialization
must preserve the existing branch function. Compare constant, learned, shuffled,
and removed communication before attributing benefit to selective bonding.
**Priority: deferred**, since this adds another architectural treatment.

### 8. Transition abruptness: test a curriculum before diagnosing lost plasticity

Liu and Mou, **Do Neural Networks Lose Plasticity in a Gradually Changing World?**,
submitted **2026-02-09**, **v2 revised 2026-06-16; preprint**.
[Paper](https://arxiv.org/html/2602.09234v2).
Their controlled image/sequence tasks show that more gradual task transitions
can substantially reduce plasticity loss. The revised claim is conditional,
not that plasticity loss is universally an artifact. For Supermix, use a
preregistered old/new corpus mixture ramp as a low-cost comparator to growth.
Keep cumulative per-family tokens matched when comparing ramped and abrupt
exposure. **Priority: experiment design**, before interpreting a continuation
setback as insufficient model capacity.

## Proposed integration and evidence requirements

1. Freeze a calibration subset from training membership, with row/group hashes
   disjoint from retention and final evaluation. Running in `eval()` disables
   training behavior; it does not make data used to choose topology held-out.
2. Bound active modules, experts, edges, total accepted edits, and per-parent
   births. Track age and repeated low utility before proposing removal. Keep a
   minimum expert count sufficient for top-k and protect scarce family coverage.
3. Apply each proposed mutation transactionally on future training integration:
   reject non-finite or excessive witness loss changes, restore edited model and
   optimizer state plus RNG on rejection, and record accepted/rejected reasons.
   A passed witness is local stability evidence, not generalization evidence.
4. Log birth shock and subsequent newborn activation, routing, gradient/update
   norms, survival, and contribution. Avoid events too near training completion
   to test integration; measure learning progress as well as final accuracy.
5. Compare fixed-topology continuation, v93-style growth, and bounded mature
   utility growth from the same verified checkpoint and corpus/token schedule.
   Include a control with the treatment's final capacity active from the start
   when feasible. Report both token-matched and measured-compute comparisons.
6. Freeze old-family retention and new-family evaluation before runs. Use paired
   generation receipts and uncertainty intervals, plus gate/commissure/growth
   ablations. More parameters, lower training loss, or nonzero gates alone
   establish neither usefulness nor promotion readiness.

The current connectome dynamics multiply dense masked matrices, and capacity
slots remain allocated. Fewer live edges are not proof of lower memory or CPU
time; measure latency, peak memory, and diagnostic overhead. Cross-hemisphere
gates and gradient-aware synaptogenesis remain unproven follow-up ideas until
isolated comparisons justify adding them to a training candidate.
