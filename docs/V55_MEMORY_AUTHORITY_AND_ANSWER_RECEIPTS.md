# Supermix v55: Memory Authority Firewall and Verified Answer Receipts

## Release contract

| Item | Value |
| --- | --- |
| Product version | `55.0.0` |
| Conversation memory | `supermix-conversation-memory-v3` |
| Memory authority schema | `supermix-memory-authority-v1` |
| Memory authority policy | `supermix-memory-authority-firewall-v1` |
| Memory extraction rules | `supermix-explicit-user-memory-v3` |
| Grounding runtime | `supermix-grounding-runtime-v5` |
| Answer receipt | `supermix-verified-answer-receipt-v1` |

V55 is an additive trust-boundary and release-integrity upgrade. It preserves
the v52 model line, the v53 MiMoMix research stack, v54 exact finite-Bernoulli
reasoning, Formal Deliberation v3, and the Qwen Promotion Evidence v4 gate. It
does not train, promote, activate, or package a new checkpoint or adapter.

## Why this boundary exists

Persistent memory creates a second instruction and knowledge channel. A string
can be harmless when first stored but dangerous when it is later retrieved in a
different task, stripped of its original speaker, or treated as more trustworthy
because it is relevant. V55 therefore separates four decisions:

1. who produced the text;
2. whether the text is eligible for storage and recall;
3. what narrow use the stored item may have; and
4. whether the item is relevant to the current request.

Relevance is evaluated last. It can rank already eligible items, but it cannot
change origin, truth status, lifecycle, allowed use, or authority.

This design is informed by persistent agent-memory and retrieval-poisoning work:

- [AgentPoison, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html)
  demonstrates that small, retrievable poisoning sets can persistently alter
  agent behavior.
- [MINJA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42a97bbd9844d2bf68596730af80bcdf-Abstract-Conference.html)
  studies memory injection through ordinary interaction rather than direct
  database access.
- [PoisonedRAG, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag)
  motivates treating retrieval relevance and source authority as separate
  properties.
- [Task Shield, ACL 2025](https://aclanthology.org/2025.acl-long.1435/) and
  [CaMeL](https://arxiv.org/abs/2503.18813) motivate explicit task alignment and
  capability boundaries instead of relying only on prompt-injection wording
  filters.
- [LongMemEval](https://arxiv.org/abs/2410.10813) motivates measuring useful
  long-term recall as well as attack blocking.

These publications motivate the threat model and evaluation shape. They do not
validate Supermix's implementation, filters, policy classes, or model quality.

## Memory Authority Firewall v1

### Origin-bound admission

New memory is extracted only from the current direct-user turn and only through
the bounded explicit patterns already supported by Studio. Every accepted item
is bound to:

- origin and source-turn ID;
- memory kind and canonical text;
- authority class and exact allowed uses;
- confirmation and truth status;
- lifecycle state; and
- a canonical SHA-256 content digest.

Assistant replies, tool results, consultant output, and unknown roles are never
admitted as direct-user memory. Conversation-state normalization accepts only
exact `user` and `assistant` roles; system, tool, consultant, and unknown role
dictionaries are dropped instead of being coerced to user messages.

The admission layer also rejects prompt-control language, chat-role tokens,
quoted or blockquoted memory requests, fenced or inline code, and long encoded
blobs. Unicode text is normalized with NFKC before these checks. This is defense
in depth, not a claim that lexical filtering alone solves memory poisoning.

### Allowed-use classes

| Memory kind | Authority class | Truth status | Maximum allowed use |
| --- | --- | --- | --- |
| Identity | user personalization | self-reported | response personalization |
| Preference | user personalization | self-reported | response personalization |
| Project | user-attributed context | unverified | attributed answer context |
| Fact | user-attributed claim | unverified | attributed answer context |
| Assistant/tool/consultant/legacy | none | unverified | none |

All memory classes are prohibited from acting as evidence, grounding, route or
compute control, tool authorization, permission, safety override, or solver
authority. Confirmation records the user's continued intent but never converts
an attributed claim into externally verified truth.

Only two high-precision slots may enter the shared model prompt: the user's
name and a standing preference for concise versus detailed answers. Other
preferences, projects, and factual claims remain visible in structured,
attributed diagnostics but are not inserted into the shared planner or
tool-capable worker context. Historical assistant exemplars are suppressed.

### Integrity and legacy behavior

The digest covers the canonical origin, kind, text, source turn, policy class,
allowed uses, confirmation state, and truth status. A mismatch blocks recall.
The digest detects accidental corruption and unsynchronized rewrites; it is not
a signature, MAC, trusted timestamp, secure database, or proof against a local
attacker who can rewrite both data and code.

Rows without the v1 authority binding remain readable for inspection but are
`legacy_unbound` and prompt-ineligible. A current direct-user restatement creates
a new bound row; assistant or tool echoes cannot upgrade the legacy row.

### Lifecycle and review API

Bound rows have `active`, `superseded`, `quarantined`, or `revoked` lifecycle
states. Deterministic subject slots supersede earlier active values. A
superseded value cannot be restored over its active successor; the user must
restate it, preserving an auditable current-turn origin.

The local Studio web service and its Memory Authority panel expose:

- `POST /api/memory` with `session_id` for a review-safe snapshot; and
- `POST /api/memory/review` with `session_id`, exact `memory_id`, and one of
  `confirm`, `quarantine`, `revoke`, or `restore`.

The browser persists one opaque random session handle in local storage so a
reload reconnects to the same durable memory instead of orphaning it. The
existing Clear action removes that session's server-side state. Both routes
require an explicit session ID, accept only loopback same-origin management
requests, and return `Cache-Control: no-store`. The inspection response is
allowlisted to schema/policy versions, counts, reviewable memory records, and
update time; it excludes recent prompt/answer turns and route feedback. Review
requires one exact bound ID. Only quarantined rows can be restored, and restore
fails when a newer active binding owns the same deterministic subject slot.
Revocation is terminal through the review API and does not erase the stored
audit row; an exact fresh direct-user restatement can reissue the binding with a
new source-turn receipt.

Studio chat also requires an explicit session ID. When the service is
deliberately bound beyond loopback, the server forces durable memory retrieval
and writes off for those remote callers, even if a client requests memory. This
keeps unauthenticated remote chat available without exposing the local durable
memory channel.

## Verified Answer Receipt v1

Every grounding finalization builds a receipt from canonical allowlisted
diagnostics. Source and compatibility runtimes use byte-identical receipt code,
and terminal chat, packaged web chat, Qwen web, and Studio propagate or display
the same object.

A receipt can report:

- exact arithmetic, deliberate reasoning, or no applicable deterministic path;
- attempted, solved, verified, selected, not selected, or abstained state;
- recognized problem class and method;
- independent-verification status;
- bounded consensus path count and conflict state;
- high-stakes suppression or strict-evidence precedence; and
- explicit-assumption and model-conditional status, with calibration always
  declared false.

The receipt never stores or emits the prompt, generated answer, deterministic
answer, arithmetic expression, proof steps, source/evidence text, or arbitrary
strings from a caller-supplied result. Unknown problem classes, methods, and
reason strings fail into fixed unrecognized categories.

The receipt is diagnostic only. Its authority object fixes compute, routing,
interaction strategy, tools, permissions, safety, and promotion control to
false. It describes the result selected by the pre-existing grounding boundary;
it does not make that selection or change response behavior.

## Release-integrity closure

The checked Studio manifest no longer relies only on a hand-curated list. The
generator recursively parses local imports reachable from the desktop app, web
app, route-study console, route-shadow console, and compatibility terminal/web
entry points. Every reachable local Python
module must be hashed in the manifest. The only exclusions are explicit,
nonblank, reachable, training-only lazy imports; stale, overlapping, escaping,
or unjustified exclusions fail generation.

The manifest binds the memory and answer-receipt schema/policy constants from
both required compatibility mirrors and declares the relevant non-authority
guards. CI now runs Ruff on release-integrity surfaces and includes packaged
web, concurrency, exact-replay, prompt-understanding, and planner-integration
suites that were previously outside the named runtime workflow.

## Verification

Run the focused contract checks:

```bash
python -m pytest -q test_memory_authority.py test_multimodel_memory_tools.py test_conversation_state.py test_multimodel_runtime.py test_grounding_runtime.py test_qwen_grounding_runtime.py test_multimodel_grounding_hook.py test_supermix_multimodel_web_app.py test_studio_runtime_manifest.py
```

Run release parity and hygiene checks:

```bash
python source/sync_runtime_model_variants.py --check
python source/generate_studio_runtime_manifest.py --check
python -m ruff check source runtime_python test_memory_authority.py test_multimodel_memory_tools.py
python -m pytest -q
git diff --check
```

Source readiness does not establish a Windows binary release. Before publishing
v55, separately build and inspect every expected executable, build the installer
from those exact artifacts, validate install/upgrade/uninstall behavior, and
independently recompute release hashes.
