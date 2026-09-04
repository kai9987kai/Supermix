# Nexus source-locked temporal evidence ledger

Status: implemented as a shadow-only v2 component on 2026-09-03. It is not
wired into answer routing, conversation memory, tool permissions, model
activation, training, or promotion. No trained weights or active pointers are
changed.

## Why this exists

The grounding runtime can diagnose caller-supplied evidence, but a citation
identifier alone does not prove which source bytes were fetched, which spans
were opened for a turn, or whether a fact is still current. This ledger keeps
those concerns explicit and separate from personal/conversation memory.

## Contract

- Only `origin=server_fetch` snapshots carrying a ledger-instance-bound,
  server-fetch receipt from the trusted adapter can be persisted. A direct
  `record_snapshot` call, caller-provided text, or a receipt from another
  ledger instance is rejected; caller text is classified as
  `untrusted_ephemeral` and is never written. The private adapter hook binds
  all source metadata (including validity windows) to the receipt. This is an
  in-process trust boundary, not cryptographic authentication of the adapter.
- A snapshot records provider identity, canonical HTTP(S) URI, fetch/publication,
  event/mention and validity times, extractor version, source-byte hash, and
  bounded spans with character and UTF-8 byte offsets.
- A turn seals one ordered evidence set, then binds the exact ordered hashes of
  every generated sentence before claims are recorded. The ledger stores the
  sentence hashes, not the generated text a second time. Binding requires an
  immutable, ledger-instance-bound capability from the trusted server-generation
  adapter; request payloads cannot supply it. Neither the evidence set nor output
  manifest can be reopened with different content. Like the fetch capability,
  this is an in-process boundary rather than cryptographic authentication.
- Every claim names one opened `(snapshot, span)` and one relation:
  `quotation`, `compression`, or `inference`. Quotations must match the opened
  span after whitespace normalization. Compression remains auditable but
  unproved. Caller-supplied checker IDs and booleans are rejected. An inference
  can be mechanically marked only by an immutable receipt minted after a
  configured allowlisted checker actually runs. The receipt binds the ledger,
  turn, exact output sentence, source span hash, checker version and source
  digest, result, and one-use run nonce. Only a `passed` and
  `algorithmically_independent` result counts. Even then,
  `authority_granted=false`.
- Evaluation is complete only when every bound output sentence has a claim and
  every claim is mechanically verified. Missing, compressed, unchecked,
  failed, or non-applicable relations return `coverage_defer`; a single
  arbitrary claim can no longer make a turn look complete.
- Explicit conflicts retain both snapshots and make a turn `conflict_defer`.
  Revisions are append-only and require an explicit `supersedes_snapshot_id`;
  ambiguous contradictions are not silently replaced.
- Freshness-sensitive turns reject unknown, future, or expired validity windows.
  Hashes are domain-separated integrity metadata, not signatures, authentication,
  trusted timestamps, or proof that a source is true.

The implementation uses short `BEGIN IMMEDIATE` SQLite transactions, WAL,
`busy_timeout`, foreign keys, an append-only v1-to-v2 schema migration record,
versioned schema-shape checks, and SQLite
append-only update/delete triggers whose definitions are health-checked. Reads
revalidate snapshot, span, opened-span, turn, output-manifest, claim, checker
receipt, conflict, and revision invariants. Health also runs SQLite integrity
and foreign-key checks, and returns `degraded` when any check fails. Canonical JSON follows an
RFC 8785-style deterministic representation for local hashing; it is not a
replacement for a signed JCS envelope. The design is compatible with [W3C
PROV-O](https://www.w3.org/TR/prov-o/) concepts and [RFC
8785](https://www.rfc-editor.org/rfc/rfc8785.html), but does not claim their
stronger interoperability or authenticity properties.

## Research mapping and evaluation boundary

The separation of retrieval, reading, temporal updates, and abstention follows
[LongMemEval](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html).
Sentence-level typed provenance is motivated by [TROVE, ACL
2025](https://aclanthology.org/2025.acl-long.577/) and [GenProve, ACL
2026](https://aclanthology.org/2026.acl-long.228/). GenProve reports a material
gap between surface quotation and inference provenance, which is why inference
now requires an executed checker receipt instead of a self-attested flag.
[From Agent Traces to Trust](https://arxiv.org/abs/2606.04990) argues that final
answer accuracy cannot expose which evidence supported each claim; the output
manifest makes that coverage inspectable. Recent temporal-memory
stress tests such as [STALE](https://arxiv.org/abs/2605.06527) motivate keeping
invalidated memories from silently becoming current again, while [Prompt-Based
Abstention Fails Under Misleading Context](https://arxiv.org/abs/2608.22228)
supports structural ledger gates instead of prompt-only abstention.

The v2 receipt is deliberately described as local integrity metadata, not a
transparency receipt. [RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html)
requires an ordered append-only sequence, non-equivocation, replayability, and
verifiable receipts; [RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html)
adds signed tree heads plus inclusion and consistency proofs. This local SQLite
sidecar has no signature, external witness, global Merkle sequence, inclusion
proof, consistency proof, or fork detection. It therefore must not be marketed
as cryptographically transparent, authenticated, or non-equivocating.

Before any integration decision, measure evidence recall, opened-span citation
precision, unopened-citation violations, stale-fact errors, conflict-defer rate,
latency/tokens, corruption recovery, and concurrent-writer behavior. A passing
shadow metric is not a promotion receipt.

Tests: `test_nexus_evidence_ledger.py` exercises immutable snapshots, untrusted
input rejection, ordered sealing, exact output binding, complete-sentence
coverage, quotation checking, trusted checker execution, forged/cross-ledger
receipt rejection, nonce replay rejection, restart validation, temporal
deferral, conflict/revision retention, direct-SQL claim/content tamper
detection, schema fail-fast behavior, bounded adversarial iterables, and
concurrent SQLite writers.

## Next research-gated frontier

The next high-value component is a separate per-session temporal-memory
transition sidecar, not an automatic hookup of this evidence ledger to active
conversation memory. [TRUSTMEM](https://arxiv.org/abs/2606.25161) motivates
coverage, preservation, and faithfulness checks on each memory update;
[MemoryGraft](https://arxiv.org/abs/2512.16962) demonstrates why retrieved
successful experiences must never acquire policy authority; and STALE motivates
explicit supersession, implicit-conflict handling, premise resistance, and
as-of retrieval. These 2025-2026 systems include recent preprints, so their
reported metrics are research signals rather than Supermix capability evidence.
Any implementation remains shadow-only until it demonstrates zero cross-session
leakage, zero authority escalation, deterministic replay, stale/conflict
deferral, corruption detection, and unchanged live prompts and outputs.
