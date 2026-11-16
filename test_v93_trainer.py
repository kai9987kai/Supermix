"""v93 trainer-side contracts (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D5, D8).

What the trainer must do around the model-side growth primitives of
`test_v93_core.py`:

* extend a checkpoint's vocabulary as a prefix and grow the tied embedding
  (`WordTokenizer.extend`, `load_initial_weights(extend_vocab=True)`);
* accept only the three kinds of shape/key departure a warm start may have
  (spare expert rows, appended identity blocks, the connectome graft) and
  refuse everything else;
* run the neurogenesis controller inside the eval branch, log every event,
  carry its counters through the recovery checkpoint, and reproduce a
  continuous run bit for bit after a crash resume (the v63 invariant);
* report the v93 ablations and the neurogenesis summary in the receipt.

Everything runs on a synthetic ~300-row corpus, hidden 32, 2 layers, an
8-module two-hemisphere graph with 4 spare slots, 4 experts + 2 spare, 12
steps. One test drives the real `train_supervised.py` as a subprocess and
prints its wall time.

`source/neurogenesis.py` is owned by another implementer. When it is not
importable (or lacks the class) these tests fall back to a stub with the
exact D6 interface, written into a temporary directory; `USING_STUB` says
which one ran, and the stubbed tests carry it in their receipt line.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import malecns_connectome as mc  # noqa: E402
import mimomix_text as mt  # noqa: E402
import train_mimomix_generalisation as trainer  # noqa: E402
import train_supervised as supervisor  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel, pin_layout_for_growth  # noqa: E402
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

N_MODULES = 8      # live modules in the synthetic two-hemisphere graph
SPARE_MODULES = 4  # dead slots after them
CAPACITY = N_MODULES + SPARE_MODULES
LIVE_EXPERTS = 4
SPARE_EXPERTS = 2
STEPS = 12
EVAL_EVERY = 4
GROW_EVERY = 4

BASE_WORDS = ["alpha", "beta", "gamma", "delta", "value", "count", "weight", "item", "box", "cup"]
NEW_WORDS = ["omega", "sigma", "theta", "kappa", "zeta", "lambda"]


# ---------------------------------------------------------------------------
# the controller: real module when importable, else a stub with the D6 interface
# ---------------------------------------------------------------------------

STUB_SOURCE = r'''
"""Stub of source/neurogenesis.py with the v93 D6 interface.

Written by test_v93_trainer.py because the real controller was not importable
when the tests ran. It performs real slot growth through the core primitives
(so the trainer's checkpoint/resume/receipt paths are exercised) but makes
its decisions with the simplest deterministic rules: no consecutive-event
apoptosis counters beyond bookkeeping, births whenever the load rule fires.
"""
import json
import time
from dataclasses import asdict, dataclass

import torch

from mimomix_core import SparseMoEFeedForward

STUB = True


@dataclass
class GrowthSettings:
    grow_every: int = 0
    grow_modules: int = 0
    grow_edges: int = 0
    grow_taps: int = 0
    grow_experts: bool = False
    prune_threshold: float = 1e-4
    expert_dead_load_fraction: float = 0.1
    expert_split_load_fraction: float = 2.0
    cross_quota: float = 0.5
    edge_logit: float = -7.0
    expert_noise: float = 0.01
    witness_rows: int = 8
    seed: int = 93


def build_settings_from_args(args):
    settings = GrowthSettings()
    for name in ("grow_every", "grow_modules", "grow_edges", "grow_taps",
                 "grow_experts", "prune_threshold", "witness_rows"):
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(settings, name, type(getattr(settings, name))(getattr(args, name)))
    return settings


def _n(value):
    if isinstance(value, dict):
        return int(value.get("count", sum(v for v in value.values() if isinstance(v, int))))
    if isinstance(value, list):
        return len(value)
    return int(value or 0)


class NeurogenesisController:
    STUB = True

    def __init__(self, model, settings, log_path, witness):
        self.model = model
        self.settings = settings
        self.log_path = str(log_path)
        self.witness = witness
        self.events = []
        self.consecutive = {"edges": {}, "experts": {}, "events_seen": 0}

    def _core(self):
        return getattr(self.model, "cns_core", None)

    def _moes(self):
        return [m for m in self.model.modules() if isinstance(m, SparseMoEFeedForward)]

    def begin_dev_pass(self):
        core = self._core()
        if core is not None:
            core.collect_stats = True
            core.reset_stats()
        for moe in self._moes():
            moe.collect_stats = True
            moe.reset_stats()

    def end_dev_pass(self):
        core = self._core()
        if core is not None:
            core.collect_stats = False
        for moe in self._moes():
            moe.collect_stats = False

    @torch.no_grad()
    def _witness_loss(self):
        was_training = self.model.training
        self.model.eval()
        try:
            x, y = self.witness
            out = self.model(x, labels=y, return_mtp=False)
            return float(out.lm_loss)
        finally:
            self.model.train(was_training)

    @staticmethod
    def _zero_moments(optimiser, parameters):
        for parameter in parameters:
            state = optimiser.state.get(parameter)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()

    def maybe_grow(self, step, optimiser):
        s = self.settings
        if s.grow_every <= 0 or step % s.grow_every != 0:
            return None
        started = time.perf_counter()
        before = self._witness_loss()
        core = self._core()
        record = {
            "step": int(step),
            "modules_split": [],
            "edges_opened": {"count": 0, "by_block": {"LL": 0, "RR": 0, "LR": 0, "RL": 0}},
            "edges_pruned": {"grown": 0, "real": 0},
            "taps_opened": {"in": 0, "out": 0},
            "modules_killed": [],
            "experts_born": [],
            "experts_killed": [],
        }
        touched = []
        if core is not None:
            stats = core.growth_statistics()
            pruned = core.prune_edges(s.prune_threshold, grown_only=True)
            record["edges_pruned"]["grown"] = int(pruned)
            score = stats["mean_rate"].clone()
            score[~stats["alive"]] = -1.0
            for _ in range(int(s.grow_modules)):
                if not core.free_slots():
                    break
                parent = int(torch.argmax(score))
                child = core.split_module(parent, step)
                if child is None:
                    break
                record["modules_split"].append([parent, child])
                score[parent] = -1.0
            n = core.n_nodes
            alive = core.alive.bool()
            eye = torch.eye(n, dtype=torch.bool)
            candidate = (core.mask == 0) & alive.unsqueeze(1) & alive.unsqueeze(0) & ~eye
            cross = core.hemisphere.unsqueeze(1) != core.hemisphere.unsqueeze(0)
            cov = stats["rate_cov"]
            quota_cross = (int(s.grow_edges) + 1) // 2
            opened = 0
            for want_cross, quota in ((True, quota_cross), (False, int(s.grow_edges) - quota_cross)):
                allowed = candidate & (cross if want_cross else ~cross)
                ranked = torch.where(allowed, cov, torch.full_like(cov, -float("inf")))
                taken = 0
                for flat in torch.argsort(ranked.flatten(), descending=True).tolist():
                    if taken >= quota:
                        break
                    post, pre = divmod(int(flat), n)
                    if not bool(allowed[post, pre]):
                        break
                    if core.grow_edge(post, pre, step, s.edge_logit):
                        taken += 1
                        opened += 1
                        record["edges_opened"]["by_block"][core.last_event["block"]] += 1
            record["edges_opened"]["count"] = opened
            resid = stats["resid_cov"]
            for kind in ("in", "out"):
                mask_vec = core.in_mask if kind == "in" else core.out_mask
                taken = 0
                for module in torch.argsort(resid, descending=True).tolist():
                    if taken >= int(s.grow_taps):
                        break
                    if not bool(alive[module]) or float(mask_vec[module]) > 0:
                        continue
                    if core.open_tap(module, kind):
                        taken += 1
                        record["taps_opened"][kind] += 1
            touched.extend(core.parameters())
        if s.grow_experts:
            for layer_index, moe in enumerate(self._moes()):
                stats = moe.growth_statistics()
                load, n_alive = stats["mean_load"], int(stats["n_alive"])
                parent = int(torch.argmax(load))
                if n_alive > 0 and float(load[parent]) > s.expert_split_load_fraction / n_alive:
                    child = moe.birth_expert(parent, s.expert_noise)
                    if child is not None:
                        record["experts_born"].append([layer_index, parent, child])
                        touched.extend(moe.experts[child].parameters())
                        touched.append(moe.gate.weight)
        self._zero_moments(optimiser, touched)
        after = self._witness_loss()
        record.update({
            "witness_loss_before": before,
            "witness_loss_after": after,
            "witness_delta": after - before,
            "flagged": abs(after - before) > 1e-3,
            "alive_modules": int(core.alive.sum()) if core is not None else 0,
            "alive_edges": int(core.mask.sum()) if core is not None else 0,
            "alive_experts_per_layer": [m.alive_count() for m in self._moes()],
            "seconds": round(time.perf_counter() - started, 3),
        })
        self.consecutive["events_seen"] += 1
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.events.append(record)
        return record

    def summary(self):
        keys = ("modules_split", "modules_killed", "edges_opened", "edges_pruned",
                "taps_opened", "experts_born", "experts_killed")
        return {
            "stub": True,
            "settings": asdict(self.settings),
            "n_events": len(self.events),
            "totals": {key: sum(_n(e[key]) for e in self.events) for key in keys},
            "flagged_events": [e["step"] for e in self.events if e["flagged"]],
            "events": [
                {"step": e["step"], "witness_delta": round(e["witness_delta"], 6), "flagged": e["flagged"]}
                for e in self.events
            ],
        }

    def state_dict(self):
        return {"consecutive": json.loads(json.dumps(self.consecutive))}

    def load_state_dict(self, payload):
        self.consecutive = dict(payload.get("consecutive", self.consecutive))
'''


def _real_controller_importable() -> bool:
    try:
        module = importlib.import_module("neurogenesis")
    except Exception:  # noqa: BLE001 - any import failure means "use the stub"
        return False
    return all(hasattr(module, name) for name in
               ("NeurogenesisController", "GrowthSettings", "build_settings_from_args"))


USING_STUB = not _real_controller_importable()


@pytest.fixture(scope="session")
def controller_dir(tmp_path_factory) -> Path | None:
    """Directory holding the stub `neurogenesis.py`, or None when the real one exists.

    Put first on `sys.path` (in-process) and on PYTHONPATH (subprocess) so
    the stub wins; both are no-ops when the real module is importable.
    """

    if not USING_STUB:
        return None
    directory = tmp_path_factory.mktemp("neurogenesis_stub")
    (directory / "neurogenesis.py").write_text(STUB_SOURCE, encoding="utf-8")
    sys.path.insert(0, str(directory))
    sys.modules.pop("neurogenesis", None)
    return directory


# ---------------------------------------------------------------------------
# fixtures: corpus, graph, base checkpoint
# ---------------------------------------------------------------------------


def write_corpus(path: Path, rows: int, new_words: bool, seed: int) -> None:
    rng = random.Random(seed)
    words = BASE_WORDS + (NEW_WORDS if new_words else [])
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(rows):
            a, b = rng.randint(10, 99), rng.randint(1, 9)
            subject = rng.choice(words)
            other = rng.choice(words)
            user = f"What is {a} plus {b} for the {subject}?"
            assistant = f"{a} + {b} = {a + b}, so the {subject} and the {other} total {a + b}."
            handle.write(json.dumps({"user": user, "assistant": assistant}) + "\n")


def write_hemisphere_npz(path: Path, seed: int = 0) -> str:
    """An 8-module mirror-scheme graph (0-3 left, 4-7 right) in the layout
    `malecns_connectome.py hemispheres` writes, dense enough for every block."""

    rng = np.random.default_rng(seed)
    roles = np.array(["sensory", "central", "descending", "output"] * 2, dtype=object)
    sign = np.array([1, -1, 1, 1, 1, -1, 1, 1], dtype=np.int8)
    matrix = rng.random((N_MODULES, N_MODULES)) * (rng.random((N_MODULES, N_MODULES)) < 0.6)
    np.fill_diagonal(matrix, 0.0)
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.01)
    r_post, r_pre, _ = mc.degree_preserving_rewire(post, pre, N_MODULES, seed=1)
    side = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    block = (side[post] * 2 + side[pre]).astype(np.int8)  # 0 LL, 1 LR(post L, pre R)... provenance only
    np.savez_compressed(
        path, module_role=roles, module_sign=sign, edge_post=post, edge_pre=pre,
        edge_fraction=fraction.astype(np.float32), rewired_post=r_post, rewired_pre=r_pre,
        module_side=side, homolog=np.array([4, 5, 6, 7, 0, 1, 2, 3]), edge_block=block,
        module_label=np.array([f"m{i}" for i in range(N_MODULES)], dtype=object),
    )
    return str(path)


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("v93_trainer")
    small = root / "small.jsonl"
    full = root / "full.jsonl"
    write_corpus(small, 300, new_words=False, seed=1)
    write_corpus(full, 300, new_words=True, seed=1)
    graph = write_hemisphere_npz(root / "hemispheres.npz")
    return {"root": root, "small": small, "full": full, "graph": graph}


COMMON_FLAGS = [
    "--batch_size", "4", "--eval_batch_size", "8", "--sequence_length", "48",
    "--hidden_size", "32", "--n_layers", "2", "--n_heads", "2", "--n_kv_heads", "1",
    "--intermediate_size", "64", "--moe_intermediate_size", "16",
    "--n_routed_experts", str(LIVE_EXPERTS), "--n_mtp_layers", "1",
    "--thinking_latent_dim", "8", "--accuracy_every", "0", "--select_on", "dev_loss",
    "--sample_tokens", "2", "--dev_fraction", "0.1", "--test_fraction", "0.1",
    "--max_row_fraction_per_sentence", "0.05", "--turn_aligned_packing", "--digit_tokens",
    "--torch_threads", "1", "--checkpoint_every_improvement", "--eval_every", str(EVAL_EVERY),
]


def base_flags(corpus: Path, out_dir: Path, name: str) -> list:
    return [
        "--corpus_jsonl", str(corpus), "--output_dir", str(out_dir), "--run_name", name,
        "--steps", "4", *COMMON_FLAGS,
    ]


def v93_flags(ws: dict, out_dir: Path, name: str, base_checkpoint: Path, extra=()) -> list:
    return [
        "--corpus_jsonl", str(ws["full"]), "--output_dir", str(out_dir), "--run_name", name,
        "--steps", str(STEPS), *COMMON_FLAGS,
        "--init_from", str(base_checkpoint), "--extend_vocab", "--max_new_tokens_vocab", "64",
        "--cns_core", "--cns_graph", ws["graph"], "--cns_nodes", str(CAPACITY),
        "--cns_spare_nodes", str(SPARE_MODULES), "--cns_after_layer", "1",
        "--cns_read_layers", "0", "1", "--cns_write_layers", "1", "--cns_to_thinking",
        "--cns_steps", "2", "--moe_spare_experts", str(SPARE_EXPERTS), "--grow_layers", "1",
        "--grow_every", str(GROW_EVERY), "--grow_modules", "1", "--grow_edges", "4",
        "--grow_taps", "1", "--grow_experts", "--witness_rows", "4", "--ablation_rows", "8",
        "--new_param_lr_mult", "2.0", *extra,
    ]


@pytest.fixture(scope="session")
def base_checkpoint(workspace) -> Path:
    """A tiny plain model trained by this trainer on the small corpus (4 steps)."""

    out_dir = workspace["root"] / "base"
    args = trainer.build_parser().parse_args(base_flags(workspace["small"], out_dir, "base"))
    trainer.run(args)
    return out_dir / "base.pt"


def load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def corpus_texts(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        yield row["user"]
        yield row["assistant"]


# ---------------------------------------------------------------------------
# 1. WordTokenizer.extend
# ---------------------------------------------------------------------------


def test_extend_keeps_the_base_as_a_prefix_and_appends_in_frequency_order(workspace):
    base = mt.WordTokenizer.build(corpus_texts(workspace["small"]), digit_tokens=True)
    extended = mt.WordTokenizer.extend(base, corpus_texts(workspace["full"]), max_new=100)
    assert extended.tokens[: base.vocab_size] == base.tokens
    assert extended.vocab_size > base.vocab_size
    assert extended.digit_tokens and not extended.reverse_digits
    new = extended.tokens[base.vocab_size:]
    assert set(w.strip() for w in new) == set(NEW_WORDS)
    assert all(token in new and " " + token in new for token in NEW_WORDS)  # both spacing forms
    # a superset corpus does not reorder: build() would (frequency), extend() may not
    rebuilt = mt.WordTokenizer.build(corpus_texts(workspace["full"]), digit_tokens=True)
    assert rebuilt.tokens[: base.vocab_size] != base.tokens
    # encoding of any base-covered text is identical
    for text in list(corpus_texts(workspace["small"]))[:40]:
        assert base.encode(text) == extended.encode(text)
    # a corpus the base already covers adds nothing
    again = mt.WordTokenizer.extend(base, corpus_texts(workspace["small"]), max_new=100)
    assert again.tokens == base.tokens
    # the cap is a hard cap on appended ids
    capped = mt.WordTokenizer.extend(base, corpus_texts(workspace["full"]), max_new=3)
    assert capped.vocab_size == base.vocab_size + 3 and capped.tokens[: base.vocab_size] == base.tokens
    # to_dict/from_dict shape unchanged: nothing records the prefix boundary
    assert set(extended.to_dict()) == set(base.to_dict())
    assert mt.WordTokenizer.from_dict(extended.to_dict()).tokens == extended.tokens


# ---------------------------------------------------------------------------
# 2. load_initial_weights: vocabulary, depth, MoE slots, missing-key policy
# ---------------------------------------------------------------------------


def _config_from(payload: dict, **overrides) -> MiMoMixConfig:
    config = dict(payload["config"])
    config.update(overrides)
    return MiMoMixConfig(**config)


def test_extend_vocab_warm_start_grows_the_tied_embedding(workspace, base_checkpoint):
    payload = load_payload(base_checkpoint)
    base = mt.WordTokenizer.from_dict(payload["tokenizer"])
    tokenizer = mt.WordTokenizer.extend(base, corpus_texts(workspace["full"]), max_new=64)
    added = tokenizer.vocab_size - base.vocab_size
    assert added > 0
    torch.manual_seed(0)
    model = MiMoMixModel(_config_from(payload, vocab_size=tokenizer.vocab_size))
    provenance = trainer.load_initial_weights(
        model, tokenizer, str(base_checkpoint), extend_vocab=True, seed=7,
    )
    old = payload["state_dict"]["embed_tokens.weight"]
    embed = model.embed_tokens.weight.detach()
    assert torch.equal(embed[: base.vocab_size], old)                       # rows copied
    assert model.lm_head.weight is model.embed_tokens.weight               # tie kept
    mean, std = old.mean(dim=0), float(old.std())
    new_rows = embed[base.vocab_size:]
    assert new_rows.shape == (added, old.shape[1])
    # new rows are mean(old) + 0.1 std N(0,1): per-element deviation ~0.1 std, never 0
    deviation = (new_rows - mean).abs()
    assert 0.03 * std < float(deviation.mean()) < 0.2 * std
    assert not torch.equal(new_rows[0], new_rows[1])
    # the draw is seeded: the same seed reproduces the rows, another seed does not
    torch.manual_seed(0)
    twin = MiMoMixModel(_config_from(payload, vocab_size=tokenizer.vocab_size))
    trainer.load_initial_weights(twin, tokenizer, str(base_checkpoint), extend_vocab=True, seed=7)
    assert torch.equal(twin.embed_tokens.weight, model.embed_tokens.weight)
    other = MiMoMixModel(_config_from(payload, vocab_size=tokenizer.vocab_size))
    trainer.load_initial_weights(other, tokenizer, str(base_checkpoint), extend_vocab=True, seed=8)
    assert not torch.equal(other.embed_tokens.weight[base.vocab_size:], new_rows)
    grown = provenance["grown_vocab"]
    assert grown["old"] == base.vocab_size and grown["new"] == tokenizer.vocab_size
    assert grown["added"] == added and grown["seed"] == 7 and grown["head_tied_to_embedding"] is True
    assert provenance["missing_keys"] == [] and provenance["allowed_missing"] == {}
    # the caller's payload was not mutated
    assert payload["state_dict"]["embed_tokens.weight"].shape[0] == base.vocab_size


def test_non_prefix_vocabulary_still_raises_and_prefix_without_flag_hints(workspace, base_checkpoint):
    payload = load_payload(base_checkpoint)
    reordered = mt.WordTokenizer.build(corpus_texts(workspace["full"]), digit_tokens=True)
    model = MiMoMixModel(_config_from(payload, vocab_size=reordered.vocab_size))
    with pytest.raises(ValueError, match="different vocabulary"):
        trainer.load_initial_weights(model, reordered, str(base_checkpoint), extend_vocab=True)
    base = mt.WordTokenizer.from_dict(payload["tokenizer"])
    extended = mt.WordTokenizer.extend(base, corpus_texts(workspace["full"]), max_new=64)
    model = MiMoMixModel(_config_from(payload, vocab_size=extended.vocab_size))
    with pytest.raises(ValueError, match="pass --extend_vocab"):
        trainer.load_initial_weights(model, extended, str(base_checkpoint))
    # same prefix but different digit setting: refused, and the message says why
    flipped = mt.WordTokenizer.from_dict({**extended.to_dict(), "digit_tokens": False})
    model = MiMoMixModel(_config_from(payload, vocab_size=flipped.vocab_size))
    with pytest.raises(ValueError, match="digit_tokens/reverse_digits differ"):
        trainer.load_initial_weights(model, flipped, str(base_checkpoint), extend_vocab=True)


def test_grow_layers_gives_identical_logits_and_a_grafted_group(base_checkpoint):
    payload = load_payload(base_checkpoint)
    tokenizer = mt.WordTokenizer.from_dict(payload["tokenizer"])
    config = _config_from(payload)
    base = MiMoMixModel(config).eval()
    base.load_state_dict(payload["state_dict"])
    grown_config = pin_layout_for_growth(config, 1)
    assert grown_config.n_layers == config.n_layers + 1 and grown_config.grow_layers == 1
    torch.manual_seed(3)
    grown = MiMoMixModel(grown_config).eval()
    provenance = trainer.load_initial_weights(
        grown, tokenizer, str(base_checkpoint), grow_layers=1,
    )
    assert provenance["grown_layers"] == {
        "first_new_layer": config.n_layers, "count": 1, "source_had_blocks": False,
        "zeroed": {"layers": 1, "o_proj": 1, "down_proj": 1},
    }
    assert set(provenance["allowed_missing"]) == {"grown_layers"}
    assert all(k.startswith(f"layers.{config.n_layers}.") for k in provenance["missing_keys"])
    x = torch.randint(6, tokenizer.vocab_size, (2, 12))
    with torch.no_grad():
        assert torch.allclose(grown(x, past_length=0).logits, base(x, past_length=0).logits, atol=1e-6)
    prefixes = trainer.new_parameter_prefixes(grown_config)
    assert prefixes == ("cns_core.", f"layers.{config.n_layers}.")
    assert trainer.new_parameter_prefixes(config) == trainer.NEW_PARAMETER_PREFIXES
    from train_mimomix_talk import parameter_groups
    groups = trainer.split_new_parameter_groups(
        grown, parameter_groups(grown, 0.01, "all"), prefixes=prefixes, lr_mult=2.0,
    )
    assert len(groups) == 3 and groups[1]["_lr_mult"] == groups[2]["_lr_mult"] == 2.0
    assert all(n.startswith(f"layers.{config.n_layers}.") for n in groups[1]["_names"] + groups[2]["_names"])
    assert all(p.ndim >= 2 for p in groups[1]["params"]) and all(p.ndim <= 1 for p in groups[2]["params"])
    assert sum(len(g["params"]) for g in groups) == len(list(grown.parameters()))
    # a second load of a checkpoint that already carries the blocks leaves them alone
    trained = {k: v.clone() for k, v in grown.state_dict().items()}
    with torch.no_grad():
        grown.layers[-1].self_attn.o_proj.weight.fill_(0.5)
    carried = {"state_dict": grown.state_dict(), "tokenizer": tokenizer.to_dict(), "extra": {}}
    again = MiMoMixModel(grown_config)
    provenance = trainer.load_initial_weights(again, tokenizer, "carried", payload=carried, grow_layers=1)
    assert provenance["grown_layers"]["source_had_blocks"] and provenance["grown_layers"]["zeroed"] is None
    assert float(again.layers[-1].self_attn.o_proj.weight.abs().mean()) == pytest.approx(0.5)
    del trained


def test_moe_spare_rows_are_copied_into_the_leading_slice(base_checkpoint):
    payload = load_payload(base_checkpoint)
    tokenizer = mt.WordTokenizer.from_dict(payload["tokenizer"])
    base = MiMoMixModel(_config_from(payload)).eval()
    base.load_state_dict(payload["state_dict"])
    spare_config = _config_from(payload, moe_spare_experts=SPARE_EXPERTS)
    torch.manual_seed(11)
    model = MiMoMixModel(spare_config).eval()
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    provenance = trainer.load_initial_weights(model, tokenizer, str(base_checkpoint))
    moe_layers = [m for m in model.modules() if hasattr(m, "expert_alive")]
    assert moe_layers, "the tiny config has no MoE layer"
    for index, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        prefix = f"layers.{index}.mlp."
        gate = payload["state_dict"][prefix + "gate.weight"]
        bias = payload["state_dict"][prefix + "expert_bias"]
        assert torch.equal(layer.mlp.gate.weight[:LIVE_EXPERTS], gate)
        assert torch.equal(layer.mlp.expert_bias[:LIVE_EXPERTS], bias)
        # spare rows keep their construction init, and the slots are dead
        assert torch.equal(layer.mlp.gate.weight[LIVE_EXPERTS:], fresh[prefix + "gate.weight"][LIVE_EXPERTS:])
        assert layer.mlp.expert_alive.tolist() == [1] * LIVE_EXPERTS + [0] * SPARE_EXPERTS
    keys = {entry["key"] for entry in provenance["grown_experts"]}
    # A checkpoint written by a v93 build carries expert_alive at the live
    # count, so it is padded like the router row and bias (SLOT_GROWN_LEAF_SUFFIXES).
    assert keys and all(k.endswith(trainer.SLOT_GROWN_LEAF_SUFFIXES) for k in keys)
    assert set(provenance["allowed_missing"]) == {"spare_experts"}
    assert all(f".mlp.experts.{j}." in k for k in provenance["missing_keys"] for j in (4, 5) if f".experts.{j}." in k)
    x = torch.randint(6, tokenizer.vocab_size, (2, 10))
    with torch.no_grad():  # dead slots are -inf'd: the grown model computes the base's function
        assert torch.allclose(model(x, past_length=0).logits, base(x, past_length=0).logits, atol=1e-6)


def test_missing_key_policy_refuses_an_unexpected_absence(base_checkpoint):
    payload = load_payload(base_checkpoint)
    tokenizer = mt.WordTokenizer.from_dict(payload["tokenizer"])
    model = MiMoMixModel(_config_from(payload))
    dropped = dict(payload["state_dict"])
    victim = next(k for k in dropped if k.startswith("layers.0.self_attn."))
    del dropped[victim]
    with pytest.raises(ValueError, match="lacks 1 tensor"):
        trainer.load_initial_weights(
            model, tokenizer, "dropped", payload={**payload, "state_dict": dropped},
        )
    # an expert key below the spare range is not a spare slot, even with spares on
    spare = MiMoMixModel(_config_from(payload, moe_spare_experts=SPARE_EXPERTS))
    dropped = dict(payload["state_dict"])
    victim = next(k for k in dropped if ".mlp.experts.0." in k)
    del dropped[victim]
    with pytest.raises(ValueError, match="lacks 1 tensor"):
        trainer.load_initial_weights(
            spare, tokenizer, "dropped", payload={**payload, "state_dict": dropped},
        )
    # any other shape difference still raises
    wider = dict(payload["state_dict"])
    wider["norm.weight"] = torch.ones(64)
    with pytest.raises(ValueError, match="shape differs"):
        trainer.load_initial_weights(model, tokenizer, "wider", payload={**payload, "state_dict": wider})


# ---------------------------------------------------------------------------
# 3. the whole run: train_supervised.py, receipt, crash resume
# ---------------------------------------------------------------------------


def _events(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_full_tiny_run_through_train_supervised(workspace, base_checkpoint, controller_dir):
    out_dir = workspace["root"] / "supervised"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    if controller_dir is not None:
        environment["PYTHONPATH"] = str(controller_dir) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable, str(SOURCE_DIR / "train_supervised.py"), "--max_restarts", "1", "--",
        *v93_flags(workspace, out_dir, "v93", base_checkpoint),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - started
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    print(f"\n[v93 tiny run] train_supervised.py wall time {seconds:.1f}s "
          f"({'stub' if USING_STUB else 'real'} controller)")
    assert "=== finished after 1 leg(s) ===" in completed.stdout

    events = _events(out_dir / "neurogenesis.jsonl")
    assert [e["step"] for e in events] == [4, 8, 12]
    for event in events:
        for key in ("witness_loss_before", "witness_loss_after", "witness_delta", "flagged", "modules_split",
                    "edges_opened", "edges_pruned", "taps_opened", "experts_born", "alive_modules",
                    "alive_edges", "alive_experts_per_layer", "seconds"):
            assert key in event, key
    assert events[0]["modules_split"], "the first event split a module into a spare slot"
    assert events[-1]["alive_modules"] > N_MODULES

    receipt = json.loads((out_dir / "generalisation_results.json").read_text(encoding="utf-8"))
    assert receipt["neurogenesis"]["n_events"] == 3 and receipt["neurogenesis"]["events_logged"] == 3
    assert receipt["neurogenesis"]["log"].endswith("neurogenesis.jsonl")
    ablations = receipt["v93_ablations"]["ablations"]
    assert set(ablations) == {"gates_off", "cross_off", "left_only", "right_only", "grown_off"}
    assert receipt["v93_ablations"]["rows"] == 8
    for name, result in ablations.items():
        assert "cost_nats" in result and "dev_loss" in result, name
    assert ablations["gates_off"]["gates"] == 2  # primary write gate + thinking bond (one write site)
    assert receipt["cns_core"]["ablation_cost_nats"] is not None  # v91 block still present
    hyper = receipt["hyperparameters"]
    assert hyper["extend_vocab"] and hyper["grow_every"] == GROW_EVERY and hyper["grow_layers"] == 1
    assert hyper["cns_read_layers"] == [0, 1] and hyper["cns_write_layers"] == [1]
    assert hyper["moe_spare_experts"] == SPARE_EXPERTS and hyper["cns_spare_nodes"] == SPARE_MODULES
    assert hyper["ablation_rows"] == 8 and hyper["witness_rows"] == 4 and hyper["grow_experts"] is True
    assert "v93 neurogenesis" in receipt["architecture"]
    assert receipt["initialised_from"]["grown_vocab"]["added"] > 0
    assert receipt["initialised_from"]["vocabulary"]["source"] == "extended"
    assert receipt["initialised_from"]["grown_layers"]["zeroed"] == {"layers": 1, "o_proj": 1, "down_proj": 1}
    assert set(receipt["initialised_from"]["allowed_missing"]) == {"cns_core", "grown_layers", "spare_experts"}
    assert receipt["parameters"] == receipt["parameters_at_construction"]
    # the receipt describes the SELECTED weights, whose slot masks are those
    # of the event at the selected step (growth runs before selection)
    selected = next(e for e in events if e["step"] == receipt["selection"]["best_step"])
    assert receipt["growth"]["alive_modules"] == selected["alive_modules"]
    assert receipt["growth"]["alive_experts_per_layer"] == selected["alive_experts_per_layer"]
    assert [g["site"] for g in receipt["growth"]["gates"]] == [1, "thinking"]
    for entry in receipt["history"]:
        assert "growth" in entry and "neurogenesis" in entry
        assert entry["neurogenesis"]["modules_split"] >= 0 and "witness_delta" in entry["neurogenesis"]
    assert receipt["config"]["n_layers"] == 3 and receipt["config"]["grow_layers"] == 1
    assert len(receipt["optimiser_groups"]) == 3 and receipt["optimiser_groups"][1]["lr_mult"] == 2.0

    # downstream consumers rebuild from config and load strictly
    model, tokenizer, payload = load_talk_checkpoint(out_dir / "v93.pt")
    assert model.cns_core.n_nodes == CAPACITY and int(model.cns_core.alive.sum()) == selected["alive_modules"]
    assert tokenizer.vocab_size == receipt["config"]["vocab_size"]
    partial = load_payload(out_dir / "v93.partial.pt")
    assert partial["extra"]["neurogenesis_state"] is not None
    assert partial["tokenizer"]["tokens"] == tokenizer.tokens


def test_crash_resume_reproduces_the_continuous_run_including_growth(workspace, base_checkpoint, controller_dir, monkeypatch):
    def arguments(name, extra=()):
        return trainer.build_parser().parse_args(
            v93_flags(workspace, workspace["root"] / name, name, base_checkpoint, extra)
        )

    continuous = trainer.run(arguments("continuous"))
    save = trainer.save_progress_checkpoints

    def crash_after_first_checkpoint(**kwargs):
        save(**kwargs)
        if kwargs["extra"]["steps"] == GROW_EVERY:
            raise InterruptedError("simulated process crash after the first partial checkpoint")

    monkeypatch.setattr(trainer, "save_progress_checkpoints", crash_after_first_checkpoint)
    with pytest.raises(InterruptedError):
        trainer.run(arguments("resumed"))
    monkeypatch.setattr(trainer, "save_progress_checkpoints", save)
    partial = workspace["root"] / "resumed" / "resumed.partial.pt"
    assert _events(workspace["root"] / "resumed" / "neurogenesis.jsonl") and load_payload(partial)["extra"]["steps"] == GROW_EVERY
    resume_flags = supervisor.resume_arguments(
        v93_flags(workspace, workspace["root"] / "resumed", "resumed", base_checkpoint), partial, GROW_EVERY,
    )
    recovered = trainer.run(trainer.build_parser().parse_args(resume_flags))

    left = load_payload(workspace["root"] / "continuous" / "continuous.pt")
    right = load_payload(workspace["root"] / "resumed" / "resumed.pt")
    assert left["state_dict"].keys() == right["state_dict"].keys()
    mismatched = [k for k, v in left["state_dict"].items() if not torch.equal(v, right["state_dict"][k])]
    assert mismatched == []
    assert left["tokenizer"] == right["tokenizer"]
    assert [r["step"] for r in recovered["history"]] == [4, 8, 12]
    assert [r["dev_loss"] for r in recovered["history"]] == [r["dev_loss"] for r in continuous["history"]]
    assert [r["neurogenesis"] for r in recovered["history"]] == [r["neurogenesis"] for r in continuous["history"]]
    assert [r["growth"] for r in recovered["history"]] == [r["growth"] for r in continuous["history"]]
    left_events = _events(workspace["root"] / "continuous" / "neurogenesis.jsonl")
    right_events = _events(workspace["root"] / "resumed" / "neurogenesis.jsonl")
    assert [e["step"] for e in right_events] == [e["step"] for e in left_events] == [4, 8, 12]
    for a, b in zip(left_events, right_events):
        assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
    assert recovered["neurogenesis"]["events_logged"] == 3
    assert recovered["initialised_from"]["vocabulary"]["source"] == "checkpoint"
    assert recovered["initialised_from"]["grown_layers"]["source_had_blocks"] is True
    assert recovered["initialised_from"]["allowed_missing"] == {}
    assert recovered["v93_ablations"]["ablations"]["grown_off"]["cost_nats"] == \
        continuous["v93_ablations"]["ablations"]["grown_off"]["cost_nats"]


def test_supervisor_resume_keeps_every_v93_flag_and_the_extended_tokenizer(workspace, base_checkpoint):
    flags = v93_flags(workspace, workspace["root"] / "x", "x", base_checkpoint)
    resumed = supervisor.resume_arguments(flags, Path("x.partial.pt"), 4)
    kept = [t for t in flags if t not in ("--init_from", str(base_checkpoint))]
    assert resumed[: len(kept)] == kept
    assert resumed[-4:] == ["--init_from", "x.partial.pt", "--start_step", "4"]
    # the resume leg's --extend_vocab reproduces the partial checkpoint's token list:
    # the checkpoint already carries the extended tokenizer, so extend() adds nothing
    partial = workspace["root"] / "resumed" / "resumed.partial.pt"
    if not partial.exists():
        pytest.skip("needs the partial checkpoint of the crash-resume test")
    payload = load_payload(partial)
    carried = mt.WordTokenizer.from_dict(payload["tokenizer"])
    again = mt.WordTokenizer.extend(carried, corpus_texts(workspace["full"]), max_new=64)
    assert again.tokens == carried.tokens
    receipt = json.loads((workspace["root"] / "resumed" / "generalisation_results.json").read_text(encoding="utf-8"))
    assert receipt["tokenizer"]["vocab_size"] == carried.vocab_size


def test_growth_flags_are_inert_by_default_and_validated():
    parser = trainer.build_parser()
    args = parser.parse_args([])
    assert not args.extend_vocab and args.grow_every == 0 and args.grow_layers == 0
    assert args.cns_spare_nodes == 0 and args.cns_read_layers is None and args.moe_spare_experts == 0
    assert not args.cns_to_thinking and not args.cns_temporal and not args.grow_experts
    from train_mimomix_talk import build_config
    config = build_config(args, 97)
    assert config.cns_spare_nodes == 0 and config.cns_read_layers == () and config.cns_write_layers == ()
    assert config.moe_spare_experts == 0 and config.grow_layers == 0
    wrong = parser.parse_args(["--grow_every", "6", "--eval_every", "4", "--steps", "12"])
    with pytest.raises(SystemExit, match="multiple of --eval_every"):
        trainer.run(wrong)
