"""Deterministic advisory growth policy, isolated from v93 training.

One policy owns one population (connectome modules or one MoE layer). Utility
is supplied by a future training-calibration probe; routing load is not a
substitute. Proposals do not mutate tensors, authorize promotion or implement
rollback. The caller must apply any accepted proposal transactionally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _calibration(role, fingerprint):
    if role != "growth_calibration":
        raise ValueError("growth decisions require training growth_calibration membership")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("a SHA256 calibration membership fingerprint is required")


@dataclass(frozen=True)
class GrowthConfig:
    capacity: int
    max_alive: int
    min_alive: int = 2
    max_births: int = 1
    max_prunes: int = 1
    maturity_steps: int = 500
    low_utility_patience: int = 2
    prune_below: float = 0.01
    grow_above: float = 0.1
    ema_decay: float = 0.9

    def __post_init__(self):
        for name in ("capacity", "max_alive", "min_alive", "maturity_steps", "low_utility_patience"):
            _integer(getattr(self, name), name, 1)
        for name in ("max_births", "max_prunes"):
            _integer(getattr(self, name), name)
        if not self.min_alive <= self.max_alive <= self.capacity:
            raise ValueError("require min_alive <= max_alive <= capacity")
        for name in ("prune_below", "grow_above", "ema_decay"):
            _finite(getattr(self, name), name)
        if self.prune_below >= self.grow_above or self.ema_decay >= 1:
            raise ValueError("require prune_below < grow_above and ema_decay < 1")


@dataclass(frozen=True)
class UnitObservation:
    slot: int
    alive: bool
    born_step: int
    utility: float
    original: bool = False


class GrowthPolicy:
    """EMA utility, maturity and repeated-low-utility hysteresis.

Original units are protected from pruning. Stable slot IDs break ties. A
newborn receives a full maturity window before pruning or becoming a parent.
Each parent gets at most one proposed child per event. Dead slots only are
used for births; a just-pruned slot is not reused in the same event.
"""

    def __init__(self, config: GrowthConfig, membership_sha256: str):
        _calibration("growth_calibration", membership_sha256)
        self.config = config
        self.membership_sha256 = membership_sha256
        self.last_step = -1
        self.total_steps = None
        self.history = {}
        self.provenance = {}

    def propose(self, observations, *, step: int, total_steps: int,
                split_role: str, membership_sha256: str) -> dict:
        _calibration(split_role, membership_sha256)
        if membership_sha256 != self.membership_sha256:
            raise ValueError("calibration membership changed")
        _integer(step, "step")
        _integer(total_steps, "total_steps", 1)
        if step <= self.last_step or step > total_steps:
            raise ValueError("steps must increase strictly and not exceed total_steps")
        if self.total_steps is not None and total_steps != self.total_steps:
            raise ValueError("the registered training horizon cannot change")
        units = list(observations)
        if any(not isinstance(u, UnitObservation) for u in units):
            raise ValueError("expected UnitObservation records")
        for unit in units:
            _integer(unit.slot, "slot")
            _integer(unit.born_step, "born_step")
            _finite(unit.utility, "utility")
            if type(unit.alive) is not bool or type(unit.original) is not bool:
                raise ValueError("alive and original must be booleans")
            if unit.born_step > step or (unit.original and (unit.born_step != 0 or not unit.alive)):
                raise ValueError("invalid birth provenance")
        if len(units) != self.config.capacity or {u.slot for u in units} != set(range(self.config.capacity)):
            raise ValueError("each capacity slot must be observed exactly once")
        live = [u for u in units if u.alive]
        cfg = self.config
        if not cfg.min_alive <= len(live) <= cfg.max_alive:
            raise ValueError("observed active count is outside policy budget")
        provenance = {}
        for unit in units:
            prior = self.provenance.get(str(unit.slot))
            if prior:
                if prior["original"] != unit.original or (prior["original"] and not unit.alive):
                    raise ValueError("original-unit provenance changed")
                if prior["alive"] and unit.alive and prior["born_step"] != unit.born_step:
                    raise ValueError("live unit birth changed without an observed dead transition")
                if not prior["alive"] and unit.alive and unit.born_step < self.last_step:
                    raise ValueError("new or reused slots cannot have backdated births")
            provenance[str(unit.slot)] = {
                "alive": unit.alive, "original": unit.original,
                "born_step": prior["born_step"] if prior and not unit.alive else unit.born_step,
            }
        history = {}
        for unit in live:
            prior = self.history.get(str(unit.slot))
            if prior and prior["born_step"] != unit.born_step:
                prior = None  # explicit reuse of a capacity slot resets its age/EMA
            if prior and prior["original"] != unit.original:
                raise ValueError("original-unit provenance changed")
            ema = (cfg.ema_decay * prior["ema"] + (1 - cfg.ema_decay) * unit.utility
                   if prior else unit.utility)
            mature = step - unit.born_step >= cfg.maturity_steps
            low = (prior["low_count"] if prior else 0) + 1 if mature and ema < cfg.prune_below else 0
            history[str(unit.slot)] = {"born_step": unit.born_step, "original": unit.original,
                                       "ema": ema, "low_count": low}
        candidates = [u for u in live if not u.original and
                      history[str(u.slot)]["low_count"] >= cfg.low_utility_patience]
        candidates.sort(key=lambda u: (history[str(u.slot)]["ema"], u.slot))
        retired = [u.slot for u in candidates[:min(cfg.max_prunes, len(live) - cfg.min_alive)]]
        parents = [u for u in live if u.slot not in retired and
                   step - u.born_step >= cfg.maturity_steps and
                   history[str(u.slot)]["ema"] >= cfg.grow_above]
        parents.sort(key=lambda u: (-history[str(u.slot)]["ema"], u.slot))
        free = sorted(u.slot for u in units if not u.alive)
        budget = min(cfg.max_births, len(free), cfg.max_alive - len(live) + len(retired))
        enough_time = total_steps - step >= cfg.maturity_steps
        births = [{"parent": u.slot, "child": child} for u, child in
                  zip(parents[:budget], free[:budget])] if enough_time else []
        # Commit only after all input checks and calculations succeed.
        self.history, self.last_step = history, step
        self.total_steps, self.provenance = total_steps, provenance
        return {"schema": "supermix-v94-growth-proposal-v1", "step": step,
                "split_role": split_role, "membership_sha256": membership_sha256,
                "births": births, "prune_slots": retired,
                "active_before": len(live),
                "active_if_all_applied": len(live) - len(retired) + len(births),
                "growth_stopped_for_maturation": not enough_time,
                "advisory_only": True}

    def state_dict(self) -> dict:
        return {"schema": "supermix-v94-growth-policy-v1", "config": asdict(self.config),
                "membership_sha256": self.membership_sha256, "last_step": self.last_step,
                "total_steps": self.total_steps,
                "history": {key: dict(value) for key, value in sorted(self.history.items())},
                "provenance": {key: dict(value) for key, value in sorted(self.provenance.items())}}

    @classmethod
    def from_state_dict(cls, state: dict):
        if not isinstance(state, dict) or set(state) != {
            "schema", "config", "membership_sha256", "last_step", "history", "total_steps", "provenance"
        } or state["schema"] != "supermix-v94-growth-policy-v1":
            raise ValueError("invalid policy state schema")
        if not isinstance(state["config"], dict) or set(state["config"]) != set(GrowthConfig.__dataclass_fields__):
            raise ValueError("invalid policy config schema")
        policy = cls(GrowthConfig(**state["config"]), state["membership_sha256"])
        _integer(state["last_step"], "last_step", -1)
        if not isinstance(state["history"], dict):
            raise ValueError("invalid history")
        if not isinstance(state["provenance"], dict):
            raise ValueError("invalid provenance")
        if state["last_step"] == -1:
            if state["total_steps"] is not None or state["history"] or state["provenance"]:
                raise ValueError("uninitialized policy state must be empty")
            return policy
        _integer(state["total_steps"], "total_steps", 1)
        if state["total_steps"] < state["last_step"]:
            raise ValueError("training horizon precedes last step")
        if set(state["provenance"]) != {str(i) for i in range(policy.config.capacity)}:
            raise ValueError("provenance must cover every capacity slot")
        for slot, entry in state["provenance"].items():
            if not isinstance(entry, dict) or set(entry) != {"born_step", "original", "alive"}:
                raise ValueError("invalid provenance schema")
            _integer(entry["born_step"], "born_step")
            if type(entry["original"]) is not bool or type(entry["alive"]) is not bool:
                raise ValueError("invalid provenance flags")
            if entry["born_step"] > state["last_step"] or (entry["original"] and
                    (entry["born_step"] != 0 or not entry["alive"])):
                raise ValueError("invalid birth provenance")
            policy.provenance[slot] = dict(entry)
        for slot, entry in state["history"].items():
            if not isinstance(slot, str) or not slot.isdecimal() or str(int(slot)) != slot or not 0 <= int(slot) < policy.config.capacity:
                raise ValueError("invalid history slot")
            if not isinstance(entry, dict) or set(entry) != {"born_step", "original", "ema", "low_count"}:
                raise ValueError("invalid history schema")
            _integer(entry["born_step"], "born_step")
            _integer(entry["low_count"], "low_count")
            _finite(entry["ema"], "ema")
            if type(entry["original"]) is not bool or entry["born_step"] > state["last_step"]:
                raise ValueError("invalid history birth provenance")
            if entry["original"] and entry["born_step"] != 0:
                raise ValueError("original units must have birth step zero")
            provenance = policy.provenance[slot]
            if not provenance["alive"] or (entry["born_step"], entry["original"]) != (
                    provenance["born_step"], provenance["original"]):
                raise ValueError("history and provenance differ")
            policy.history[slot] = dict(entry)
        if set(policy.history) != {slot for slot, entry in policy.provenance.items() if entry["alive"]}:
            raise ValueError("history must cover exactly the live slots")
        if state["last_step"] >= 0 and not policy.config.min_alive <= len(policy.history) <= policy.config.max_alive:
            raise ValueError("history active count is outside policy budget")
        policy.last_step = state["last_step"]
        policy.total_steps = state["total_steps"]
        return policy


def witness_acceptance(before: list[dict], after: list[dict], *, split_role: str,
                       membership_sha256: str, max_delta_nats: float = 0.001) -> dict:
    """Reject local mutations that regress any witnessed family beyond budget.

Rows have row_id, family, loss_sum, token_count. This is adaptive training
feedback, not held-out evidence. It does not perform model/optimizer rollback.
"""
    _calibration(split_role, membership_sha256)
    _finite(max_delta_nats, "max_delta_nats")

    def index(rows):
        result = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"row_id", "family", "loss_sum", "token_count"}:
                raise ValueError("invalid witness row schema")
            for field in ("row_id", "family"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"invalid witness {field}")
            _finite(row["loss_sum"], "loss_sum")
            _integer(row["token_count"], "token_count", 1)
            if row["row_id"] in result:
                raise ValueError("duplicate witness row_id")
            result[row["row_id"]] = row
        if not result:
            raise ValueError("empty witness")
        return result

    left, right = index(before), index(after)
    if left.keys() != right.keys():
        raise ValueError("witness row IDs differ")
    families = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        if (a["family"], a["token_count"]) != (b["family"], b["token_count"]):
            raise ValueError("witness pairing differs")
        family = families.setdefault(a["family"], {"deltas": [], "tokens": 0})
        family["deltas"].append(b["loss_sum"] - a["loss_sum"])
        family["tokens"] += a["token_count"]
    deltas = {name: math.fsum(value["deltas"]) / value["tokens"] for name, value in families.items()}
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("nonfinite witness aggregate")
    rejected = sorted(name for name, value in deltas.items() if value > max_delta_nats)
    return {"accepted": not rejected, "regressed_families": rejected,
            "family_delta_nats": deltas, "max_delta_nats": max_delta_nats,
            "membership_sha256": membership_sha256, "held_out_evidence": False}
