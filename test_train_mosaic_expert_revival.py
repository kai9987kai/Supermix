import json
import sys
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parent / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import build_mosaic_expert_revival_dataset as mosaic_data  # noqa: E402
import mimomix_text  # noqa: E402
import train_mosaic_expert_revival as revival  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402


def _tiny_model() -> MiMoMixModel:
    config = MiMoMixConfig(
        vocab_size=48,
        hidden_size=16,
        n_layers=3,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=32,
        sliding_window=16,
        native_context=16,
        max_position_embeddings=64,
        use_moe=True,
        n_dense_layers=1,
        n_routed_experts=8,
        n_shared_experts=0,
        moe_top_k=2,
        moe_intermediate_size=20,
        n_mtp_layers=0,
        use_thinking_core=False,
        dropout=0.0,
    )
    return MiMoMixModel(config)


def _synthetic_plan(model: MiMoMixModel):
    dialogue = {}
    maths = {}
    for name, _ in revival.moe_layers(model):
        # Experts 0..3 are least-used; expert 4 is dialogue-affine and expert 5
        # is math-affine.  Donor choice must not use the revived set.
        dialogue[name] = [0.01, 0.01, 0.01, 0.01, 0.70, 0.08, 0.10, 0.08]
        maths[name] = [0.01, 0.01, 0.01, 0.01, 0.08, 0.70, 0.10, 0.08]
    return revival.build_expert_plan(model, dialogue, maths)


def test_four_experts_are_cloned_and_only_allowed_state_can_train(tmp_path: Path) -> None:
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = _tiny_model()
    plan = _synthetic_plan(model)
    assert all(entry["revived"] == [0, 1, 2, 3] for entry in plan["layers"].values())
    assert all(entry["dialogue_donor"] == 4 for entry in plan["layers"].values())
    assert all(entry["math_donor"] == 5 for entry in plan["layers"].values())

    revival.revive_from_donors(model, plan)
    for name, layer in revival.moe_layers(model):
        entry = plan["layers"][name]
        for destination in entry["revived"]:
            donor = entry["assignments"][str(destination)]
            for destination_value, donor_value in zip(
                layer.experts[destination].state_dict().values(),
                layer.experts[donor].state_dict().values(),
            ):
                assert torch.equal(destination_value, donor_value)
            assert torch.equal(layer.gate.weight[destination], layer.gate.weight[donor])

    baseline = revival.snapshot_persistent_state(model)
    optimizer, hooks = revival.configure_isolated_optimizer(model, plan)
    try:
        model.train()
        inputs = torch.randint(6, model.config.vocab_size, (3, 12))
        labels = inputs.clone()
        optimizer.zero_grad(set_to_none=True)
        with revival.temporary_selected_routing_boost(model, plan, 5.0):
            output = model(inputs, labels=labels, return_mtp=False)
        assert output.loss is not None
        output.loss.backward()
        revival.assert_gradient_isolation(model, plan)
        optimizer.step()
        revival.update_selected_router_biases(model, plan)
        revival.assert_frozen_state_unchanged(model, baseline, plan)

        generator = torch.Generator().manual_seed(91)
        provenance = {
            "parent_checkpoint_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "dataset_manifest_sha256": "3" * 64,
            "v71_failure_receipt_sha256": "4" * 64,
            "v71_promotion_manifest_sha256": "7" * 64,
            "v71_candidate_checkpoint_sha256": "8" * 64,
            "preregistration_sha256": revival.preregistration_sha256(),
            "source_sha256": revival.dependency_source_hashes(),
            "runtime": revival.runtime_provenance("cpu"),
            "post_revival_pretraining_state_sha256": "6" * 64,
        }
        checkpoint = tmp_path / "checkpoint_step_00400.pt"
        revival._set_optimizer_lr(optimizer, revival.learning_rate_at_step(400))
        for parameter_state in optimizer.state.values():
            parameter_state["step"].fill_(400)
        revival.save_recoverable_checkpoint(
            checkpoint,
            model=model,
            tokenizer=mimomix_text.WordTokenizer([f"token{index}" for index in range(42)]),
            optimizer=optimizer,
            step=400,
            generator=generator,
            expert_plan=plan,
            provenance=provenance,
        )
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            revival.save_recoverable_checkpoint(
                checkpoint,
                model=model,
                tokenizer=mimomix_text.WordTokenizer([f"token{index}" for index in range(42)]),
                optimizer=optimizer,
                step=400,
                generator=generator,
                expert_plan=plan,
                provenance=provenance,
            )
        saved_state_hash = revival.persistent_state_sha256(model)
        tampered_decay = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tampered_decay["optimiser_state"]["param_groups"][1]["weight_decay"] = 0.5
        tampered_decay_path = tmp_path / "tampered_decay.pt"
        torch.save(tampered_decay, tampered_decay_path)
        with pytest.raises(ValueError, match="weight decay violates gate-row isolation"):
            revival.load_recoverable_checkpoint(
                tampered_decay_path,
                expected_sha256=revival.sha256_file(tampered_decay_path),
                model=model,
                optimizer=optimizer,
                generator=generator,
                expert_plan=plan,
                expected_provenance=provenance,
            )
        tampered_moment = torch.load(checkpoint, map_location="cpu", weights_only=False)
        gate_parameter_id = tampered_moment["optimiser_state"]["param_groups"][1]["params"][0]
        tampered_moment["optimiser_state"]["state"][gate_parameter_id]["exp_avg"][4, 0] = 1.0
        tampered_moment_path = tmp_path / "tampered_moment.pt"
        torch.save(tampered_moment, tampered_moment_path)
        with pytest.raises(ValueError, match="nonzero frozen-row exp_avg"):
            revival.load_recoverable_checkpoint(
                tampered_moment_path,
                expected_sha256=revival.sha256_file(tampered_moment_path),
                model=model,
                optimizer=optimizer,
                generator=generator,
                expert_plan=plan,
                expected_provenance=provenance,
            )
        tampered_step = torch.load(checkpoint, map_location="cpu", weights_only=False)
        gate_parameter_id = tampered_step["optimiser_state"]["param_groups"][1]["params"][0]
        tampered_step["optimiser_state"]["state"][gate_parameter_id]["step"].fill_(399)
        tampered_step_path = tmp_path / "tampered_step.pt"
        torch.save(tampered_step, tampered_step_path)
        with pytest.raises(ValueError, match="step state differs from checkpoint step"):
            revival.load_recoverable_checkpoint(
                tampered_step_path,
                expected_sha256=revival.sha256_file(tampered_step_path),
                model=model,
                optimizer=optimizer,
                generator=generator,
                expert_plan=plan,
                expected_provenance=provenance,
            )
        first_name, first_layer = revival.moe_layers(model)[0]
        with torch.no_grad():
            first_layer.gate.weight[plan["layers"][first_name]["revived"][0]].add_(1.0)
        assert revival.persistent_state_sha256(model) != saved_state_hash
        restored_step, _ = revival.load_recoverable_checkpoint(
            checkpoint,
            expected_sha256=revival.sha256_file(checkpoint),
            model=model,
            optimizer=optimizer,
            generator=generator,
            expert_plan=plan,
            expected_provenance=provenance,
        )
        assert restored_step == 400
        assert revival.persistent_state_sha256(model) == saved_state_hash
        revival.assert_frozen_state_unchanged(model, baseline, plan)
        loaded_model, _, loaded_payload = revival.load_talk_checkpoint(checkpoint)
        assert loaded_payload["fallback_schema"] == revival.CHECKPOINT_SCHEMA
        assert revival.persistent_state_sha256(loaded_model) == saved_state_hash
        optimizer.zero_grad(set_to_none=True)
        revival._set_optimizer_lr(optimizer, revival.learning_rate_at_step(401))
        with revival.temporary_selected_routing_boost(model, plan, 1.0):
            resumed_output = model(inputs, labels=labels, return_mtp=False)
        assert resumed_output.loss is not None
        resumed_output.loss.backward()
        revival.assert_gradient_isolation(model, plan)
        optimizer.step()
        revival.update_selected_router_biases(model, plan)
        revival.assert_frozen_state_unchanged(model, baseline, plan)
        wrong_runtime = dict(provenance)
        wrong_runtime["runtime"] = dict(provenance["runtime"], requested_device="cuda")
        with pytest.raises(ValueError, match="provenance mismatch for runtime"):
            revival.load_recoverable_checkpoint(
                checkpoint,
                expected_sha256=revival.sha256_file(checkpoint),
                model=model,
                optimizer=optimizer,
                generator=generator,
                expert_plan=plan,
                expected_provenance=wrong_runtime,
            )
        missing_binding = dict(provenance)
        del missing_binding["v71_candidate_checkpoint_sha256"]
        with pytest.raises(ValueError, match="omits immutable keys"):
            revival.load_recoverable_checkpoint(
                checkpoint,
                expected_sha256=revival.sha256_file(checkpoint),
                model=model,
                optimizer=optimizer,
                generator=generator,
                expert_plan=plan,
                expected_provenance=missing_binding,
            )

        changed_selected_row = False
        state = model.state_dict()
        for name, layer in revival.moe_layers(model):
            revived = plan["layers"][name]["revived"]
            frozen = [index for index in range(layer.n_routed) if index not in revived]
            assert torch.equal(state[f"{name}.gate.weight"][frozen], baseline[f"{name}.gate.weight"][frozen])
            assert torch.equal(state[f"{name}.expert_bias"][frozen], baseline[f"{name}.expert_bias"][frozen])
            changed_selected_row |= not torch.equal(
                state[f"{name}.gate.weight"][revived], baseline[f"{name}.gate.weight"][revived]
            )
        assert changed_selected_row
    finally:
        for hook in hooks:
            hook.remove()


def test_v70_tokenizer_unknown_token_is_rejected() -> None:
    tokenizer = mimomix_text.WordTokenizer.build(["Known text", "Known reply"], digit_tokens=True)
    known = {
        "schema": mosaic_data.ATOMIC_ROW_SCHEMA,
        "row_id": "known",
        "component": {"user": "Known text", "assistant": "Known reply"},
    }
    assert revival.validate_tokenizer_coverage([known], tokenizer) == {
        "checked_rows": 1,
        "unknown_rows": 0,
    }
    unknown = {
        "schema": mosaic_data.ATOMIC_ROW_SCHEMA,
        "row_id": "unknown",
        "component": {"user": "Alien text", "assistant": "Known reply"},
    }
    with pytest.raises(ValueError, match="without <unk>"):
        revival.validate_tokenizer_coverage([known, unknown], tokenizer)


def test_preregistered_schedule_has_exact_mix_warmup_cosine_and_boost() -> None:
    assert revival.PREREGISTRATION["calibration"]["split"] == "train"
    assert set(revival.dependency_source_hashes()) == {
        "train_mosaic_expert_revival.py",
        "build_mosaic_expert_revival_dataset.py",
        "mimomix_core.py",
        "mimomix_text.py",
        "train_mimomix_talk.py",
    }
    assert [revival.batch_kind_at_step(step) for step in range(1, 11)] == [
        "mosaic",
        "mosaic",
        "mosaic",
        "math",
        "dialogue",
    ] * 2
    assert revival.learning_rate_at_step(1) == pytest.approx(4e-4 / 192)
    assert revival.learning_rate_at_step(192) == pytest.approx(4e-4)
    assert revival.learning_rate_at_step(2400) == pytest.approx(4e-5)
    assert revival.learning_rate_at_step(800) < revival.learning_rate_at_step(192)
    assert revival.routing_boost_at_step(1) == pytest.approx(1.5)
    assert 0.0 < revival.routing_boost_at_step(599) < 1.5
    assert revival.routing_boost_at_step(600) == 0.0
    assert revival.routing_boost_at_step(2400) == 0.0


def test_calibration_subset_is_fixed_train_only_and_never_dev() -> None:
    rows = [
        {
            "schema": mosaic_data.ATOMIC_ROW_SCHEMA,
            "row_id": identifier,
            "split": "train",
            "domain": "math",
        }
        for identifier in ("c", "a", "b")
    ]
    subset = revival.fixed_train_calibration_subset(rows, expected_domain="math", limit=2)
    assert [row["row_id"] for row in subset] == ["a", "b"]
    dev_row = dict(rows[0], split="dev")
    with pytest.raises(ValueError, match="only train atomic rows"):
        revival.fixed_train_calibration_subset([dev_row], expected_domain="math")


def _baseline_metrics():
    return {
        "split": "dev",
        "evaluation_manifest_sha256": "a" * 64,
        "original_math_families": {"addition": 0.8, "algebra": 0.7},
        "legacy_chat_items": {"greeting": 1.0, "clarification": 0.8},
        "legacy_chat_score": 0.9,
        "composition_score": 0.3,
        "math_chain_score": 0.2,
    }


def _candidate(step: int, *, checkpoint_hash: str = "b" * 64):
    return {
        "split": "dev",
        "step": step,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_manifest_sha256": "a" * 64,
        "original_math_families": {"addition": 0.8, "algebra": 0.7},
        "legacy_chat_items": {"greeting": 1.0, "clarification": 0.8},
        "legacy_chat_score": 0.9,
        "composition_score": 0.41,
        "math_chain_score": 0.31,
    }


def test_selection_is_earliest_all_gate_pass_and_otherwise_fails_closed() -> None:
    regressed = _candidate(400)
    regressed["original_math_families"]["algebra"] = 0.69
    passing = _candidate(800, checkpoint_hash="c" * 64)
    later = _candidate(1200, checkpoint_hash="d" * 64)
    receipt = revival.select_fail_closed_checkpoint(_baseline_metrics(), [later, passing, regressed])
    assert receipt["decision"] == "audit_gate_passed_no_selection_authority"
    assert receipt["mode"] == "audit_only_unbound_metrics"
    assert receipt["audit_eligible_step"] == 800
    assert receipt["audit_eligible_checkpoint_sha256"] == "c" * 64
    assert receipt["selected_step"] is None
    assert receipt["selected_checkpoint_sha256"] is None
    assert not receipt["selection_authorized"]
    assert not receipt["holdout_used_for_selection"]
    assert not receipt["activation_pointer_written"]

    weak = _candidate(400)
    weak["composition_score"] = 0.39
    dropped_chat_item = _candidate(800)
    del dropped_chat_item["legacy_chat_items"]["clarification"]
    malformed = {"step": "not-an-integer"}
    out_of_range = _candidate(2800)
    no_selection = revival.select_fail_closed_checkpoint(
        _baseline_metrics(), [weak, dropped_chat_item, malformed, out_of_range]
    )
    assert no_selection["decision"] == "no_candidate_passed"
    assert no_selection["selected_step"] is None
    assert no_selection["selected_checkpoint_sha256"] is None


def test_fallback_authorization_requires_hash_bound_frozen_v71_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The authoritative validator has its own exhaustive test suite.  This unit
    # fixture replaces only its manifest-body check so this test can isolate all
    # receipt/path/hash bindings without reproducing 160 frozen records.
    monkeypatch.setattr(revival.v72_model_promotion, "_validate_manifest", lambda manifest: ([], []))
    parent_hash = "1" * 64
    candidate = tmp_path / "v71.pt"
    candidate.write_bytes(b"candidate checkpoint")
    candidate_hash = revival.sha256_file(candidate)
    baseline_corpus = tmp_path / "v70.jsonl"
    candidate_corpus = tmp_path / "v71.jsonl"
    baseline_corpus.write_text('{"user":"baseline corpus prompt"}\n', encoding="utf-8")
    candidate_corpus.write_text('{"user":"candidate corpus prompt"}\n', encoding="utf-8")
    baseline_corpus_hash = revival.sha256_file(baseline_corpus)
    candidate_corpus_hash = revival.sha256_file(candidate_corpus)
    evaluator = SOURCE / "v72_model_promotion.py"
    manifest = {
        "schema": revival.V71_PROMOTION_MANIFEST_SCHEMA,
        "baseline": {
            "checkpoint_sha256": parent_hash,
            "corpus": str(baseline_corpus.resolve()),
            "corpus_sha256": baseline_corpus_hash,
        },
        "candidate": {
            "checkpoint_expected": str(candidate.resolve()),
            "corpus": str(candidate_corpus.resolve()),
            "corpus_sha256": candidate_corpus_hash,
        },
        "prompt_set": {"sha256": "2" * 64},
        "chat_set": {"sha256": "3" * 64},
    }
    manifest_path = tmp_path / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_hash = revival.sha256_file(manifest_path)
    receipt = {
        "schema": revival.V71_FAILURE_SCHEMA,
        "policy_id": revival.V71_PROMOTION_POLICY_ID,
        "mode": "review_only_no_write_pointer",
        "passed": False,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_hash},
        "artifact_binding": {
            "evaluator_path": str(evaluator.resolve()),
            "evaluator_sha256": revival.sha256_file(evaluator),
            "baseline_checkpoint_sha256": parent_hash,
            "candidate_checkpoint": str(candidate.resolve()),
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_corpus": str(baseline_corpus.resolve()),
            "baseline_corpus_sha256": baseline_corpus_hash,
            "candidate_corpus": str(candidate_corpus.resolve()),
            "candidate_corpus_sha256": candidate_corpus_hash,
            "prompt_set_sha256": "2" * 64,
            "chat_set_sha256": "3" * 64,
            "changed_during_evaluation": [],
        },
        "decision": {"passed": False, "blockers": ["overall_accuracy_gain_below_threshold"]},
        "pointer": {"write_requested": False, "write_supported": False, "pointer_written": False},
    }
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    expected_hash = revival.sha256_file(path)
    assert revival.verify_v71_failure_receipt(
        path,
        expected_hash,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_hash,
        candidate_checkpoint_path=candidate,
        expected_candidate_checkpoint_sha256=candidate_hash,
        expected_parent_checkpoint_sha256=parent_hash,
    ) == receipt

    receipt["passed"] = True
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not record passed=false"):
        revival.verify_v71_failure_receipt(
            path,
            revival.sha256_file(path),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_hash,
            candidate_checkpoint_path=candidate,
            expected_candidate_checkpoint_sha256=candidate_hash,
            expected_parent_checkpoint_sha256=parent_hash,
        )


def test_dataset_scans_must_exactly_bind_frozen_v70_v71_corpora() -> None:
    promotion = {
        "baseline": {"corpus_sha256": "1" * 64},
        "candidate": {"corpus_sha256": "2" * 64},
    }
    dataset = {
        "external_corpus_scans": [
            {"sha256": "1" * 64},
            {"sha256": "2" * 64},
        ]
    }
    revival.verify_dataset_external_corpus_binding(dataset, promotion)
    dataset["external_corpus_scans"][1]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="exactly the frozen v70/v71 corpora"):
        revival.verify_dataset_external_corpus_binding(dataset, promotion)


def test_recovery_inventory_excludes_stale_output_files_and_rechecks_resume_hash(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "checkpoint_step_00400.pt"
    written = tmp_path / "checkpoint_step_00800.pt"
    stale = tmp_path / "checkpoint_step_01200.pt"
    resume.write_bytes(b"bound resume")
    written.write_bytes(b"written now")
    stale.write_bytes(b"stale other lineage")
    resume_hash = revival.sha256_file(resume)
    records = revival.recoverable_checkpoint_inventory(
        resume_checkpoint=resume,
        expected_resume_checkpoint_sha256=resume_hash,
        resume_step=400,
        written_checkpoints=[(800, written)],
    )
    assert [(record["role"], record["step"]) for record in records] == [
        ("hash_verified_resume_source", 400),
        ("written_this_invocation", 800),
    ]
    assert all(stale.name not in record["path"] for record in records)

    resume.write_bytes(b"changed after load")
    with pytest.raises(ValueError, match="changed while fallback training was running"):
        revival.recoverable_checkpoint_inventory(
            resume_checkpoint=resume,
            expected_resume_checkpoint_sha256=resume_hash,
            resume_step=400,
            written_checkpoints=[],
        )


@pytest.mark.parametrize(
    ("resume_checkpoint", "resume_hash"),
    [
        (None, "a" * 64),
        (Path("checkpoint_step_00400.pt"), None),
    ],
)
def test_resume_path_and_hash_pairing_fails_before_any_artifact_access(
    tmp_path: Path,
    resume_checkpoint: Path | None,
    resume_hash: str | None,
) -> None:
    if resume_checkpoint is not None:
        resume_checkpoint = tmp_path / resume_checkpoint
    missing = tmp_path / "must_not_be_opened"
    with pytest.raises(ValueError, match="path and expected SHA-256 must be supplied together"):
        revival.train_fallback(
            parent_checkpoint=missing,
            expected_parent_sha256="1" * 64,
            expected_tokenizer_sha256="2" * 64,
            dataset_dir=missing,
            expected_dataset_manifest_sha256="3" * 64,
            v71_failure_receipt=missing,
            expected_v71_failure_receipt_sha256="4" * 64,
            v71_promotion_manifest=missing,
            expected_v71_promotion_manifest_sha256="5" * 64,
            v71_candidate_checkpoint=missing,
            expected_v71_candidate_checkpoint_sha256="6" * 64,
            output_dir=missing,
            resume_checkpoint=resume_checkpoint,
            expected_resume_checkpoint_sha256=resume_hash,
        )
