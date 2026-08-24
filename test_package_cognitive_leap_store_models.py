from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import package_cognitive_leap_store_models as packager  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, name: str, content: bytes, **extra: Any) -> dict[str, Any]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        **extra,
    }


def _json_artifact(
    root: Path,
    name: str,
    payload: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return _artifact(
        root,
        name,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        **extra,
    )


def _receipt_fixture(root: Path, *, gate_outcome: str = "pass") -> Path:
    baseline = _artifact(
        root,
        "run/baseline/baseline.pth",
        b"baseline-state",
        canonical_state_sha256="0" * 64,
        tensor_count=4,
        element_count=12,
    )
    candidate = _artifact(
        root,
        "run/selected/candidate.pth",
        b"candidate-state",
        canonical_state_sha256="1" * 64,
        tensor_count=4,
        element_count=12,
    )
    source_snapshots = {
        "source/benchmark_cognitive_leap_ultra_v51.py": _artifact(
            root,
            "run/source_snapshot/source/benchmark_cognitive_leap_ultra_v51.py",
            b"# frozen generator\n",
        ),
        "source/cognitive_leap_receipt.py": _artifact(
            root,
            "run/source_snapshot/source/cognitive_leap_receipt.py",
            b"# frozen validator\n",
        ),
    }
    member_artifacts: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for name, content in (
        ("tempered_251", b"member-state-251"),
        ("tempered_351", b"member-state-351"),
    ):
        artifact = _artifact(
            root,
            f"run/members/{name}/weights.pth",
            content,
            canonical_state_sha256=("4" if name.endswith("251") else "5") * 64,
            tensor_count=4,
            element_count=12,
        )
        training = _json_artifact(
            root,
            f"run/members/{name}/training_receipt.json",
            {
                "schema": "supermix-cognitive-leap-training-receipt-v2",
                "artifact": artifact,
                "baseline": baseline,
                "config": {"name": name, "train_seed": int(name[-3:])},
            },
        )
        member_artifacts.append(artifact)
        member_rows.append(
            {
                "name": name,
                "artifact": artifact,
                "training_receipt": training,
            }
        )
    protocol_payload = {
        "schema": "supermix-cognitive-leap-bounded-protocol-v2",
        "protocol_sha256": "6" * 64,
        "baseline": baseline,
        "source_snapshot": source_snapshots,
    }
    protocol_file = _json_artifact(root, "run/protocol.json", protocol_payload)
    protocol = {
        "path": protocol_file["path"],
        "file_sha256": protocol_file["sha256"],
        "size_bytes": protocol_file["size_bytes"],
        "sha256": protocol_payload["protocol_sha256"],
        "final_eval_seeds": [101052],
        "samples_per_seed": 2,
        "cohort_specification": {"schema": "test-cohort"},
        "cohort_specification_sha256": "7" * 64,
        "single_use": True,
    }
    lineage = _json_artifact(
        root,
        "run/lineage_manifest.json",
        {
            "schema": "supermix-cognitive-leap-lineage-v2",
            "baseline": baseline,
            "members": member_rows,
            "selected_artifact": candidate,
        },
        schema="supermix-cognitive-leap-lineage-v2",
    )
    lineage_verification = _json_artifact(
        root,
        "run/lineage_verification.json",
        {
            "schema": "supermix-cognitive-leap-lineage-verification-v1",
            "lineage_manifest": lineage,
            "selected_artifact": candidate,
        },
        schema="supermix-cognitive-leap-lineage-verification-v1",
    )
    selection_payload = {
        "schema": "supermix-cognitive-leap-development-selection-v2",
        "protocol": protocol,
        "selected": {"artifact": candidate, "members": member_artifacts},
        "lineage_manifest": lineage,
        "lineage_verification": lineage_verification,
    }
    selection_file = _json_artifact(root, "run/selection.json", selection_payload)
    selection = {
        "path": selection_file["path"],
        "sha256": selection_file["sha256"],
        "size_bytes": selection_file["size_bytes"],
        "name": "soup_blend25",
        "members": ["tempered_251", "tempered_351"],
        "member_weights": [0.5, 0.5],
        "baseline_blend_alpha": 0.25,
        "lineage_manifest": lineage,
        "lineage_verification": lineage_verification,
    }
    predictions = _artifact(
        root,
        "run/final_predictions.jsonl.gz",
        b"deterministic-gzip-placeholder",
        schema="supermix-cognitive-leap-paired-logits-jsonl-v1",
        uncompressed_sha256="2" * 64,
        row_count=40_000,
        dataset_id="dataset-v2-test",
    )
    receipt = {
        "schema": packager.BOUNDED_EVALUATION_SCHEMA,
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": packager.CONTENT_BOUND_STATUS,
        "authority": dict(packager.BOUNDED_EVALUATION_AUTHORITY),
        "gate_outcome": gate_outcome,
        "protocol": protocol,
        "selection": selection,
        "artifacts": {"baseline": baseline, "candidate": candidate},
        "per_example_artifact": predictions,
        "receipt_id": "3" * 64,
    }
    path = root / "run/bounded_evaluation_receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


def _valid_semantics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    path = Path(_args[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "valid": True,
        "gate_outcome": payload["gate_outcome"],
        "receipt_id": payload["receipt_id"],
        "receipt_file_sha256": _sha256(path),
        "candidate_sha256": payload["artifacts"]["candidate"]["sha256"],
        "lineage_sha256": payload["selection"]["lineage_manifest"]["sha256"],
        "lineage_verification_sha256": payload["selection"][
            "lineage_verification"
        ]["sha256"],
        "per_example_sha256": payload["per_example_artifact"]["sha256"],
        "per_example_uncompressed_sha256": payload["per_example_artifact"][
            "uncompressed_sha256"
        ],
    }


def _three_way_receipt_fixture(root: Path) -> Path:
    release = _artifact(
        root,
        "run/release/release_v51.pth",
        b"release-v51-state",
        canonical_state_sha256="a" * 64,
        tensor_count=4,
        element_count=12,
        strict_load=True,
    )
    prior = _artifact(
        root,
        "run/prior/prior_v51_1.pth",
        b"prior-v51.1-state",
        status="unpromoted_prior_candidate",
        canonical_state_sha256="b" * 64,
        tensor_count=4,
        element_count=12,
        strict_load=True,
    )
    candidate = _artifact(
        root,
        "run/selected/candidate_v51_2.pth",
        b"candidate-v51.2-state",
        canonical_state_sha256="c" * 64,
        tensor_count=4,
        element_count=12,
        strict_load=True,
    )
    snapshots = {
        "source/run_cognitive_leap_v51_2.py": _artifact(
            root,
            "run/source_snapshot/source/run_cognitive_leap_v51_2.py",
            b"# frozen runner\n",
        ),
        "source/cognitive_leap_three_way_receipt.py": _artifact(
            root,
            "run/source_snapshot/source/cognitive_leap_three_way_receipt.py",
            b"# frozen three-way validator\n",
        ),
    }
    protocol_payload = {
        "schema": "supermix-cognitive-leap-bounded-protocol-v2",
        "protocol_sha256": "d" * 64,
        "evaluation_profile_sha256": (
            packager.THREE_WAY_EVALUATION_PROFILE_SHA256
        ),
        "baseline": release,
        "prior_candidate": prior,
        "source_snapshot": snapshots,
    }
    protocol_file = _json_artifact(root, "run/protocol.json", protocol_payload)
    protocol = {
        "path": protocol_file["path"],
        "file_sha256": protocol_file["sha256"],
        "size_bytes": protocol_file["size_bytes"],
        "content_sha256": protocol_payload["protocol_sha256"],
    }
    member_rows: list[dict[str, Any]] = []
    member_receipts: dict[str, Any] = {}
    for index, name in enumerate(("tempered_251", "tempered_351"), start=1):
        artifact = _artifact(
            root,
            f"run/members/{name}/weights.pth",
            f"{name}-state".encode("ascii"),
            canonical_state_sha256=str(index) * 64,
            tensor_count=4,
            element_count=12,
            strict_load=True,
        )
        training_payload = {
            "schema": "supermix-cognitive-leap-training-receipt-v2",
            "artifact": artifact,
            "parent_baseline": release,
            "config": {"name": name},
        }
        training = _json_artifact(
            root,
            f"run/members/{name}/training_receipt.json",
            training_payload,
        )
        member_receipts[name] = training_payload
        member_rows.append(
            {
                "name": name,
                "artifact": artifact,
                "training_receipt": training,
            }
        )
    lineage = _json_artifact(
        root,
        "run/lineage_manifest.json",
        {
            "schema": "supermix-cognitive-leap-lineage-v2",
            "baseline": release,
            "members": member_rows,
            "selected_artifact": candidate,
        },
        schema="supermix-cognitive-leap-lineage-v2",
    )
    lineage_verification = _json_artifact(
        root,
        "run/lineage_verification.json",
        {
            "schema": "supermix-cognitive-leap-lineage-verification-v1",
            "lineage_manifest": lineage,
            "selected_artifact": candidate,
        },
        schema="supermix-cognitive-leap-lineage-verification-v1",
    )
    selection_payload = {
        "schema": "supermix-cognitive-leap-development-selection-v2",
        "selection_sha256": "e" * 64,
        "protocol": protocol,
        "member_receipts": member_receipts,
        "selected": {"artifact": candidate},
        "lineage_manifest": lineage,
        "lineage_verification": lineage_verification,
    }
    selection_file = _json_artifact(root, "run/selection.json", selection_payload)
    selection = {
        "path": selection_file["path"],
        "file_sha256": selection_file["sha256"],
        "size_bytes": selection_file["size_bytes"],
        "content_sha256": selection_payload["selection_sha256"],
    }
    predictions = _artifact(
        root,
        "run/final_three_way_predictions.jsonl.gz",
        b"deterministic-three-way-gzip-placeholder",
        schema=packager.THREE_WAY_PREDICTION_SCHEMA,
        evaluation_profile_sha256=(
            packager.THREE_WAY_EVALUATION_PROFILE_SHA256
        ),
        uncompressed_sha256="f" * 64,
        row_count=40_000,
        dataset_id="0" * 64,
    )
    comparison = {
        "passed": True,
        "checks": {"bounded": True},
        "summary": {"accuracy_delta": 0.001, "mean_candidate_loss": 0.1},
        "evidence": {"dataset_sha256": "9" * 64},
    }
    receipt = {
        "schema": packager.THREE_WAY_EVALUATION_SCHEMA,
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": packager.CONTENT_BOUND_STATUS,
        "authority": dict(packager.BOUNDED_EVALUATION_AUTHORITY),
        "gate_outcome": "pass",
        "evaluation_profile_sha256": (
            packager.THREE_WAY_EVALUATION_PROFILE_SHA256
        ),
        "protocol": protocol,
        "selection": selection,
        "artifacts": {
            "release_baseline": release,
            "prior_candidate": prior,
            "candidate": candidate,
        },
        "comparisons": {
            "release_continuity": comparison,
            "prior_candidate_superiority": comparison,
        },
        "per_example_artifact": predictions,
        "receipt_id": "8" * 64,
    }
    path = root / "run/three_way_evaluation_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


def _valid_three_way_semantics(
    receipt_path: Path,
    root: Path,
) -> dict[str, Any]:
    del root
    payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    return {
        "valid": True,
        "schema": packager.THREE_WAY_EVALUATION_SCHEMA,
        "receipt_id": payload["receipt_id"],
        "gate_outcome": payload["gate_outcome"],
        "evaluation_profile_schema": (
            "supermix-cognitive-leap-evaluation-profile-v1"
        ),
        "evaluation_profile_sha256": payload["evaluation_profile_sha256"],
        "protocol_sha256": payload["protocol"]["content_sha256"],
        "selection_sha256": payload["selection"]["content_sha256"],
        "release_baseline_sha256": payload["artifacts"]["release_baseline"][
            "sha256"
        ],
        "prior_candidate_sha256": payload["artifacts"]["prior_candidate"][
            "sha256"
        ],
        "candidate_sha256": payload["artifacts"]["candidate"]["sha256"],
        "per_example_artifact_sha256": payload["per_example_artifact"]["sha256"],
        "release_continuity_passed": payload["comparisons"][
            "release_continuity"
        ]["passed"],
        "prior_candidate_superiority_passed": payload["comparisons"][
            "prior_candidate_superiority"
        ]["passed"],
        "checkpoint_inference_replayed": True,
        "authority": dict(packager.BOUNDED_EVALUATION_AUTHORITY),
    }


def test_unprofiled_v2_package_is_disabled_by_default(tmp_path: Path) -> None:
    receipt = _receipt_fixture(tmp_path)

    with pytest.raises(ValueError, match="Unprofiled bounded-v2"):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
        )

    assert not (tmp_path / "store.zip").exists()


def test_v2_package_requires_semantic_validator_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    output = tmp_path / "store.zip"
    monkeypatch.setattr(
        packager,
        "_validate_receipt_semantics",
        lambda *_args, **_kwargs: {"valid": False, "errors": ["forged"]},
    )

    with pytest.raises(ValueError, match="semantic validation"):
        packager.build_bounded_v2_package(
            output,
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )

    assert not output.exists()


def test_v2_package_rejects_validator_hash_cross_link_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    validation = _valid_semantics(receipt, root=tmp_path)
    validation["candidate_sha256"] = "f" * 64
    monkeypatch.setattr(
        packager,
        "_validate_receipt_semantics",
        lambda *_args, **_kwargs: validation,
    )

    with pytest.raises(ValueError, match="candidate_sha256 cross-link mismatch"):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )


@pytest.mark.parametrize("forgery", ["gate", "authority", "integrity"])
def test_v2_package_rejects_forged_receipt_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if forgery == "gate":
        payload["gate_outcome"] = "reject"
    elif forgery == "authority":
        payload["authority"]["store_publication"] = True
    else:
        payload["integrity_status"] = "authenticated"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(packager, "_validate_receipt_semantics", _valid_semantics)

    with pytest.raises(ValueError):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )


@pytest.mark.parametrize(
    "artifact_key",
    ["candidate", "lineage", "lineage_verification", "predictions"],
)
def test_v2_package_rejects_tampered_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_key: str,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    records = {
        "candidate": payload["artifacts"]["candidate"],
        "lineage": payload["selection"]["lineage_manifest"],
        "lineage_verification": payload["selection"]["lineage_verification"],
        "predictions": payload["per_example_artifact"],
    }
    (tmp_path / records[artifact_key]["path"]).write_bytes(b"tampered")
    monkeypatch.setattr(packager, "_validate_receipt_semantics", _valid_semantics)

    with pytest.raises(ValueError, match="does not match its binding"):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )


def test_passing_v2_package_is_manual_only_and_cross_linked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    output = tmp_path / "store.zip"
    observed: list[tuple[Path, Path]] = []

    def validate(path: Path, root: Path) -> dict[str, Any]:
        observed.append((path, root))
        return _valid_semantics(path, root=root)

    monkeypatch.setattr(packager, "_validate_receipt_semantics", validate)
    result = packager.build_bounded_v2_package(
        output,
        receipt_path=receipt,
        root=tmp_path,
        allow_unsafe_unprofiled_v2_for_tests=True,
    )

    assert observed[0] == (receipt.resolve(), tmp_path.resolve())
    assert observed[1][0].name == "bounded_evaluation_receipt.json"
    assert observed[1][0].parent == observed[1][1]
    assert observed[1][1] != tmp_path.resolve()
    assert not observed[1][1].exists()
    assert result["sha256"] == _sha256(output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json"))
        members = {row["name"]: row for row in manifest["members"]}
        assert manifest["schema"] == packager.ARTIFACT_MANIFEST_V2_SCHEMA
        assert manifest["manual_selectable"] is True
        assert manifest["manual_selection_only"] is True
        assert manifest["auto_route_allowed"] is False
        assert manifest["default_model_allowed"] is False
        assert manifest["receipt_grants_activation"] is False
        assert manifest["receipt_grants_store_publication"] is False
        assert manifest["integrity_status"] == packager.CONTENT_BOUND_STATUS
        assert manifest["runtime_status"] == packager.CONTENT_BOUND_STATUS
        assert manifest["receipt_authority"] == packager.BOUNDED_EVALUATION_AUTHORITY
        assert set(manifest["evidence_links"]) == {
            "baseline",
            "candidate",
            "chat_meta",
            "lineage",
            "lineage_verification",
            "predictions",
            "protocol",
            "receipt",
            "selection",
        }
        for link in manifest["evidence_links"].values():
            member = members[link["archive_member"]]
            assert member["sha256"] == link["sha256"]
            assert member["size_bytes"] == link["size_bytes"]
        assert manifest["evidence_links"]["candidate"][
            "canonical_state_sha256"
        ] == "1" * 64
        assert manifest["evidence_links"]["predictions"][
            "uncompressed_sha256"
        ] == "2" * 64
        closure = {
            row["archive_member"]: row
            for row in manifest["reproducibility_closure"]
        }
        assert set(closure) == {
            "run/baseline/baseline.pth",
            "run/bounded_evaluation_receipt.json",
            "run/lineage_manifest.json",
            "run/lineage_verification.json",
            "run/members/tempered_251/training_receipt.json",
            "run/members/tempered_251/weights.pth",
            "run/members/tempered_351/training_receipt.json",
            "run/members/tempered_351/weights.pth",
            "run/protocol.json",
            "run/selected/candidate.pth",
            "run/selection.json",
            "run/source_snapshot/source/benchmark_cognitive_leap_ultra_v51.py",
            "run/source_snapshot/source/cognitive_leap_receipt.py",
            "run/final_predictions.jsonl.gz",
        }
        assert all(row["reference_sites"] for row in closure.values())
        assert manifest["runtime_files"] == {
            "checkpoint": "cognitive_leap_ultra_v51_2.pth",
            "chat_metadata": "chat_demo_meta_v51_2.json",
            "bounded_receipt": "bounded_evaluation_receipt.json",
        }
        chat_meta = json.loads(archive.read("chat_demo_meta_v51_2.json"))
        assert chat_meta["checkpoint_path"] == "cognitive_leap_ultra_v51_2.pth"
        assert chat_meta["benchmark_metrics"] == "bounded_evaluation_receipt.json"
        assert chat_meta["model_store_policy"] == {
            "manual_selection_only": True,
            "auto_route_allowed": False,
            "default_model_allowed": False,
            "general_chat_claim": False,
            "general_reasoning_claim": False,
            "receipt_grants_activation": False,
            "receipt_grants_store_publication": False,
        }
        assert chat_meta["authority"] == packager.BOUNDED_EVALUATION_AUTHORITY


def test_v2_package_is_independent_of_original_evidence_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "evidence"
    receipt = _receipt_fixture(source_root)
    output = tmp_path / "store.zip"
    monkeypatch.setattr(packager, "_validate_receipt_semantics", _valid_semantics)

    packager.build_bounded_v2_package(
        output,
        receipt_path=receipt,
        root=source_root,
        allow_unsafe_unprofiled_v2_for_tests=True,
    )
    unavailable_root = tmp_path / "evidence-unavailable"
    source_root.rename(unavailable_root)
    candidate = unavailable_root / "run/selected/candidate.pth"
    candidate.write_bytes(b"original-tree-now-mutated")

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)
    validation = packager._validate_receipt_semantics(
        extracted / "bounded_evaluation_receipt.json",
        extracted,
    )
    assert validation["valid"] is True
    _payload, closure = packager._collect_reproducibility_closure(
        extracted / "bounded_evaluation_receipt.json",
        root=extracted,
    )
    assert closure["run/selected/candidate.pth"]["sha256"] == hashlib.sha256(
        b"candidate-state"
    ).hexdigest()
    assert (extracted / "run/members/tempered_351/training_receipt.json").is_file()
    assert (extracted / "run/source_snapshot/source/cognitive_leap_receipt.py").is_file()


def test_v2_package_rejects_missing_transitive_closure_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    (tmp_path / "run/members/tempered_351/weights.pth").unlink()
    monkeypatch.setattr(packager, "_validate_receipt_semantics", _valid_semantics)

    with pytest.raises(ValueError, match="missing"):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.pth", "C:\\outside.pth", "/outside.pth", "run/../outside.pth"],
)
def test_v2_package_rejects_reference_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    receipt = _receipt_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["artifacts"]["candidate"]["path"] = unsafe_path
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(packager, "_validate_receipt_semantics", _valid_semantics)

    with pytest.raises(ValueError, match="safe repository-relative path"):
        packager.build_bounded_v2_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
            allow_unsafe_unprofiled_v2_for_tests=True,
        )


def test_legacy_build_zip_keeps_v1_manifest_default(tmp_path: Path) -> None:
    source = tmp_path / "legacy.bin"
    source.write_bytes(b"legacy")
    output = tmp_path / "legacy.zip"

    packager.build_zip(
        output,
        files=(("legacy.bin", source),),
        generated=(),
        package_contract={"model_key": "legacy", "auto_route_allowed": False},
    )

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json"))
    assert manifest["schema"] == "supermix-model-store-artifact-manifest-v1"


def test_three_way_semantic_validator_uses_inference_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cognitive_leap_three_way_receipt as three_way_receipt

    receipt = _three_way_receipt_fixture(tmp_path)
    observed: list[tuple[Path, Path, bool]] = []

    def validate(
        receipt_path: Path,
        *,
        root: Path,
        verify_inference: bool = True,
    ) -> dict[str, Any]:
        observed.append((receipt_path, root, verify_inference))
        return {"valid": True}

    monkeypatch.setattr(three_way_receipt, "validate_receipt", validate)
    result = packager._validate_three_way_receipt_semantics(
        receipt.resolve(),
        tmp_path.resolve(),
    )

    assert result == {"valid": True}
    assert observed == [(receipt.resolve(), tmp_path.resolve(), True)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_profile_sha256", "0" * 64),
        ("release_continuity_passed", False),
        ("prior_candidate_superiority_passed", False),
        ("checkpoint_inference_replayed", False),
        ("authority", {**packager.BOUNDED_EVALUATION_AUTHORITY, "release": True}),
    ],
)
def test_three_way_package_rejects_unqualified_validator_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    receipt = _three_way_receipt_fixture(tmp_path)
    validation = _valid_three_way_semantics(receipt, tmp_path)
    validation[field] = value
    monkeypatch.setattr(
        packager,
        "_validate_three_way_receipt_semantics",
        lambda *_args, **_kwargs: validation,
    )

    with pytest.raises(ValueError, match="profiled passing validator"):
        packager.build_three_way_v1_package(
            tmp_path / "store.zip",
            receipt_path=receipt,
            root=tmp_path,
        )

    assert not (tmp_path / "store.zip").exists()


def test_three_way_package_is_manual_only_complete_and_offline_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _three_way_receipt_fixture(tmp_path)
    output = tmp_path / "store.zip"
    observed: list[tuple[Path, Path]] = []

    def validate(path: Path, root: Path) -> dict[str, Any]:
        observed.append((Path(path), Path(root)))
        return _valid_three_way_semantics(Path(path), Path(root))

    monkeypatch.setattr(
        packager,
        "_validate_three_way_receipt_semantics",
        validate,
    )
    result = packager.build_three_way_v1_package(
        output,
        receipt_path=receipt,
        root=tmp_path,
    )

    assert result["sha256"] == _sha256(output)
    assert observed[0] == (receipt.resolve(), tmp_path.resolve())
    assert observed[1][0].name == "three_way_evaluation_receipt.json"
    assert observed[1][0].parent == observed[1][1]
    assert observed[1][1] != tmp_path.resolve()
    assert not observed[1][1].exists()
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json"))
        member_rows = {row["name"]: row for row in manifest["members"]}
        closure = {
            row["archive_member"]: row
            for row in manifest["reproducibility_closure"]
        }
        assert manifest["schema"] == packager.ARTIFACT_MANIFEST_V2_SCHEMA
        assert manifest["status"] == "three_way_evaluation_pass_manual_only"
        assert manifest["manual_selectable"] is True
        assert manifest["manual_selection_only"] is True
        assert manifest["auto_route_allowed"] is False
        assert manifest["default_model_allowed"] is False
        assert manifest["receipt_grants_activation"] is False
        assert manifest["receipt_grants_store_publication"] is False
        assert manifest["receipt_authority"] == (
            packager.BOUNDED_EVALUATION_AUTHORITY
        )
        assert manifest["evaluation_profile_sha256"] == (
            packager.THREE_WAY_EVALUATION_PROFILE_SHA256
        )
        assert manifest["release_continuity_passed"] is True
        assert manifest["prior_candidate_superiority_passed"] is True
        assert manifest["checkpoint_inference_replayed"] is True
        assert manifest["portable_validation"] == {
            "receipt_archive_member": "three_way_evaluation_receipt.json",
            "root": ".",
            "validator": "cognitive_leap_three_way_receipt.validate_receipt",
            "verify_inference": True,
            "verified_after_temporary_extraction": True,
        }
        semantic = manifest["semantic_validation"]
        assert semantic == packager._three_way_semantic_projection(
            _valid_three_way_semantics(receipt, tmp_path)
        )
        assert manifest["semantic_validation_sha256"] == hashlib.sha256(
            packager.canonical_json(semantic)
        ).hexdigest()
        assert set(manifest["evidence_links"]) == {
            "receipt",
            "protocol",
            "selection",
            "release_baseline",
            "prior_candidate",
            "candidate",
            "lineage",
            "lineage_verification",
            "predictions",
            "chat_meta",
        }
        for link in manifest["evidence_links"].values():
            member = member_rows[link["archive_member"]]
            assert member["sha256"] == link["sha256"]
            assert member["size_bytes"] == link["size_bytes"]
        expected_closure = {
            "run/release/release_v51.pth",
            "run/prior/prior_v51_1.pth",
            "run/selected/candidate_v51_2.pth",
            "run/protocol.json",
            "run/selection.json",
            "run/lineage_manifest.json",
            "run/lineage_verification.json",
            "run/final_three_way_predictions.jsonl.gz",
            "run/three_way_evaluation_receipt.json",
            "run/members/tempered_251/weights.pth",
            "run/members/tempered_251/training_receipt.json",
            "run/members/tempered_351/weights.pth",
            "run/members/tempered_351/training_receipt.json",
            "run/source_snapshot/source/run_cognitive_leap_v51_2.py",
            "run/source_snapshot/source/cognitive_leap_three_way_receipt.py",
        }
        assert set(closure) == expected_closure
        assert manifest["runtime_files"] == {
            "checkpoint": "cognitive_leap_ultra_v51_2_three_way.pth",
            "chat_metadata": "chat_demo_meta_v51_2_three_way.json",
            "three_way_receipt": "three_way_evaluation_receipt.json",
        }
        chat_meta = json.loads(
            archive.read("chat_demo_meta_v51_2_three_way.json")
        )
        assert chat_meta["checkpoint_path"] == (
            "cognitive_leap_ultra_v51_2_three_way.pth"
        )
        assert chat_meta["benchmark_metrics"] == (
            "three_way_evaluation_receipt.json"
        )
        assert chat_meta["authority"] == packager.BOUNDED_EVALUATION_AUTHORITY


def test_three_way_package_deletes_output_when_offline_revalidation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _three_way_receipt_fixture(tmp_path)
    output = tmp_path / "store.zip"
    calls = 0

    def validate(path: Path, root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = _valid_three_way_semantics(Path(path), Path(root))
        if calls == 2:
            result["checkpoint_inference_replayed"] = False
        return result

    monkeypatch.setattr(
        packager,
        "_validate_three_way_receipt_semantics",
        validate,
    )

    with pytest.raises(ValueError, match="cross-link mismatch"):
        packager.build_three_way_v1_package(
            output,
            receipt_path=receipt,
            root=tmp_path,
        )

    assert calls == 2
    assert not output.exists()


def test_three_way_package_bytes_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _three_way_receipt_fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    monkeypatch.setattr(
        packager,
        "_validate_three_way_receipt_semantics",
        _valid_three_way_semantics,
    )

    packager.build_three_way_v1_package(
        first,
        receipt_path=receipt,
        root=tmp_path,
    )
    packager.build_three_way_v1_package(
        second,
        receipt_path=receipt,
        root=tmp_path,
    )

    assert first.read_bytes() == second.read_bytes()
