import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "source"))

from supermix_multimodel_web_app import build_app


class _StubManager:
    def __init__(self, zip_path: Path, summary_path: Path):
        self.records = []
        self.generated_dir = zip_path.parent
        self.uploads_dir = zip_path.parent
        self.chat_payloads = []
        self.preview_payloads = []
        self.study_payloads = []
        self.review_bundle_payloads = []
        self.review_audit_payloads = []
        self.shadow_status_calls = 0
        self.feedback_payloads = []
        self._store_rows = [
            {
                "file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip",
                "size_bytes": 1647669376,
                "size_mb": 1571.34,
                "family": "fusion",
                "known": True,
                "model_key": "omni_collective_v8_preview",
                "label": "Omni Collective V8 Preview",
                "kind": "omni_collective_v8",
                "capabilities": ["chat", "vision"],
                "note": "Preview snapshot.",
                "benchmark_hint": "Interim preview.",
                "download_url": "https://example.invalid/v8-preview.zip",
                "installed": False,
                "local_path": "",
                "selectable": False,
            },
            {
                "file_name": "dcgan_v2_in_progress.zip",
                "size_bytes": 61069961,
                "size_mb": 58.23,
                "family": "gan",
                "known": True,
                "model_key": "dcgan_v2_in_progress",
                "label": "DCGAN V2 CIFAR",
                "kind": "dcgan_image",
                "capabilities": ["image"],
                "note": "GAN image model.",
                "benchmark_hint": "",
                "download_url": "https://example.invalid/dcgan-v2.zip",
                "installed": True,
                "local_path": str(zip_path),
                "selectable": True,
            },
        ]
        self._jobs = [
            {
                "job_id": "store-1",
                "file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip",
                "status": "downloading",
                "downloaded_bytes": 512,
                "total_bytes": 1024,
                "started_at": "2026-04-07T18:00:00",
                "local_path": "",
                "error": "",
            }
        ]
        self._payload = {
            "key": "three_d_generation_micro_v1",
            "label": "3D Generation Micro",
            "zip_path": str(zip_path),
            "zip_name": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "summary_path": str(summary_path),
            "summary_name": summary_path.name,
            "parameter_count": 35886,
            "train_accuracy": 1.0,
            "val_accuracy": 1.0,
            "concept_count": 14,
            "source_rows": 144,
            "train_rows": 130,
            "val_rows": 14,
            "concept_labels": ["pyramid", "tetrahedron"],
            "sample_predictions": [
                {
                    "prompt": "Create a square pyramid.",
                    "predicted_label": "square pyramid",
                    "confidence": 0.98,
                }
            ],
        }

    def three_d_model_view(self):
        return dict(self._payload)

    def model_store_catalog(self, force_refresh: bool = False):
        return {
            "repo_id": "Kai9987kai/supermix-model-zoo",
            "model_count": len(self._store_rows),
            "models": list(self._store_rows),
        }

    def model_store_jobs(self):
        return {"jobs": list(self._jobs)}

    def install_model_store_artifact(self, file_name: str):
        return {
            "job_id": "store-new",
            "file_name": file_name,
            "status": "queued",
            "downloaded_bytes": 0,
            "total_bytes": 2048,
            "started_at": "2026-04-07T18:01:00",
            "local_path": "",
            "error": "",
        }

    def handle_prompt(self, *, session_id: str, prompt: str, model_key: str, action_mode: str, settings: dict):
        self.chat_payloads.append(
            {
                "session_id": session_id,
                "prompt": prompt,
                "model_key": model_key,
                "action_mode": action_mode,
                "settings": dict(settings),
            }
        )
        return {
            "ok": True,
            "kind": "text",
            "model_key": model_key,
            "model_label": "Stub",
            "route_reason": "stub",
            "response": "stub response",
            "agent_trace": {},
        }

    def preview_route_plan(self, *, session_id: str, prompt: str, model_key: str, action_mode: str, settings: dict):
        if model_key == "missing-model":
            raise KeyError("Unknown model key: missing-model")
        if model_key == "no-local-models":
            raise RuntimeError("No local models were discovered.")
        self.preview_payloads.append(
            {
                "session_id": session_id,
                "prompt": prompt,
                "model_key": model_key,
                "action_mode": action_mode,
                "settings": dict(settings),
            }
        )
        return {
            "ok": True,
            "dry_run": True,
            "selected_agent_mode": "collective",
            "route_economics_estimate": {
                "selected_agent_mode": "collective",
                "estimated_cost_units": 3.0,
            },
            "route_alternatives": [
                {
                    "selected_agent_mode": "off",
                    "is_selected": False,
                    "estimated_cost_units": 1.0,
                    "frontier_rank": 2,
                    "budget_fit": True,
                    "budget_blocker": None,
                    "pareto_frontier": True,
                    "budget_feasible_pareto_frontier": True,
                },
                {
                    "selected_agent_mode": "collective",
                    "is_selected": True,
                    "estimated_cost_units": 3.0,
                    "frontier_rank": 1,
                    "budget_fit": True,
                    "budget_blocker": None,
                    "pareto_frontier": True,
                    "budget_feasible_pareto_frontier": True,
                },
            ],
            "route_frontier": {
                "selected_agent_mode": "collective",
                "recommended_agent_mode": "collective",
                "recommended_reason": "selected_route_is_frontier_recommended",
                "recommended_budget_blocker": None,
                "selected_budget_blocker": None,
                "budget_blockers": {"remaining_budget": 0, "pacing_cap": 0, "effective_cap": 0, "none": 2},
                "pareto_modes": ["off", "collective"],
                "budget_feasible_pareto_modes": ["off", "collective"],
                "budget_cap_cost_units": 12.0,
                "remaining_cost_units": 12.0,
                "pacing_cap_cost_units": None,
                "effective_cap_cost_units": 12.0,
                "selected_matches_recommendation": True,
                "ranked_modes": [
                    {
                        "selected_agent_mode": "collective",
                        "frontier_rank": 1,
                        "budget_feasible_pareto_frontier": True,
                    },
                    {
                        "selected_agent_mode": "off",
                        "frontier_rank": 2,
                        "budget_feasible_pareto_frontier": True,
                    },
                ],
            },
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
            },
        }

    def preview_route_study(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: dict,
        exploration_rate: float,
        planned_routes: int,
        scenario_confidence: float,
        assumed_feedback_rate: float,
        target_observed_labels: int,
        target_policy_profile: str,
        protocol_design_mode: str,
        carryover_scope: str,
        interference_scope: str,
        temporal_variation: str,
        planned_clusters: int,
        max_routes_per_cluster: int,
        analysis_every_clusters: int,
        block_length_routes: int,
        washout_routes: int,
    ):
        request = {
            "session_id": session_id,
            "prompt": prompt,
            "model_key": model_key,
            "action_mode": action_mode,
            "settings": dict(settings),
            "exploration_rate": exploration_rate,
            "planned_routes": planned_routes,
            "scenario_confidence": scenario_confidence,
            "assumed_feedback_rate": assumed_feedback_rate,
            "target_observed_labels": target_observed_labels,
            "target_policy_profile": target_policy_profile,
            "protocol_design_mode": protocol_design_mode,
            "carryover_scope": carryover_scope,
            "interference_scope": interference_scope,
            "temporal_variation": temporal_variation,
            "planned_clusters": planned_clusters,
            "max_routes_per_cluster": max_routes_per_cluster,
            "analysis_every_clusters": analysis_every_clusters,
            "block_length_routes": block_length_routes,
            "washout_routes": washout_routes,
        }
        self.study_payloads.append(request)
        label_probability = 0.05 * float(assumed_feedback_rate)
        return {
            "ok": True,
            "dry_run": True,
            "baseline_agent_mode": "collective",
            "route_study": {
                "schema_version": "route-exploration-plan-v1",
                "design_hash": "a" * 64,
                "study": {
                    "study_id": "auto-route-adjacent-explorer-v1",
                    "study_version": "1.0.0",
                },
                "charter": {
                    "enrollment": {
                        "eligible": True,
                        "reason": "eligible_adjacent_post_filter_support",
                        "baseline_action": "collective",
                        "adjacent_feasible_actions": ["off", "loop"],
                    },
                    "probability_design": {
                        "decision_type": "randomized",
                        "applied_exploration_rate": exploration_rate,
                        "minimum_positive_exploration_probability": exploration_rate / 2,
                        "action_probabilities": {
                            "off": exploration_rate / 2,
                            "collective": 1 - exploration_rate,
                            "loop": exploration_rate / 2,
                        },
                        "assignment_performed": False,
                    },
                    "traffic_scenario": {
                        "planned_routes": planned_routes,
                        "observed_label_scenario": {
                            "analysis_type": "simultaneous_alternate_label_traffic_not_power_or_mnar_correction",
                            "target_scope": "at_least_target_observed_labels_on_every_alternate_action",
                            "alternate_actions": ["off", "loop"],
                            "assumed_feedback_rate": assumed_feedback_rate,
                            "target_observed_labels_per_alternate_action": target_observed_labels,
                            "per_route_observed_label_probability_by_alternate_action": {
                                "off": label_probability,
                                "loop": label_probability,
                            },
                            "expected_routes_for_target_by_alternate_action": {
                                "off": target_observed_labels / label_probability,
                                "loop": target_observed_labels / label_probability,
                            },
                            "exact_simultaneous_target": {
                                "method": "exact_joint_multinomial_tail_inversion_two_alternates",
                                "confidence_level": scenario_confidence,
                                "minimum_routes_for_target_on_every_alternate_action": 1971,
                            },
                        },
                    },
                    "resource_forecast": {
                        "expected_for_planned_routes": {"cost_units": 6700.0},
                        "by_action": {
                            "off": {"latency_tier": "low"},
                            "collective": {"latency_tier": "moderate"},
                            "loop": {"latency_tier": "frontier"},
                        },
                    },
                    "causal_boundaries": {
                        "execution_enabled": False,
                        "off_policy_estimate_computed": False,
                        "automatic_promotion_allowed": False,
                        "activation_available": False,
                        "activation_blockers": [
                            "preassignment_seed_commitment_not_sealed",
                            "target_policy_class_not_precommitted",
                            "session_carryover_not_addressed",
                            "external_ope_not_validated",
                        ],
                    },
                },
            },
            "route_protocol_preflight": {
                "schema_version": "route-study-protocol-preflight-v1",
                "protocol_hash": "b" * 64,
                "protocol": {
                    "version": "1.0.0",
                    "label": "Stateful Route Experiment Preflight v1",
                    "state": "draft_for_independent_review",
                    "activation_available": False,
                },
                "charter": {
                    "target_policy_class": {
                        "profile_name": target_policy_profile,
                        "class_hash": "c" * 64,
                    },
                    "population": {
                        "planned_clusters": planned_clusters,
                        "max_routes_per_cluster": max_routes_per_cluster,
                    },
                    "stateful_design": {
                        "selected_design_mode": protocol_design_mode,
                        "selected_design_status": "declaration_incomplete",
                        "assignment_unit": "session_hash",
                        "carryover_scope": carryover_scope,
                        "interference_scope": interference_scope,
                        "temporal_variation": temporal_variation,
                        "selected_design_blocking_reasons": [
                            "carryover_scope_unknown",
                            "interference_scope_unknown",
                            "temporal_variation_unknown",
                        ],
                    },
                    "randomness": {
                        "assignment_implementation_available": False,
                        "assignment_performed": False,
                    },
                    "blocker_register": [
                        {
                            "code": "session_carryover_not_addressed",
                            "status": "unresolved",
                            "activation_blocking": True,
                        },
                        {
                            "code": "interference_not_addressed",
                            "status": "unresolved",
                            "activation_blocking": True,
                        },
                    ],
                    "causal_boundaries": {
                        "activation_available": False,
                        "automatic_promotion_allowed": False,
                        "activation_blockers": [
                            "session_carryover_not_addressed",
                            "interference_not_addressed",
                        ],
                    },
                },
            },
            "route_protocol_preflight_reason": "draft_for_independent_review",
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }

    def build_route_protocol_review_bundle(self, payload: dict):
        from route_policy_protocol import (
            audit_route_study_review_bundle,
            build_route_study_review_bundle_from_input,
        )

        self.review_bundle_payloads.append(payload)
        bundle = build_route_study_review_bundle_from_input(payload)
        return {
            "ok": True,
            "dry_run": True,
            "route_protocol_review_bundle": bundle,
            "verification": audit_route_study_review_bundle(bundle),
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }

    def audit_route_protocol_review_bundle(self, bundle: dict):
        from route_policy_protocol import audit_route_study_review_bundle

        self.review_audit_payloads.append(bundle)
        return {
            "ok": True,
            "dry_run": True,
            "verification": audit_route_study_review_bundle(bundle),
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }

    def record_route_feedback(self, *, session_id: str, feedback: dict):
        self.feedback_payloads.append({"session_id": session_id, "feedback": dict(feedback)})
        return {
            "ok": True,
            "feedback": {
                "selected_agent_mode": feedback.get("selected_agent_mode"),
                "rating": feedback.get("rating"),
            },
            "summary": {
                "total_feedback": len(self.feedback_payloads),
                "economics": {
                    "sample_count": len(self.feedback_payloads),
                    "avg_cost_units": 4.5,
                    "avg_elapsed_ms": 120.0,
                },
            },
        }

    def route_health_snapshot(self, session_id: str):
        return {
            "total_feedback": len(self.feedback_payloads),
            "economics": {
                "sample_count": len(self.feedback_payloads),
                "avg_cost_units": 4.5 if self.feedback_payloads else None,
                "avg_elapsed_ms": 120.0 if self.feedback_payloads else None,
            },
            "adaptive": {
                "sample_count": len(self.feedback_payloads),
                "quality_score": 0.42 if self.feedback_payloads else None,
                "quality_cost_score": 0.31 if self.feedback_payloads else None,
                "regression_signal": bool(self.feedback_payloads),
            },
            "route_usage": {
                "total_routes": len(self.feedback_payloads),
                "economics": {
                    "sample_count": len(self.feedback_payloads),
                    "avg_cost_units": 4.5 if self.feedback_payloads else None,
                    "total_cost_units": 4.5 if self.feedback_payloads else None,
                    "avg_elapsed_ms": 120.0 if self.feedback_payloads else None,
                },
            },
        }

    def route_policy_lab_snapshot(self, session_id: str, profile: str = "balanced"):
        return {
            "analysis_kind": "associational_matched_route_replay",
            "evidence_source": "durable_sqlite_ledger",
            "profile": {"name": profile, "thresholds": {"off": 0, "collective": 2, "loop": 4, "collective_loop": 5}},
            "support": {
                "usage": {"unique_route_ids": 1},
                "exact_joined_route_ids": 1,
            },
            "candidate_action_agreement": {"agreement_rate": 1.0},
            "matched_observed": {"quality_sample_count": 1, "approval_rate": 1.0},
            "propensity_readiness": {"valid_routes": 0, "checked_evaluable_usage_routes": 3},
            "evaluation_readiness": {
                "schema_version": "route-readiness-v2",
                "thresholds": {
                    "minimum_global_effective_sample_size": 20.0,
                    "minimum_per_action_effective_sample_size": 10.0,
                },
                "target_overlap": {
                    "effective_sample_size": 0.0,
                    "weakest_target_action": "collective",
                    "weakest_action_effective_sample_size": 0.0,
                    "minimum_target_probability": None,
                },
                "outcome_observation": {
                    "quality_observed_routes": 2,
                    "evaluable_routes": 3,
                },
                "ready_for_external_ope": False,
                "policy_value_estimated": False,
            },
            "outcome_contract_maturity": {
                "schema_version": "route-outcome-maturity-v1",
                "included_routes": 3,
                "precommitted_routes": 2,
                "by_outcome": {
                    "user_quality_rating": {
                        "mature_contract_count": 3,
                        "observed_event_count": 2,
                    }
                },
                "descriptive_only": True,
                "policy_value_estimate": None,
            },
            "durable_ledger": {
                "ledger_schema_version": 2,
                "counts": {"started": 3, "completed": 2, "failed": 1, "inflight": 0},
                "feedback_coverage": {"terminal_coverage_rate": 0.666667},
            },
            "promotion_gate": {
                "status": "blocked",
                "deployment": "shadow_only",
                "reason_code": "no_valid_randomized_overlap",
                "blocking_reason_codes": [
                    "no_valid_randomized_overlap",
                    "insufficient_global_overlap_ess",
                ],
                "checks": {
                    "candidate_delta_present": True,
                    "population_integrity_complete": False,
                    "execution_integrity_complete": False,
                    "logging_integrity_complete": False,
                    "minimum_overlap_routes_met": False,
                    "target_probability_floor_met": False,
                    "global_overlap_ess_met": False,
                    "per_action_overlap_met": False,
                    "outcome_evidence_integrity": False,
                    "quality_observation_ready": False,
                    "durable_lifecycle_present": True,
                    "lifecycle_reconciled": False,
                },
                "passed_checks": 2,
                "total_checks": 12,
                "automatic_promotion_allowed": False,
            },
        }

    def route_shadow_registry_snapshot(self):
        self.shadow_status_calls += 1
        return {
            "ok": True,
            "available": True,
            "status": "verified",
            "read_only": True,
            "campaign_count": 1,
            "campaigns": [
                {
                    "campaign_id": "shadow:" + "a" * 64,
                    "state": "commitments_closed",
                    "commitment_count": 2,
                    "verified_assignment_count": 0,
                    "mismatched_assignment_count": 0,
                }
            ],
            "event_chain": {"ok": True, "verified_events": 4},
            "execution_enabled": False,
            "activation_available": False,
            "automatic_promotion_allowed": False,
        }


def test_three_d_model_view_endpoint_and_downloads():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_bytes = b"zip-bytes"
        summary_bytes = b'{"artifact":"supermix_3d_generation_micro_v1_20260403.zip"}'
        zip_path.write_bytes(zip_bytes)
        summary_path.write_bytes(summary_bytes)

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        response = client.get("/api/three_d_model_view")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["model"]["key"] == "three_d_generation_micro_v1"
        assert payload["model"]["download_zip_url"] == "/download/three_d_model_zip"
        assert payload["model"]["download_summary_url"] == "/download/three_d_model_summary"

        zip_response = client.get("/download/three_d_model_zip")
        assert zip_response.status_code == 200
        assert zip_response.data == zip_bytes
        zip_response.close()

        summary_response = client.get("/download/three_d_model_summary")
        assert summary_response.status_code == 200
        assert summary_response.data == summary_bytes
        summary_response.close()


def test_index_contains_discovery_ui():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'id="modelSearch"' in html
        assert 'id="capabilityFilter"' in html
        assert 'id="quickPickChips"' in html
        assert 'id="discoveryNote"' in html
        assert 'id="sessionObjective"' in html
        assert 'id="savedDrafts"' in html
        assert 'id="contextBankList"' in html
        assert 'id="captureLastReplyBtn"' in html
        assert 'id="threadBookmarks"' in html
        assert 'id="compareSummary"' in html
        assert 'id="dispatchPreview"' in html
        assert 'id="modelStoreList"' in html
        assert 'id="refreshStoreBtn"' in html
        assert 'id="appShell"' in html
        assert 'id="composeScroll"' in html
        assert 'id="composeQuickBtn"' in html
        assert 'id="composeMediaBtn"' in html
        assert 'id="composeWorkbenchBtn"' in html
        assert 'id="loopBudget"' in html
        assert 'id="autoBudget"' in html
        assert 'id="reasoningCycles"' in html
        assert 'id="adaptiveCompute"' in html
        assert 'id="progressiveAutoCompute"' in html
        assert 'id="predictionStabilityMargin"' in html
        assert 'id="predictionStabilityRankDepth"' in html
        assert 'id="sessionBudget"' in html
        assert 'id="sessionBudgetTargetRoutes"' in html
        assert 'data-mode="auto"' in html
        assert "Adaptive Router" in html
        assert "Scoring task complexity" in html
        assert "auto_agent_budget" in html
        assert "auto_session_budget_units" in html
        assert "auto_session_budget_target_routes" in html
        assert "budget_profile" in html
        assert "session_budget_adjustment" in html
        assert 'id="routeFeedback"' in html
        assert 'id="routeGoodBtn"' in html
        assert 'id="routeBadBtn"' in html
        assert 'id="routeDeeperBtn"' in html
        assert 'id="routeCostBtn"' in html
        assert 'id="routeSlowBtn"' in html
        assert 'id="routeHealthConfidence"' in html
        assert 'id="routeHealthPreference"' in html
        assert 'id="routePlanBtn"' in html
        assert 'id="routeHealth"' in html
        assert 'id="routeHealthCount"' in html
        assert 'id="routeHealthQuality"' in html
        assert 'id="policyLab"' in html
        assert 'id="policyLabProfile"' in html
        assert 'id="policyLabGate"' in html
        assert 'id="policyLabLifecycle"' in html
        assert 'id="policyLabFeedbackCoverage"' in html
        assert 'id="policyLabOverlapEss"' in html
        assert 'id="policyLabWeakestAction"' in html
        assert 'id="policyLabReadinessChecks"' in html
        assert 'id="policyLabOutcomeCoverage"' in html
        assert 'id="policyLabContractCoverage"' in html
        assert 'id="policyLabEvidenceSource"' in html
        assert 'id="policyLabChecks"' in html
        assert 'id="policyLabBlockers"' in html
        assert 'id="routeStudy"' in html
        assert 'id="routeStudyPreview"' in html
        assert 'id="routeStudyHorizon"' in html
        assert 'id="routeStudyEpsilon"' in html
        assert 'id="routeStudyResponseRate"' in html
        assert 'id="routeStudyTargetLabels"' in html
        assert 'id="routeProtocolTarget"' in html
        assert 'id="routeProtocolDesign"' in html
        assert 'id="routeProtocolCarryover"' in html
        assert 'id="routeProtocolInterference"' in html
        assert 'id="routeProtocolTemporal"' in html
        assert 'id="routeProtocolClusters"' in html
        assert 'id="routeProtocolBlock"' in html
        assert 'id="routeProtocolWashout"' in html
        assert 'id="routeStudyStatus" role="status" aria-live="polite"' in html
        assert 'id="routeStudyDistribution"' in html
        assert 'id="routeProtocolMode"' in html
        assert 'id="routeProtocolPolicy"' in html
        assert 'id="routeProtocolReview"' in html
        assert 'id="routeProtocolHash"' in html
        assert 'id="routeProtocolBlockers"' in html
        assert 'id="routeBundleAdd"' in html
        assert 'id="routeBundleBuild"' in html
        assert 'id="routeBundleDownload"' in html
        assert 'id="routeBundleImport"' in html
        assert 'id="routeBundleClear"' in html
        assert 'id="routeBundleInventory"' in html
        assert 'id="routeBundleVerification"' in html
        assert 'id="routeShadowRegistry"' in html
        assert 'id="routeShadowRegistryRefresh"' in html
        assert 'id="routeShadowRegistryStatus" role="status" aria-live="polite"' in html
        assert "full_source_bound_reconstruction" in html
        assert "Bounded Exposure Rehearsal" in html
        assert "Rehearsal only - execution off" in html
        assert 'id="controlPanel"' in html
        assert 'id="panelToggle"' in html
        assert 'aria-controls="controlPanel"' in html
        assert 'aria-label="Open control panel"' in html
        assert 'id="panelClose"' in html
        assert '<div class="panel-backdrop" id="panelBackdrop" aria-hidden="true"></div>' in html
        assert "function setPanelOpen" in html
        assert "function openPanelTab" in html
        assert "openPanelTab('mode')" in html
        assert "renderPolicyLabUnavailable" in html
        assert ".panel.is-open" in html
        assert ".panel { display:none; }" not in html
        assert "/api/route_plan" in html
        assert "/api/route_feedback" in html
        assert "/api/route_health" in html
        assert "/api/route_policy_lab" in html
        assert "/api/route_study_plan" in html
        assert "/api/route_study_protocol_bundle" in html
        assert "/api/route_shadow_registry/status" in html
        assert "function refreshRouteShadowRegistry" in html
        assert "cannot seal, assign, reveal, activate, or promote" in html
        assert "/api/route_shadow_registry/seal" not in html
        assert "function renderRouteStudy" in html
        assert "function previewRouteStudy" in html
        assert "function buildRouteReviewBundle" in html
        assert "function importRouteReviewFile" in html
        assert "exact_simultaneous_target" in html
        assert "% joint" in html
        assert "Hypothetical repetition of this prompt-specific support" in html
        assert "Activation blockers" in html
        assert "Declarations are not validation" in html
        assert "route_protocol_preflight" in html
        assert "protocol_design_mode" in html
        assert "willRandomizeRoute" not in html
        assert "Activate study" not in html
        assert "No policy value was estimated" in html
        assert "Nondurable compatibility evidence; readiness is blocked." in html
        assert "maturity is diagnostic only" in html
        assert "buildRoutePayload" in html
        assert '<option value="auto">Auto Router</option>' in html
        assert '<option value="loop">Loop Agent</option>' in html
        assert '<option value="collective_loop">Collective + Loop</option>' in html
        assert "function scorePct" in html
        assert "auto_agent_policy" in html
        assert "loop_stop_reason_code" in html
        assert "route_economics" in html
        assert "function computePills" in html
        assert "requested_reasoning_cycles" in html
        assert "prediction_confidence_delta" in html
        assert "auto_compute_plan" in html
        assert "Accepted probe reused" in html
        assert "prediction_stability_margin" in html
        assert "Top-1 margin" in html
        assert "Decision margin" in html
        assert "prediction_stability_rank_depth" in html
        assert "prediction_class_count" in html
        assert "Verifier scope" in html
        assert "rows[selectedIndex]" in html
        assert "rows[rows.length - 1].mutual_stability_shadow" not in html
        assert "route_alternatives" in html
        assert "route_frontier" in html
        assert "recommended_agent_mode" in html
        assert "recommended_budget_blocker" in html
        assert "recommended_estimated_quality_cost_score" in html
        assert "selected_budget_blocker" in html
        assert "estimated_quality_cost_score" in html
        assert "quality_evidence_status" in html
        assert "quality_source" in html
        assert "budget_feasible_pareto_modes" in html
        assert "budget_feasible_pareto_frontier" in html
        assert "frontier_rank" in html
        assert "remaining_cost_units" in html
        assert "pacing_cap_cost_units" in html
        assert "effective_cap_cost_units" in html
        assert "route_usage" in html
        assert "routeEconomicsPills" in html
        assert "renderRouteHealth" in html
        assert "routeQualityText" in html
        assert "adaptive_quality_cost_preferred_neighbor" in html
        assert "Pareto" in html
        assert "Planned calls" in html
        assert "Session budget" in html
        assert "Paced" in html
        assert "review_score" in html
        assert "trace-score" in html
        assert 'id="toggleSidebarBtn"' in html
        assert 'id="toggleThreadDensityBtn"' in html
        assert 'id="responseDeck"' in html
        assert 'id="deliverableTarget"' in html
        assert 'id="successChecks"' in html
        assert 'id="riskBox"' in html
        assert 'id="confidenceMode"' in html
        assert 'id="evidenceMode"' in html
        assert 'id="clarifyMode"' in html
        assert 'id="assumptionMode"' in html
        assert 'id="refinementDeck"' in html
        assert 'id="refineLastReplyBtn"' in html
        assert 'id="challengeLastReplyBtn"' in html


def test_chat_endpoint_passes_session_budget_setting_to_manager():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/chat",
            json={
                "session_id": "chat-session",
                "message": "Run a budgeted route.",
                "model_key": "auto",
                "action_mode": "text",
                "settings": {
                    "agent_mode": "auto",
                    "auto_agent_budget": "balanced",
                    "auto_session_budget_units": 7.5,
                    "auto_session_budget_target_routes": 4,
                },
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert stub.chat_payloads[0]["session_id"] == "chat-session"
        assert stub.chat_payloads[0]["settings"]["auto_session_budget_units"] == 7.5
        assert stub.chat_payloads[0]["settings"]["auto_session_budget_target_routes"] == 4


def test_route_plan_endpoint_passes_payload_to_preview_without_chat():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/route_plan",
            json={
                "session_id": "preview-session",
                "message": "Preview this budgeted route.",
                "model_key": "auto",
                "action_mode": "text",
                "settings": {
                    "agent_mode": "auto",
                    "auto_session_budget_units": 12,
                    "auto_session_budget_target_routes": 4,
                },
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["dry_run"] is True
        assert payload["execution_plan"]["will_run_inference"] is False
        assert payload["route_alternatives"][1]["selected_agent_mode"] == "collective"
        assert payload["route_alternatives"][1]["is_selected"] is True
        assert payload["route_frontier"]["recommended_agent_mode"] == "collective"
        assert payload["route_frontier"]["ranked_modes"][0]["frontier_rank"] == 1
        assert payload["route_frontier"]["remaining_cost_units"] == 12.0
        assert payload["route_frontier"]["effective_cap_cost_units"] == 12.0
        assert payload["route_frontier"]["selected_budget_blocker"] is None
        assert payload["route_frontier"]["budget_blockers"]["none"] == 2
        assert payload["route_frontier"]["budget_feasible_pareto_modes"] == ["off", "collective"]
        assert stub.chat_payloads == []
        assert stub.preview_payloads[0]["session_id"] == "preview-session"
        assert stub.preview_payloads[0]["settings"]["auto_session_budget_target_routes"] == 4


def test_route_study_endpoint_rehearses_without_chat_or_assignment():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/route_study_plan",
            json={
                "session_id": "study-session",
                "message": "Research and verify this integration.",
                "model_key": "auto",
                "action_mode": "text",
                "settings": {"agent_mode": "auto", "auto_agent_budget": "balanced"},
                "exploration_rate": 0.10,
                "planned_routes": 2000,
                "scenario_confidence": 0.95,
                "assumed_feedback_rate": 0.30,
                "target_observed_labels": 20,
                "target_policy_profile": "quality_first",
                "protocol_design_mode": "clustered_switchback",
                "carryover_scope": "within_session",
                "interference_scope": "shared_resource",
                "temporal_variation": "nonstationary",
                "planned_clusters": 480,
                "max_routes_per_cluster": 12,
                "analysis_every_clusters": 60,
                "block_length_routes": 20,
                "washout_routes": 4,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["dry_run"] is True
        assert payload["route_study"]["study"]["study_id"] == "auto-route-adjacent-explorer-v1"
        assert payload["route_study"]["charter"]["probability_design"]["assignment_performed"] is False
        assert payload["route_protocol_preflight"]["protocol"]["activation_available"] is False
        assert payload["route_protocol_preflight"]["charter"]["stateful_design"][
            "selected_design_mode"
        ] == "clustered_switchback"
        assert payload["route_protocol_preflight"]["charter"]["randomness"][
            "assignment_performed"
        ] is False
        assert payload["execution_plan"] == {
            "will_run_inference": False,
            "will_write_memory": False,
            "will_write_ledger": False,
            "will_assign_route": False,
            "will_randomize": False,
            "activation_available": False,
        }
        assert stub.chat_payloads == []
        assert stub.preview_payloads == []
        assert len(stub.study_payloads) == 1
        request = stub.study_payloads[0]
        assert request["session_id"] == "study-session"
        assert request["exploration_rate"] == 0.10
        assert request["planned_routes"] == 2000
        assert request["assumed_feedback_rate"] == 0.30
        assert request["target_observed_labels"] == 20
        assert request["target_policy_profile"] == "quality_first"
        assert request["protocol_design_mode"] == "clustered_switchback"
        assert request["carryover_scope"] == "within_session"
        assert request["interference_scope"] == "shared_resource"
        assert request["temporal_variation"] == "nonstationary"
        assert request["planned_clusters"] == 480
        assert request["max_routes_per_cluster"] == 12
        assert request["analysis_every_clusters"] == 60
        assert request["block_length_routes"] == 20
        assert request["washout_routes"] == 4


def test_route_protocol_bundle_endpoints_build_and_reconstruct_without_chat():
    from route_policy_protocol_cli import _example_bundle_input

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()
        build_input = _example_bundle_input()
        response = client.post("/api/route_study_protocol_bundle", json=build_input)

        assert response.status_code == 200
        payload = response.get_json()
        bundle = payload["route_protocol_review_bundle"]
        assert payload["verification"]["verification_level"] == (
            "full_source_bound_reconstruction"
        )
        assert payload["verification"]["support_stratum_count"] == 2
        assert payload["execution_plan"]["will_run_inference"] is False
        assert payload["execution_plan"]["will_write_ledger"] is False
        assert stub.chat_payloads == []
        assert stub.preview_payloads == []
        assert stub.study_payloads == []
        assert len(stub.review_bundle_payloads) == 1

        audit_response = client.post(
            "/api/route_study_protocol_bundle/audit", json={"bundle": bundle}
        )
        assert audit_response.status_code == 200
        audit = audit_response.get_json()["verification"]
        assert audit["ok"] is True
        assert audit["source_plan_reconstruction_performed"] is True
        assert len(stub.review_audit_payloads) == 1
        assert stub.chat_payloads == []

        unsafe = client.post(
            "/api/route_study_protocol_bundle",
            json={**build_input, "prompt": "must not enter review artifacts"},
        )
        assert unsafe.status_code == 400
        assert "non-prompt-free fields: prompt" in unsafe.get_json()["error"]

        too_many = client.post(
            "/api/route_study_protocol_bundle",
            json={**build_input, "study_plans": build_input["study_plans"] * 51},
        )
        assert too_many.status_code == 400
        assert "at most 100 browser strata" in too_many.get_json()["error"]

        duplicate_json = json.dumps(build_input, separators=(",", ":"))[:-1] + (
            ',"study_plans":[]}'
        )
        duplicate = client.post(
            "/api/route_study_protocol_bundle",
            data=duplicate_json,
            content_type="application/json",
        )
        assert duplicate.status_code == 400
        assert "duplicate object key: study_plans" in duplicate.get_json()["error"]
        assert len(stub.review_bundle_payloads) == 2


def test_route_shadow_registry_endpoint_is_get_only_and_read_only():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()
        response = client.get("/api/route_shadow_registry/status")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["route_shadow_registry"]["read_only"] is True
        assert payload["route_shadow_registry"]["campaign_count"] == 1
        assert payload["route_shadow_registry"]["execution_enabled"] is False
        assert payload["route_shadow_registry"]["activation_available"] is False
        assert stub.shadow_status_calls == 1
        assert client.post("/api/route_shadow_registry/status", json={}).status_code == 405
        assert client.post("/api/route_shadow_registry/seal", json={}).status_code == 404
        assert stub.chat_payloads == []


def test_route_plan_endpoint_returns_compact_error_for_bad_model_key():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/route_plan",
            json={
                "session_id": "preview-session",
                "message": "Preview this route.",
                "model_key": "missing-model",
                "action_mode": "text",
                "settings": {"agent_mode": "auto"},
            },
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["ok"] is False
        assert "missing-model" in payload["error"]
        assert stub.chat_payloads == []
        assert stub.preview_payloads == []


def test_route_plan_endpoint_returns_compact_error_when_no_models_are_discovered():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/route_plan",
            json={
                "session_id": "preview-session",
                "message": "Preview this route.",
                "model_key": "no-local-models",
                "action_mode": "text",
                "settings": {"agent_mode": "auto"},
            },
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["ok"] is False
        assert "No local models were discovered" in payload["error"]
        assert stub.chat_payloads == []
        assert stub.preview_payloads == []


def test_route_feedback_endpoint_passes_compact_payload_to_manager():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        app = build_app(stub)
        client = app.test_client()

        response = client.post(
            "/api/route_feedback",
            json={
                "session_id": "route-session",
                "route_id": "route-1",
                "prompt": "Research and implement.",
                "response": "done",
                "selected_agent_mode": "collective_loop",
                "rating": "down",
                "feedback_intent": "too_costly",
                "auto_agent_policy": {"selected_agent_mode": "collective_loop", "score": 5},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 120.0,
                        "model_calls": 4,
                        "cost_units": 4.5,
                    },
                },
                "model_key": "omni_collective_v8",
                "route_reason": "Auto orchestration selected collective_loop.",
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["summary"]["total_feedback"] == 1
        assert stub.feedback_payloads[0]["session_id"] == "route-session"
        assert stub.feedback_payloads[0]["feedback"]["route_id"] == "route-1"
        assert stub.feedback_payloads[0]["feedback"]["rating"] == "down"
        assert stub.feedback_payloads[0]["feedback"]["feedback_intent"] == "too_costly"
        assert "selected_agent_mode" not in stub.feedback_payloads[0]["feedback"]
        assert "route_economics" not in stub.feedback_payloads[0]["feedback"]


def test_route_health_endpoint_returns_session_route_economics():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        stub = _StubManager(zip_path, summary_path)
        stub.feedback_payloads.append({"session_id": "route-session", "feedback": {}})
        app = build_app(stub)
        client = app.test_client()

        response = client.post("/api/route_health", json={"session_id": "route-session"})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["route_health"]["total_feedback"] == 1
        assert payload["route_health"]["economics"]["avg_cost_units"] == 4.5
        assert payload["route_health"]["adaptive"]["quality_score"] == 0.42
        assert payload["route_health"]["adaptive"]["regression_signal"] is True
        assert payload["route_health"]["route_usage"]["total_routes"] == 1
        assert payload["route_health"]["route_usage"]["economics"]["total_cost_units"] == 4.5


def test_route_policy_lab_endpoint_is_read_only_and_shadow_gated():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()
        response = client.post(
            "/api/route_policy_lab",
            json={"session_id": "route-session", "profile": "efficiency"},
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["policy_lab"]["profile"]["name"] == "efficiency"
        assert payload["policy_lab"]["analysis_kind"] == "associational_matched_route_replay"
        assert payload["policy_lab"]["promotion_gate"]["deployment"] == "shadow_only"
        assert payload["policy_lab"]["promotion_gate"]["automatic_promotion_allowed"] is False
        assert payload["policy_lab"]["evaluation_readiness"]["ready_for_external_ope"] is False
        assert payload["policy_lab"]["evaluation_readiness"]["policy_value_estimated"] is False
        gate = payload["policy_lab"]["promotion_gate"]
        assert gate["blocking_reason_codes"] == [
            "no_valid_randomized_overlap",
            "insufficient_global_overlap_ess",
        ]
        assert gate["passed_checks"] == 2
        assert gate["total_checks"] == 12
        assert set(gate["checks"]) == {
            "candidate_delta_present",
            "population_integrity_complete",
            "execution_integrity_complete",
            "logging_integrity_complete",
            "minimum_overlap_routes_met",
            "target_probability_floor_met",
            "global_overlap_ess_met",
            "per_action_overlap_met",
            "outcome_evidence_integrity",
            "quality_observation_ready",
            "durable_lifecycle_present",
            "lifecycle_reconciled",
        }
        assert payload["policy_lab"]["evidence_source"] == "durable_sqlite_ledger"
        assert payload["policy_lab"]["durable_ledger"]["counts"] == {
            "started": 3,
            "completed": 2,
            "failed": 1,
            "inflight": 0,
        }


def test_model_store_endpoints():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        store_response = client.get("/api/model_store")
        assert store_response.status_code == 200
        store_payload = store_response.get_json()
        assert store_payload["ok"] is True
        assert store_payload["repo_id"] == "Kai9987kai/supermix-model-zoo"
        assert len(store_payload["models"]) == 2
        assert store_payload["models"][0]["file_name"] == "supermix_omni_collective_v8_preview_20260407_001155.zip"

        jobs_response = client.get("/api/model_store/jobs")
        assert jobs_response.status_code == 200
        jobs_payload = jobs_response.get_json()
        assert jobs_payload["ok"] is True
        assert jobs_payload["jobs"][0]["status"] == "downloading"

        install_response = client.post("/api/model_store/install", json={"file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip"})
        assert install_response.status_code == 200
        install_payload = install_response.get_json()
        assert install_payload["ok"] is True
        assert install_payload["job"]["status"] == "queued"
