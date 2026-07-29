"""Benchmark v52.1 progressive compute against the legacy probe/rerun cost."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

import chat_app


class SyntheticBudgetModel(torch.nn.Module):
    """Deterministic confidence-growth model for controller regression checks."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[int] = []
        self.register_buffer("last_cycles_used", torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        reasoning_cycles: int = 1,
        adaptive_compute: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        del adaptive_compute
        cycles = int(reasoning_cycles or 1)
        self.calls.append(cycles)
        self.last_cycles_used = torch.tensor(float(cycles))
        logits = torch.zeros(x.shape[0], x.shape[1], chat_app.MODEL_CLASSES)
        logits[..., 0] = float(cycles)
        return logits


PROMPTS = [
    "Hello there.",
    "Explain how photosynthesis works.",
    "Solve this problem step by step and test the result.",
    "Compare the options, constraints, benchmarks, and verify the best implementation.",
    "Verify the latest medical evidence and dosage statistics before answering.",
]


def benchmark() -> Dict[str, Any]:
    rows = []
    for prompt in PROMPTS:
        model = SyntheticBudgetModel()
        x = chat_app.text_to_model_input(prompt, feature_mode="legacy")
        interaction_plan = chat_app.plan_interaction(prompt)
        _, _, auto_plan = chat_app.run_metacognitive_auto_compute(
            model,
            x,
            list(range(chat_app.MODEL_CLASSES)),
            interaction_plan=interaction_plan,
        )
        candidate_cycles = list(auto_plan["candidate_cycles"])
        selected_cycles = int(auto_plan["selected_reasoning_cycles"])
        legacy_cycle_cost = sum(candidate_cycles) + selected_cycles
        progressive_cycle_cost = sum(model.calls)
        rows.append(
            {
                "prompt": prompt,
                "difficulty_score": interaction_plan["deliberation"]["difficulty_score"],
                "epistemic_risk": interaction_plan["deliberation"]["epistemic_risk"],
                "minimum_reasoning_cycles": interaction_plan["deliberation"]["minimum_reasoning_cycles"],
                "candidate_cycles": candidate_cycles,
                "executed_cycles": list(model.calls),
                "selected_reasoning_cycles": selected_cycles,
                "stop_reason": auto_plan["reason"],
                "legacy_cycle_cost": legacy_cycle_cost,
                "progressive_cycle_cost": progressive_cycle_cost,
                "avoided_cycle_cost": legacy_cycle_cost - progressive_cycle_cost,
            }
        )

    legacy_total = sum(row["legacy_cycle_cost"] for row in rows)
    progressive_total = sum(row["progressive_cycle_cost"] for row in rows)
    return {
        "benchmark": "v52.1_metacognitive_controller",
        "synthetic": True,
        "note": "Measures controller execution cost, not model quality.",
        "rows": rows,
        "summary": {
            "prompt_count": len(rows),
            "legacy_cycle_cost": legacy_total,
            "progressive_cycle_cost": progressive_total,
            "avoided_cycle_cost": legacy_total - progressive_total,
            "cycle_cost_reduction_percent": round(
                100.0 * (legacy_total - progressive_total) / max(1, legacy_total),
                3,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="output/v52_verification/v521_metacognitive_controller.json",
    )
    args = parser.parse_args()
    result = benchmark()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
