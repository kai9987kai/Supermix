"""NexusMind Interactive CLI Terminal.

Command-line interface for the Supermix v72 / NexusMind Unified Engine.
Supports interactive multi-turn REPL, single-shot evaluation, mode switching,
swarm deliberation transcripts, Graph-of-Thoughts tree rendering, and live
Dem-Lab telemetry dashboards.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from nexus_engine import NexusConfig, NexusEngine, build_default_engine
import nexus_got as got
import nexus_swarm as swarm


def render_banner() -> None:
    print(r"""
===================================================================
  _   _                       __  __ _           _ 
 | \ | |                     |  \/  (_)         | |
 |  \| | _____  ___   _ ___  | \  / |_ _ __   __| |
 | . ` |/ _ \ \/ / | | / __| | |\/| | | '_ \ / _` |
 | |\  |  __/>  <| |_| \__ \ | |  | | | | | | (_| |
 |_| \_|\___/_/\_\\__,_|___/ |_|  |_|_|_| |_|\__,_|
  Supermix v72 Unified Hybrid Thinking System
  [Xiaomi MiMo + Supermix Cognition + AI-Dem-Lab Swarms & GoT]
===================================================================
Commands:
  /mode <auto|fast|deep|swarm|got|scientific> : Change thinking mode
  /swarm <query>                               : Run 5-agent cognitive swarm debate
  /got <query>                                 : Run Graph-of-Thoughts tree search
  /science <query>                             : Run verified deterministic physics/chemistry solver
  /telemetry                                   : View live Dem-Lab statistical telemetry
  /help                                        : Show this help menu
  /exit                                        : Quit CLI
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusMind Unified Thinking CLI")
    parser.add_argument("--query", "-q", type=str, help="Single-shot prompt to run")
    parser.add_argument(
        "--mode",
        "-m",
        default="auto",
        choices=["auto", "fast", "deep", "agent", "swarm", "got", "scientific"],
        help="Thinking mode",
    )
    args = parser.parse_args()

    engine = build_default_engine()

    if args.query:
        print(f"[*] Processing query in mode '{args.mode}': {args.query}\n")
        res = engine.process(query=args.query, mode=args.mode)
        print("--- Thinking Chain ---")
        for step in res.thought_steps:
            print(f"[{step.stage.upper()}] {step.content}")
        print("\n--- Final Answer ---")
        print(res.final_output)
        print(f"\n[Confidence: {res.confidence:.2f} | Latency: {res.latency_ms:.1f}ms | Mode: {res.mode_selected}]")
        return

    render_banner()
    current_mode = args.mode

    while True:
        try:
            line = input(f"\n[nexus:{current_mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting NexusMind CLI.")
            break

        if not line:
            continue

        if line.startswith("/exit") or line.startswith("/quit"):
            print("Goodbye.")
            break
        elif line.startswith("/help"):
            render_banner()
            continue
        elif line.startswith("/mode"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1 and parts[1] in ("auto", "fast", "deep", "agent", "swarm", "got", "scientific"):
                current_mode = parts[1]
                print(f"[*] Thinking mode set to: {current_mode}")
            else:
                print("[!] Invalid mode. Choose from: auto, fast, deep, agent, swarm, got, scientific")
            continue
        elif line.startswith("/swarm"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Analyze strategic convergence under uncertainty"
            print(f"[*] Launching 5-Agent Cognitive Swarm on: '{q}'...")
            res = engine.swarm_engine.deliberate(query=q)
            for r in res.rounds:
                print(f"\n--- Debate Round {r.round_index} (Consensus: {r.inter_agent_consensus:.2f}) ---")
                for a_id, contrib in r.contributions.items():
                    print(f"  [{contrib.role.value.upper()}] (w={contrib.weight:.2f}): {contrib.perspective}")
                    for arg in contrib.arguments:
                        print(f"    + {arg}")
                    for fl in contrib.detected_flaws:
                        print(f"    ! Flaw: {fl}")
            print(f"\n[Consensus Output]: {res.consensus_output}")
            print(f"[Receipt]: {res.receipt.schema_version} | Final Entropy: {res.receipt.consensus_entropy:.3f}")
            continue
        elif line.startswith("/got"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Formulate optimal decision tree"
            print(f"[*] Launching Graph-of-Thoughts Search on: '{q}'...")
            res = engine.got_engine.search(query=q)
            print("\n--- Optimal Thought Path ---")
            for node in res.best_path_nodes:
                print(f"  -> [{node.branch_type} depth={node.depth} score={node.score:.2f}] {node.step_text}")
            print(f"\n[Final Output]: {res.final_output}")
            print(f"[GoT Receipt]: {res.receipt.total_nodes_generated} nodes generated, {res.receipt.nodes_pruned} pruned, {res.receipt.nodes_merged} merged.")
            continue
        elif line.startswith("/telemetry"):
            print("\n--- Dem-Lab Live Statistical Telemetry ---")
            sample_res = engine.process("Telemetry query", mode="fast")
            for k, v in sample_res.telemetry.items():
                print(f"  {k:25s}: {v}")
            continue

        # Standard query execution
        res = engine.process(query=line, mode=current_mode)
        print("\n--- Thinking Steps ---")
        for step in res.thought_steps:
            print(f"[{step.stage.upper()}] {step.content}")
        print("\n--- Result ---")
        print(res.final_output)
        print(f"\n[Confidence: {res.confidence:.2f} | Latency: {res.latency_ms:.1f}ms | Mode: {res.mode_selected}]")


if __name__ == "__main__":
    main()
