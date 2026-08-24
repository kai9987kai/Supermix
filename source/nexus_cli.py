"""NexusMind 2.0 Interactive CLI Terminal.

Command-line interface for the Supermix v78 / NexusMind Omniscience System.
Supports interactive multi-turn REPL, exact math/science derivations,
creative innovation & TRIZ/SCAMPER brainstorming, dynamic persona chat,
5-agent swarm debate transcripts, Graph-of-Thoughts tree search, and live
Dem-Lab telemetry dashboards.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from nexus_engine import NexusConfig, NexusEngine, build_default_engine
import nexus_chat as chat
import nexus_got as got
import nexus_ideation as ideation
import nexus_solver as solver
import nexus_swarm as swarm


def render_banner() -> None:
    print(r"""
===================================================================
  _   _                       __  __ _           _   ____   ___  
 | \ | |                     |  \/  (_)         | | |___ \ / _ \ 
 |  \| | _____  ___   _ ___  | \  / |_ _ __   __| |   __) | | | |
 | . ` |/ _ \ \/ / | | / __| | |\/| | | '_ \ / _` |  / __/| | | |
 | |\  |  __/>  <| |_| \__ \ | |  | | | | | | (_| | |_____| |_| |
 |_| \_|\___/_/\_\\__,_|___/ |_|  |_|_|_| |_|\__,_|        \___/ 
  Supermix v78 Omniscience & Omniverse Hybrid Intelligence Suite
  [Xiaomi MiMo + Supermix + AI-Dem-Lab + Solver + Ideation + Chat]
===================================================================
Commands:
  /solve <math or science query>              : Run exact multi-step SI solver & derivation
  /innovate <topic or problem>                : Run SCAMPER/TRIZ/FNIR creative ideation
  /persona <mentor|catalyst|scientist|empathetic|analyst> : Change active conversational persona
  /chat <message>                             : Chat with active persona and multi-turn memory
  /mode <auto|fast|deep|solve|innovate|chat|swarm|got|scientific> : Change general thinking mode
  /swarm <query>                              : Run 5-agent cognitive swarm debate
  /got <query>                                : Run Graph-of-Thoughts multi-branch tree search
  /science <query>                            : Run closed-world verified physics/chemistry solver
  /telemetry                                  : View live Dem-Lab statistical telemetry
  /help                                       : Show this help menu
  /exit                                       : Quit CLI
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusMind 2.0 Omniscience CLI")
    parser.add_argument("--query", "-q", type=str, help="Single-shot prompt to run")
    parser.add_argument(
        "--mode",
        "-m",
        default="auto",
        choices=["auto", "fast", "deep", "agent", "swarm", "got", "scientific", "solve", "innovate", "chat"],
        help="Thinking mode",
    )
    parser.add_argument(
        "--persona",
        "-p",
        default="empathetic_conversationalist",
        help="Conversational persona",
    )
    args = parser.parse_args()

    engine = build_default_engine()
    active_persona = args.persona

    if args.query:
        print(f"[*] Processing query in mode '{args.mode}': {args.query}\n")
        res = engine.process(query=args.query, mode=args.mode, persona=active_persona)
        print("--- Thinking Chain ---")
        for step in res.thought_steps:
            print(f"[{step.stage.upper()}] {step.content}")
        print("\n--- Final Answer ---")
        print(res.final_output)
        print(f"\n[Confidence: {res.confidence:.2f} | Latency: {res.latency_ms:.1f}ms | Mode: {res.mode_selected}]")
        return

    render_banner()
    current_mode = args.mode
    session_id = "cli_session_001"

    while True:
        try:
            line = input(f"\n[nexus:{current_mode}|{active_persona[:8]}] > ").strip()
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
        elif line.startswith("/persona"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                p_map = {
                    "mentor": "socratic_mentor",
                    "catalyst": "creative_catalyst",
                    "scientist": "rigorous_scientist",
                    "empathetic": "empathetic_conversationalist",
                    "analyst": "executive_analyst",
                }
                req_p = parts[1].strip().lower()
                active_persona = p_map.get(req_p, req_p)
                print(f"[*] Active persona set to: {active_persona}")
            else:
                print(f"[*] Active persona: {active_persona}")
            continue
        elif line.startswith("/mode"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1 and parts[1] in ("auto", "fast", "deep", "agent", "swarm", "got", "scientific", "solve", "innovate", "chat"):
                current_mode = parts[1]
                print(f"[*] Thinking mode set to: {current_mode}")
            else:
                print("[!] Invalid mode. Choose from: auto, fast, deep, agent, swarm, got, scientific, solve, innovate, chat")
            continue
        elif line.startswith("/solve"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "acceleration with force 50 N, mass 10 kg"
            print(f"[*] Executing Exact Solver on: '{q}'...\n")
            res = engine.solver_engine.solve(q)
            if res.solved:
                print(f"Domain: {res.domain.upper()} | Scenario: {res.scenario} | Target: {res.target}")
                print(f"Formula ID: {res.formula_id}\n")
                print("Derivation Steps:")
                for s in res.steps:
                    print(f"  [{s.step_index}] {s.description}")
                    print(f"      Formula:      {s.formula_latex}")
                    print(f"      Substitution: {s.substitution_latex}")
                    print(f"      Result:       {s.evaluated_value} {s.unit}")
                print(f"\nFinal Result: {res.display_answer} {res.unit}")
                if res.receipt:
                    print(f"Receipt SHA-256: {res.receipt.receipt_sha256[:16]}...")
            else:
                print(f"[!] Solver could not match exact deterministic pattern: {res.explanation}")
            continue
        elif line.startswith("/innovate"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "decentralized AI memory routing"
            print(f"[*] Executing Lateral Ideation & SCAMPER on: '{q}'...\n")
            res = engine.ideation_engine.brainstorm(q)
            print("Generated Concepts (Ranked by FNIR Score):")
            for c in res.concepts:
                pareto_flag = " [PARETO OPTIMAL]" if c.is_pareto_optimal else ""
                print(f"  * [{c.operator}] {c.title} (Score: {c.composite_score:.2f}){pareto_flag}")
                print(f"    Benefit: {c.target_benefit}")
                print(f"    F={c.feasibility:.2f} | N={c.novelty:.2f} | I={c.impact:.2f} | R={c.robustness:.2f}\n")
            print("--- Synthesis Proposal ---")
            print(res.synthesis_proposal)
            continue
        elif line.startswith("/chat"):
            parts = line.split(maxsplit=1)
            msg = parts[1] if len(parts) > 1 else "Hello!"
            c_res = engine.chat_engine.chat(session_id, msg, requested_persona=active_persona)
            print(f"\n[{c_res.persona_used.display_name}]:")
            print(c_res.reply)
            continue
        elif line.startswith("/swarm"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Analyze strategic convergence under uncertainty"
            print(f"[*] Launching 5-Agent Cognitive Swarm on: '{q}'...")
            res = engine.swarm_engine.deliberate(query=q)
            for r in res.rounds:
                print(f"\n--- Debate Round {r.round_index} (Consensus: {r.inter_agent_consensus:.2f}) ---")
                for agent_id, c in r.contributions.items():
                    print(f"  [{agent_id.upper()} (w={c.weight:.2f})]: {c.perspective}")
            print("\n--- Swarm Consensus Output ---")
            print(res.consensus_output)
            print(f"\n[Consensus Confidence: {res.final_confidence:.2f} | SHA-256: {res.receipt.receipt_sha256[:16]}...]")
            continue
        elif line.startswith("/got"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Multi-step optimization with constrained latency"
            print(f"[*] Executing Graph-of-Thoughts Search on: '{q}'...")
            res = engine.got_engine.search(query=q)
            print(f"\nNodes Explored: {len(res.all_nodes)} | Pruned: {res.receipt.nodes_pruned} | Merged: {res.receipt.nodes_merged}")
            print("\n--- Optimal Thought Path ---")
            for n in res.best_path_nodes:
                print(f"  -> [{n.branch_type} depth={n.depth} score={n.score:.2f}]: {n.step_text}")
            print("\n--- Final Synthesis Output ---")
            print(res.final_output)
            continue
        elif line.startswith("/science"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "final velocity with initial velocity 5 m/s, acceleration 2 m/s^2, time 3 s"
            print(f"[*] Querying Closed-World Scientific Solver on: '{q}'...")
            res = engine.solver_engine.solve(q)
            if res.solved:
                print(f"Formula: {res.formula_id} | Answer: {res.display_answer} {res.unit}")
            else:
                print(f"Could not verify: {res.explanation}")
            continue
        elif line.startswith("/telemetry"):
            tel = engine.process("ping", mode="fast").telemetry
            print("\n--- Dem-Lab Statistical Telemetry ---")
            for k, v in tel.items():
                print(f"  * {k}: {v}")
            continue

        # Standard Query Execution
        res = engine.process(query=line, mode=current_mode, persona=active_persona, session_id=session_id)
        print(f"\n--- Output [{res.mode_selected.upper()}] ---")
        for step in res.thought_steps:
            print(f"[{step.stage.upper()}] {step.content}")
        print(f"\n{res.final_output}")
        print(f"\n(Confidence: {res.confidence:.2f} | Latency: {res.latency_ms:.1f}ms)")


if __name__ == "__main__":
    main()
