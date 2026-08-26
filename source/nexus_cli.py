"""NexusMind experimental evidence-first CLI.

Only fresh, strictly verified closed-world calculations are rendered as
answers. Persona, ideation, swarm, and graph surfaces are labeled analysis
only; the untrained neural core abstains and exposes architecture telemetry.
"""

from __future__ import annotations

import argparse
import sys
from nexus_engine import NexusResult, build_default_engine


def render_banner() -> None:
    print(r"""
===================================================================
  _   _                       __  __ _           _   ____   ___  
 | \ | |                     |  \/  (_)         | | |___ \ / _ \ 
 |  \| | _____  ___   _ ___  | \  / |_ _ __   __| |   __) | | | |
 | . ` |/ _ \ \/ / | | / __| | |\/| | | '_ \ / _` |  / __/| | | |
 | |\  |  __/>  <| |_| \__ \ | |  | | | | | | (_| | |_____| |_| |
 |_| \_|\___/_/\_\\__,_|___/ |_|  |_|_|_| |_|\__,_|        \___/ 
  Supermix v78 Experimental Evidence-First Interface
  [Exact Verifier + Bounded Heuristics + Neural Telemetry]
===================================================================
Commands:
  /solve <math or science query>              : Run strict verifier-first calculation
  /innovate <topic or problem>                : Run analysis-only concept ideation
  /persona <mentor|catalyst|scientist|empathetic|analyst> : Change active conversational persona
  /chat <message>                             : Chat with active persona and multi-turn memory
  /mode <auto|fast|deep|solve|innovate|chat|swarm|got|scientific> : Change general thinking mode
  /swarm <query>                              : Run analysis-only template debate
  /got <query>                                : Run analysis-only graph scaffold
  /science <query>                            : Run strict allowlisted science verifier
  /telemetry                                  : View diagnostic/synthetic telemetry
  /help                                       : Show this help menu
  /exit                                       : Quit CLI
""")


def render_result(res: NexusResult, *, include_steps: bool = True) -> None:
    """Render answer authority without treating internal scores as confidence."""

    epistemic = res.epistemics or {}
    decision = str(epistemic.get("decision") or "unavailable")
    authority = bool(epistemic.get("answer_authority") is True)
    if decision == "answered" and authority:
        heading = "Verified Closed-World Answer"
    elif decision == "analysis_only":
        heading = "Analysis Only — Not a Verified Answer"
    else:
        heading = "Answer Withheld"

    if include_steps and res.thought_steps:
        print("--- Process Trace ---")
        for step in res.thought_steps:
            print(f"[{step.stage.upper()}] {step.content}")
    print(f"\n--- {heading} ---")
    print(res.final_output)
    confidence_text = (
        f"{res.confidence:.2f} deterministic in-scope"
        if res.confidence is not None
        else "unavailable"
    )
    print(
        f"\n[Decision: {decision} | Correctness confidence: {confidence_text} | "
        f"Answer authority: {str(authority).lower()} | Latency: {res.latency_ms:.1f}ms | "
        f"Mode: {res.mode_selected}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusMind experimental evidence-first CLI")
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
        render_result(res)
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
            print(f"[*] Running strict verifier-first solve on: '{q}'...\n")
            render_result(engine.process(q, mode="solve"))
            continue
        elif line.startswith("/innovate"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "decentralized AI memory routing"
            print(f"[*] Running analysis-only ideation on: '{q}'...\n")
            render_result(engine.process(q, mode="innovate"))
            continue
        elif line.startswith("/chat"):
            parts = line.split(maxsplit=1)
            msg = parts[1] if len(parts) > 1 else "Hello!"
            render_result(
                engine.process(
                    msg,
                    mode="chat",
                    persona=active_persona,
                    session_id=session_id,
                )
            )
            continue
        elif line.startswith("/swarm"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Analyze strategic convergence under uncertainty"
            print(f"[*] Running analysis-only swarm scaffold on: '{q}'...")
            render_result(engine.process(q, mode="swarm"))
            continue
        elif line.startswith("/got"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "Multi-step optimization with constrained latency"
            print(f"[*] Running analysis-only graph scaffold on: '{q}'...")
            render_result(engine.process(q, mode="got"))
            continue
        elif line.startswith("/science"):
            parts = line.split(maxsplit=1)
            q = parts[1] if len(parts) > 1 else "final velocity with initial velocity 5 m/s, acceleration 2 m/s^2, time 3 s"
            print(f"[*] Running strict closed-world science verification on: '{q}'...")
            render_result(engine.process(q, mode="scientific"))
            continue
        elif line.startswith("/telemetry"):
            tel = engine.process("ping", mode="fast").telemetry
            print("\n--- Dem-Lab Statistical Telemetry ---")
            for k, v in tel.items():
                print(f"  * {k}: {v}")
            continue

        # Standard Query Execution
        res = engine.process(query=line, mode=current_mode, persona=active_persona, session_id=session_id)
        render_result(res)


if __name__ == "__main__":
    main()
