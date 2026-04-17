import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch

from device_utils import configure_torch_runtime, resolve_device
from chat_pipeline import (
    MODEL_CLASSES,
    build_context,
    choose_bucket_from_logits,
    cleanup_response_text,
    infer_style_mode,
    pick_response,
    rank_response_candidates,
    resolve_feature_mode,
    text_to_model_input,
)
from llm_database import LLMDatabase
from model_variants import (
    build_model,
    detect_large_head_expansion_dim,
    detect_model_size_from_state_dict,
    detect_xlarge_aux_expansion_dim,
    detect_xxlarge_third_expansion_dim,
    detect_xxxlarge_fourth_expansion_dim,
    detect_ultralarge_fifth_expansion_dim,
    detect_megalarge_sixth_expansion_dim,
    EXPANSION_DIM_MODEL_SIZES,
    EXTRA_EXPANSION_DIM_MODEL_SIZES,
    FIFTH_EXPANSION_DIM_MODEL_SIZES,
    FOURTH_EXPANSION_DIM_MODEL_SIZES,
    load_weights_for_model,
    SIXTH_EXPANSION_DIM_MODEL_SIZES,
    SUPPORTED_MODEL_SIZES,
    THIRD_EXPANSION_DIM_MODEL_SIZES,
)
from chat_memory import ChatMemoryDB, render_memory_block
from run import safe_load_state_dict


# UI color constants
class TerminalColors:
    USER = "\033[94m"     # Blue
    BOT = "\033[92m"      # Green
    SYSTEM = "\033[93m"   # Yellow
    RESET = "\033[0m"


def load_metadata(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{TerminalColors.SYSTEM}Error loading metadata from {path}: {e}{TerminalColors.RESET}")
        return {}


def _int_or_default(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


_FOLLOWUP_QUERY_RE = re.compile(r"\b(it|that|this|they|them|same|previous|above|more|continue|deeper|expand)\b", re.I)
VALID_RUNTIME_MODEL_SIZES = SUPPORTED_MODEL_SIZES


def _build_db_query(user: str, history: List[Tuple[str, str]], memory_rows: List[Dict], max_turns: int = 2) -> str:
    """
    Conversation-aware retrieval query (lightweight QRHEAD-style expansion).
    Uses the raw user query when standalone, and appends recent context for follow-ups.
    """
    user_text = (user or "").strip()
    if not user_text:
        return ""

    use_context = bool(_FOLLOWUP_QUERY_RE.search(user_text)) or len(user_text.split()) <= 4
    if not use_context and history:
        # If the user explicitly asks about a prior topic, keep some context anyway.
        use_context = user_text.endswith("?") and any(k in user_text.lower() for k in ("also", "again", "same", "earlier"))
    if not use_context:
        return user_text

    parts = [user_text]
    for hu, ha in history[-max(0, int(max_turns)):]:
        if hu:
            parts.append(hu)
        if ha:
            parts.append(ha[:220])
    for row in memory_rows[:2]:
        u = str(row.get("user_text", "")).strip()
        a = str(row.get("assistant_text", "")).strip()
        if u:
            parts.append(u)
        if a:
            parts.append(a[:160])
            
    # Deduplicate exact segments while preserving order.
    dedup: List[str] = []
    seen = set()
    for p in parts:
        s = " ".join(p.split())
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    return " | ".join(dedup[:6])


def _resolve_expansion_dim(arg_val: Optional[int], meta: Dict, meta_key: str, default_val: int, 
                           inferred_size: str, allowed_sizes: set, detect_fn, sd: Dict) -> int:
    """Helper to dynamically resolve and infer model dimensionalities, keeping main() DRY."""
    if arg_val is not None:
        return arg_val
    val = _int_or_default(meta.get(meta_key), default_val)
    if inferred_size in allowed_sizes and detect_fn is not None:
        return detect_fn(sd, default=val)
    return val


def resolve_runtime_model_size(requested_model_size: str, meta_model_size: str, inferred_from_weights: str) -> Tuple[str, str]:
    resolved_model_size = str(requested_model_size or "auto").strip().lower() or "auto"
    warning = ""
    if resolved_model_size == "auto":
        meta_model_size = str(meta_model_size or "").strip().lower()
        if meta_model_size in VALID_RUNTIME_MODEL_SIZES and meta_model_size != inferred_from_weights:
            warning = (
                f"{TerminalColors.SYSTEM}Warning: metadata model_size="
                f"{meta_model_size} but weights look like {inferred_from_weights}; using weights.{TerminalColors.RESET}"
            )
        resolved_model_size = inferred_from_weights
    if resolved_model_size not in VALID_RUNTIME_MODEL_SIZES:
        raise RuntimeError(f"Invalid model_size={resolved_model_size!r} (from args/meta).")
    return resolved_model_size, warning


def _default_expansion_dim_for_model_size(model_size: str) -> int:
    if model_size in THIRD_EXPANSION_DIM_MODEL_SIZES:
        return 1024
    if model_size == "xlarge":
        return 768
    return 512


def _default_extra_expansion_dim_for_model_size(model_size: str, expansion_dim: int) -> int:
    base = 2048 if model_size in THIRD_EXPANSION_DIM_MODEL_SIZES else 1024
    return max(base, expansion_dim * 2)


def _format_ms(seconds: float) -> str:
    return f"{max(0.0, float(seconds)) * 1000.0:.1f} ms"


def _format_duration(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _print_chat_help() -> None:
    print(f"{TerminalColors.SYSTEM}Commands:{TerminalColors.RESET}")
    print("  /help               Show this help")
    print("  /stats              Show session stats")
    print("  /clear              Clear in-memory conversation history")
    print("  /style <mode>       Set style: auto|balanced|creative|concise|analyst")
    print("  /creativity <0-1>   Set creative rewrite strength")
    print("  /top <n>            Show top reranked candidates each turn (0 disables)")
    print("  /timing on|off      Toggle per-turn timing output")
    print("  /memory on|off      Toggle memory retrieval/writes for this session")
    print("  /db on|off          Toggle local LLM DB retrieval for this session")
    print("  /config             Print current runtime config")
    print("  /quit               Exit")


def main():
    ap = argparse.ArgumentParser(description="Run an advanced retrieval-style chat app on fine-tuned ChampionNet.")
    ap.add_argument("--weights", default="champion_model_chat_ft.pth")
    ap.add_argument("--meta", default="chat_model_meta.json")
    ap.add_argument(
        "--model_size",
        choices=["auto", *VALID_RUNTIME_MODEL_SIZES],
        default="auto",
    )
    ap.add_argument("--expansion_dim", type=int, default=None)
    ap.add_argument("--extra_expansion_dim", type=int, default=None)
    ap.add_argument("--third_expansion_dim", type=int, default=None)
    ap.add_argument("--fourth_expansion_dim", type=int, default=None)
    ap.add_argument("--fifth_expansion_dim", type=int, default=None)
    ap.add_argument("--sixth_expansion_dim", type=int, default=None)
    ap.add_argument("--adapter_dropout", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--device_preference",
        default="cuda,npu,xpu,dml,mps,cpu",
        help="Priority order used when --device auto.",
    )
    ap.add_argument("--torch_num_threads", type=int, default=0, help="PyTorch intra-op CPU threads (0=auto).")
    ap.add_argument("--torch_interop_threads", type=int, default=0, help="PyTorch inter-op CPU threads (0=auto).")
    ap.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch float32 matmul precision when supported.",
    )
    ap.add_argument("--disable_tf32", action="store_true", help="Disable TF32 on supported CUDA devices.")
    ap.add_argument("--max_turns", type=int, default=2, help="How many previous turns to include in context.")
    ap.add_argument("--top_labels", type=int, default=3, help="How many top predicted buckets to fuse for retrieval.")
    ap.add_argument("--llm_db", default="llm_chat.db", help="Optional local LLM retrieval DB (SQLite).")
    ap.add_argument("--db_top_k", type=int, default=120, help="Top DB candidates to include per turn.")
    ap.add_argument(
        "--db_query_context_turns",
        type=int,
        default=2,
        help="How many recent turns to use for conversation-aware DB query rewriting.",
    )
    ap.add_argument(
        "--db_score_scale",
        type=float,
        default=1.0,
        help="Scale factor for DB candidate confidence contribution.",
    )
    ap.add_argument(
        "--pool_mode",
        choices=["topk", "all"],
        default="all",
        help="Candidate pool source: top-k labels only or all labels weighted by classifier probability.",
    )
    ap.add_argument(
        "--response_temperature",
        type=float,
        default=0.10,
        help="Response sampling temperature over top reranked candidates; 0 disables sampling.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for argmax bucket choice, >0 for sampling from class probabilities.",
    )
    ap.add_argument(
        "--style_mode",
        choices=["auto", "balanced", "creative", "concise", "analyst"],
        default="auto",
        help="Response style mode; auto infers from the user query.",
    )
    ap.add_argument(
        "--creativity",
        type=float,
        default=0.25,
        help="Creative rewrite strength in [0,1] when style is creative.",
    )
    ap.add_argument(
        "--show_top_responses",
        type=int,
        default=0,
        help="Print top reranked response candidates each turn (debug).",
    )
    ap.add_argument(
        "--show_timing",
        action="store_true",
        help="Print per-turn timing breakdown (memory/db/inference/total).",
    )
    ap.add_argument("--memory_db", default="chat_memory.db", help="Persistent memory SQLite DB.")
    ap.add_argument("--memory_top_k", type=int, default=4, help="Number of memory snippets to retrieve each turn.")
    ap.add_argument("--memory_pool_size", type=int, default=400, help="How many recent memory rows to score per turn.")
    ap.add_argument(
        "--memory_recency_half_life_hours",
        type=float,
        default=168.0,
        help="Recency half-life for memory ranking.",
    )
    ap.add_argument(
        "--memory_score_scale",
        type=float,
        default=0.45,
        help="Scale factor for memory-derived candidate confidence.",
    )
    ap.add_argument("--disable_memory", action="store_true", help="Disable persistent memory retrieval and writes.")
    args = ap.parse_args()

    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    meta = load_metadata(args.meta)
    
    raw_feature_mode = str(meta.get("feature_mode", "legacy")).strip().lower()
    feature_mode = resolve_feature_mode(raw_feature_mode, smarter_auto=True)
    feature_mode_note = ""
    if feature_mode != raw_feature_mode:
        feature_mode_note = f" (auto-upgraded from {raw_feature_mode or 'legacy'})"
        
    sd = safe_load_state_dict(args.weights)
    inferred_from_weights = detect_model_size_from_state_dict(sd)
    resolved_model_size, model_size_warning = resolve_runtime_model_size(
        args.model_size,
        str(meta.get("model_size", "")),
        inferred_from_weights,
    )
    if model_size_warning:
        print(model_size_warning)

    # Resolve architectural dimensions via unified helper function
    expansion_dim = _resolve_expansion_dim(
        args.expansion_dim, meta, "expansion_dim",
        _default_expansion_dim_for_model_size(resolved_model_size),
        inferred_from_weights, EXPANSION_DIM_MODEL_SIZES,
        detect_large_head_expansion_dim, sd
    )

    extra_expansion_dim = _resolve_expansion_dim(
        args.extra_expansion_dim, meta, "extra_expansion_dim",
        _default_extra_expansion_dim_for_model_size(resolved_model_size, expansion_dim),
        inferred_from_weights, EXTRA_EXPANSION_DIM_MODEL_SIZES,
        detect_xlarge_aux_expansion_dim, sd
    )

    third_expansion_dim = _resolve_expansion_dim(
        args.third_expansion_dim, meta, "third_expansion_dim",
        max(3072, extra_expansion_dim + expansion_dim),
        inferred_from_weights, THIRD_EXPANSION_DIM_MODEL_SIZES,
        detect_xxlarge_third_expansion_dim, sd
    )

    fourth_expansion_dim = _resolve_expansion_dim(
        args.fourth_expansion_dim, meta, "fourth_expansion_dim",
        max(4096, third_expansion_dim + expansion_dim),
        inferred_from_weights, FOURTH_EXPANSION_DIM_MODEL_SIZES,
        detect_xxxlarge_fourth_expansion_dim, sd
    )

    fifth_expansion_dim = _resolve_expansion_dim(
        args.fifth_expansion_dim, meta, "fifth_expansion_dim",
        max(6144, fourth_expansion_dim + expansion_dim),
        inferred_from_weights, FIFTH_EXPANSION_DIM_MODEL_SIZES,
        detect_ultralarge_fifth_expansion_dim, sd
    )

    sixth_expansion_dim = _resolve_expansion_dim(
        args.sixth_expansion_dim, meta, "sixth_expansion_dim",
        max(8192, fifth_expansion_dim + expansion_dim),
        inferred_from_weights, SIXTH_EXPANSION_DIM_MODEL_SIZES,
        detect_megalarge_sixth_expansion_dim, sd
    )

    adapter_dropout = float(meta.get("adapter_dropout", 0.1)) if args.adapter_dropout is None else args.adapter_dropout

    model = build_model(
        model_size=resolved_model_size,
        expansion_dim=expansion_dim,
        dropout=adapter_dropout,
        extra_expansion_dim=extra_expansion_dim,
        third_expansion_dim=third_expansion_dim,
        fourth_expansion_dim=fourth_expansion_dim,
        fifth_expansion_dim=fifth_expansion_dim,
        sixth_expansion_dim=sixth_expansion_dim,
    ).to(device).eval()
    
    missing, unexpected = load_weights_for_model(model, sd, model_size=resolved_model_size)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

    buckets_raw = meta.get("buckets", {})
    buckets: Dict[int, List[Dict]] = {}
    for k, v in buckets_raw.items():
        try:
            label = int(k)
        except ValueError:
            continue
        if isinstance(v, list) and v:
            buckets[label] = v

    available_labels = sorted(buckets.keys())
    if not available_labels:
        available_labels = list(range(MODEL_CLASSES))
        print(f"{TerminalColors.SYSTEM}Warning: metadata has no buckets; relying on DB/memory retrieval + classifier priors.{TerminalColors.RESET}")
        
    recent_assistant_messages: List[str] = []
    history: List[Tuple[str, str]] = []
    
    llm_db: Optional[LLMDatabase] = None
    if args.llm_db and Path(args.llm_db).exists():
        llm_db = LLMDatabase(str(Path(args.llm_db)))
    elif args.llm_db:
        print(f"{TerminalColors.SYSTEM}Warning: llm_db not found at {args.llm_db}; continuing without DB retrieval.{TerminalColors.RESET}")

    memory_db: Optional[ChatMemoryDB] = None
    if not args.disable_memory and args.memory_db:
        memory_db = ChatMemoryDB(args.memory_db)
    session_memory_enabled = memory_db is not None
    session_db_enabled = llm_db is not None
    show_timing = bool(args.show_timing)
    session_started_at = time.time()
    turn_count = 0
    last_turn_timing: Dict[str, float] = {}

    print(f"\n{TerminalColors.SYSTEM}--- Session Info ---")
    print(f"Loaded: {Path(args.weights).name} [{resolved_model_size}] | Available labels: {len(available_labels)}")
    print(f"Device: {device_info.get('resolved', args.device)} | Threads intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    print(f"Feature mode: {feature_mode}{feature_mode_note} | Style mode: {args.style_mode} (creativity={max(0.0, min(1.0, float(args.creativity))):.2f})")
    
    if llm_db:
        print(f"LLM DB: {args.llm_db} (top_k={args.db_top_k})")
    if memory_db:
        print(f"Memory DB: {args.memory_db} (top_k={max(1, int(args.memory_top_k))}, pool={max(1, int(args.memory_pool_size))})")
    print(f"--------------------{TerminalColors.RESET}")
    print(f"{TerminalColors.BOT}Chat app ready. Type 'exit'/'quit' or use /help for commands.{TerminalColors.RESET}\n")

    # Initialize ThreadPoolExecutor for concurrent DB operations and computational inference
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            try:
                user = input(f"{TerminalColors.USER}You: {TerminalColors.RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{TerminalColors.SYSTEM}Closing chat...{TerminalColors.RESET}")
                break

            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user.startswith("/"):
                cmdline = user[1:].strip()
                parts = cmdline.split()
                cmd = parts[0].lower() if parts else ""
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in {"help", "h", "?"}:
                    _print_chat_help()
                elif cmd == "quit":
                    break
                elif cmd == "clear":
                    history.clear()
                    recent_assistant_messages.clear()
                    print(f"{TerminalColors.SYSTEM}Cleared session conversation history.{TerminalColors.RESET}")
                elif cmd == "stats":
                    uptime = time.time() - session_started_at
                    print(f"{TerminalColors.SYSTEM}Session stats:{TerminalColors.RESET}")
                    print(f"  turns={turn_count} history_turns={len(history)} uptime={_format_duration(uptime)}")
                    print(
                        f"  style={args.style_mode} creativity={float(args.creativity):.2f} "
                        f"top_debug={int(args.show_top_responses)} timing={'on' if show_timing else 'off'}"
                    )
                    print(
                        f"  memory={'on' if session_memory_enabled else 'off'} "
                        f"(db={'ready' if memory_db is not None else 'missing'}) | "
                        f"llm_db={'on' if session_db_enabled else 'off'} "
                        f"(db={'ready' if llm_db is not None else 'missing'})"
                    )
                    if last_turn_timing:
                        print(
                            "  last_timing="
                            + ", ".join(f"{k}={_format_ms(v)}" for k, v in last_turn_timing.items())
                        )
                elif cmd == "config":
                    print(f"{TerminalColors.SYSTEM}Runtime config:{TerminalColors.RESET}")
                    print(
                        f"  style={args.style_mode} creativity={float(args.creativity):.2f} "
                        f"response_temp={float(args.response_temperature):.3f} class_temp={float(args.temperature):.3f}"
                    )
                    print(
                        f"  pool_mode={args.pool_mode} top_labels={int(args.top_labels)} "
                        f"show_top_responses={int(args.show_top_responses)}"
                    )
                    print(
                        f"  memory={'on' if session_memory_enabled else 'off'} top_k={int(args.memory_top_k)} "
                        f"pool={int(args.memory_pool_size)} score_scale={float(args.memory_score_scale):.3f}"
                    )
                    print(
                        f"  llm_db={'on' if session_db_enabled else 'off'} top_k={int(args.db_top_k)} "
                        f"score_scale={float(args.db_score_scale):.3f}"
                    )
                elif cmd == "style":
                    if arg not in {"auto", "balanced", "creative", "concise", "analyst"}:
                        print(f"{TerminalColors.SYSTEM}Usage: /style auto|balanced|creative|concise|analyst{TerminalColors.RESET}")
                    else:
                        args.style_mode = arg
                        print(f"{TerminalColors.SYSTEM}Style mode set to {arg}.{TerminalColors.RESET}")
                elif cmd == "creativity":
                    try:
                        args.creativity = max(0.0, min(1.0, float(arg)))
                        print(f"{TerminalColors.SYSTEM}Creativity set to {float(args.creativity):.2f}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /creativity 0.0-1.0{TerminalColors.RESET}")
                elif cmd == "top":
                    try:
                        args.show_top_responses = max(0, int(arg))
                        print(f"{TerminalColors.SYSTEM}Top-candidate debug set to {int(args.show_top_responses)}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /top <int>{TerminalColors.RESET}")
                elif cmd == "timing":
                    if arg.lower() in {"on", "1", "true"}:
                        show_timing = True
                    elif arg.lower() in {"off", "0", "false"}:
                        show_timing = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /timing on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Per-turn timing {'enabled' if show_timing else 'disabled'}.{TerminalColors.RESET}")
                elif cmd == "memory":
                    if memory_db is None and arg.lower() in {"on", "1", "true"}:
                        print(f"{TerminalColors.SYSTEM}Memory DB is not available in this session.{TerminalColors.RESET}")
                        continue
                    if arg.lower() in {"on", "1", "true"}:
                        session_memory_enabled = True
                    elif arg.lower() in {"off", "0", "false"}:
                        session_memory_enabled = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /memory on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Memory retrieval/writes {'enabled' if session_memory_enabled else 'disabled'} for this session.{TerminalColors.RESET}")
                elif cmd == "db":
                    if llm_db is None and arg.lower() in {"on", "1", "true"}:
                        print(f"{TerminalColors.SYSTEM}LLM DB is not available in this session.{TerminalColors.RESET}")
                        continue
                    if arg.lower() in {"on", "1", "true"}:
                        session_db_enabled = True
                    elif arg.lower() in {"off", "0", "false"}:
                        session_db_enabled = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /db on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}LLM DB retrieval {'enabled' if session_db_enabled else 'disabled'} for this session.{TerminalColors.RESET}")
                else:
                    print(f"{TerminalColors.SYSTEM}Unknown command: /{cmd}. Use /help.{TerminalColors.RESET}")
                continue

            turn_t0 = time.perf_counter()
            t_memory = 0.0
            t_db_wait = 0.0
            t_infer = 0.0
            t_rank = 0.0

            # 1. Evaluate memory queries synchronously (needed for model context & LLM DB query)
            memory_rows: List[Dict] = []
            if session_memory_enabled and memory_db is not None:
                _t = time.perf_counter()
                memory_rows = memory_db.query(
                    user,
                    top_k=max(1, int(args.memory_top_k)),
                    pool_size=max(1, int(args.memory_pool_size)),
                    recency_half_life_hours=max(1.0, float(args.memory_recency_half_life_hours)),
                )
                t_memory += max(0.0, time.perf_counter() - _t)

            # 2. Fire off background Thread for heavy LLM DB querying
            future_llm_db = None
            if session_db_enabled and llm_db is not None:
                db_query = _build_db_query(
                    user=user,
                    history=history,
                    memory_rows=memory_rows,
                    max_turns=max(0, int(args.db_query_context_turns)),
                )
                future_llm_db = executor.submit(llm_db.query, db_query or user, top_k=max(1, args.db_top_k))

            # 3. Proceed with model context building & inference while thread waits on DB IO
            _t = time.perf_counter()
            context = build_context(history, user_text=user, max_turns=args.max_turns)
            if memory_rows:
                memory_block = render_memory_block(memory_rows)
                if memory_block:
                    context = memory_block + "\n" + context
                    
            x = text_to_model_input(context, feature_mode=feature_mode).to(device)
            with torch.no_grad():
                logits = model(x)[0, 0]  # (10,)
            t_infer += max(0.0, time.perf_counter() - _t)

            idx = torch.tensor(available_labels, dtype=torch.long, device=logits.device)
            avail_logits = logits.index_select(0, idx)
            probs = torch.softmax(avail_logits, dim=0)
            
            if args.pool_mode == "all":
                top_pos = list(range(len(available_labels)))
            else:
                k = max(1, min(args.top_labels, len(available_labels)))
                top_pos = torch.topk(avail_logits, k=k).indices.tolist()

            pooled_candidates: List[Dict] = []
            for pos in top_pos:
                label = available_labels[int(pos)]
                bucket_score = float(probs[int(pos)].item())
                for row in buckets.get(label, []):
                    merged = dict(row)
                    merged["bucket_score"] = bucket_score
                    merged["_source"] = "model"
                    pooled_candidates.append(merged)

            # 4. Await LLM DB fetch result and ingest
            if future_llm_db is not None:
                _t = time.perf_counter()
                db_candidates = future_llm_db.result()
                t_db_wait += max(0.0, time.perf_counter() - _t)
                for row in db_candidates:
                    merged = dict(row)
                    merged["bucket_score"] = float(merged.get("bucket_score", 0.0)) * float(args.db_score_scale)
                    merged["_source"] = "llm_db"
                    pooled_candidates.append(merged)

            # Memory candidates help continuity and long-term preference alignment.
            if memory_rows:
                for row in memory_rows:
                    text = str(row.get("assistant_text", "")).strip()
                    vec = row.get("assistant_vec")
                    ctx_vec = row.get("user_vec")
                    if not text or not isinstance(vec, list) or not isinstance(ctx_vec, list):
                        continue
                    pooled_candidates.append(
                        {
                            "text": text,
                            "count": 1,
                            "vec": vec,
                            "ctx_vec": ctx_vec,
                            "bucket_score": float(max(0.0, float(row.get("score", 0.0))) * float(args.memory_score_scale)),
                            "_source": "memory"
                        }
                    )

            # Deduplicate responses and Ensemble Boost Cross-Validated candidates
            dedup: Dict[str, Dict] = {}
            for row in pooled_candidates:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                
                prev = dedup.get(text)
                if prev is None:
                    dedup[text] = row
                    dedup[text]["_sources_set"] = {row.get("_source", "unknown")}
                    continue
                
                # Boost algorithm: 10% bonus if validated by multiple sources
                source = row.get("_source", "unknown")
                base_score = max(float(prev.get("bucket_score", 0.0)), float(row.get("bucket_score", 0.0)))
                
                if source not in prev["_sources_set"]:
                    base_score *= 1.10
                    prev["_sources_set"].add(source)

                prev["bucket_score"] = base_score
                prev["count"] = int(prev.get("count", 1)) + int(row.get("count", 1))
            
            # Clean up temporary boost tracker vars
            for k in dedup:
                dedup[k].pop("_sources_set", None)
                dedup[k].pop("_source", None)

            pooled_candidates = list(dedup.values())

            if (not pooled_candidates) and buckets:
                label = choose_bucket_from_logits(logits, available_labels, temperature=args.temperature)
                pooled_candidates = list(buckets.get(label, []))

            resolved_style = infer_style_mode(user, requested_mode=args.style_mode)

            if args.show_top_responses > 0 and pooled_candidates:
                _t = time.perf_counter()
                ranked, scores = rank_response_candidates(
                    pooled_candidates,
                    query_text=user,
                    recent_assistant_messages=recent_assistant_messages,
                    style_mode=resolved_style,
                )
                t_rank += max(0.0, time.perf_counter() - _t)
                n_show = max(1, min(int(args.show_top_responses), len(ranked)))
                print(f"{TerminalColors.SYSTEM}Top candidates:{TerminalColors.RESET}")
                shown = 0
                for ridx in ranked:
                    cand_text = str(pooled_candidates[ridx].get("text", "")).strip()
                    if not cand_text:
                        continue
                    shown += 1
                    preview = cand_text if len(cand_text) <= 120 else (cand_text[:117] + "...")
                    print(f"  {shown}. ({float(scores[ridx].item()):.3f}) {preview}")
                    if shown >= n_show:
                        break

            _t = time.perf_counter()
            response = pick_response(
                pooled_candidates,
                query_text=user,
                recent_assistant_messages=recent_assistant_messages,
                response_temperature=args.response_temperature,
                style_mode=resolved_style,
                creativity=max(0.0, min(1.0, float(args.creativity))),
            )
            t_rank += max(0.0, time.perf_counter() - _t)
            response = cleanup_response_text(response)
            if not response:
                response = "I do not have a trained response for that yet."

            print(f"{TerminalColors.BOT}Bot: {TerminalColors.RESET}{response}")
            history.append((user, response))
            recent_assistant_messages.append(response)
            turn_count += 1
            
            if session_memory_enabled and memory_db is not None:
                memory_db.add_turn(user, response)

            total_turn = max(0.0, time.perf_counter() - turn_t0)
            last_turn_timing = {
                "memory": t_memory,
                "db_wait": t_db_wait,
                "infer": t_infer,
                "rank_pick": t_rank,
                "total": total_turn,
            }
            if show_timing:
                print(
                    f"{TerminalColors.SYSTEM}Timing:{TerminalColors.RESET} "
                    + ", ".join(f"{k}={_format_ms(v)}" for k, v in last_turn_timing.items())
                )
                
    finally:
        executor.shutdown(wait=False)
        if llm_db is not None:
            llm_db.close()
        if memory_db is not None:
            memory_db.close()

    print(f"{TerminalColors.SYSTEM}Session ended.{TerminalColors.RESET}")


if __name__ == "__main__":
    main()
