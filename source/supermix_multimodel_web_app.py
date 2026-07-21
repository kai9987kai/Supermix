from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from multimodel_catalog import DEFAULT_COMMON_SUMMARY, DEFAULT_MODELS_DIR, discover_model_records
from multimodel_runtime import UnifiedModelManager
from PIL import Image

app = Flask(__name__)
manager: UnifiedModelManager | None = None
MAX_ROUTE_REVIEW_BUNDLE_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ROUTE_REVIEW_BUNDLE_WEB_STRATA = 100

def build_app(unified_manager: UnifiedModelManager) -> Flask:
    global manager
    manager = unified_manager
    return app


def _validate_route_review_request_size(value: Any) -> None:
    try:
        size = len(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("route protocol review request must be finite canonical JSON") from exc
    if size > MAX_ROUTE_REVIEW_BUNDLE_REQUEST_BYTES:
        raise ValueError("route protocol review request exceeds the 2 MiB browser limit")


def _read_strict_route_review_json() -> Any:
    """Parse the integrity-sensitive review surface without last-key-wins JSON."""

    raw = request.get_data(cache=True)
    if not raw:
        raise ValueError("route protocol review request body must contain JSON")
    if len(raw) > MAX_ROUTE_REVIEW_BUNDLE_REQUEST_BYTES:
        raise ValueError("route protocol review request exceeds the 2 MiB browser limit")

    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"route protocol review JSON contains duplicate object key: {key}"
                )
            value[key] = item
        return value

    def reject_non_finite(token):
        raise ValueError(
            f"route protocol review JSON contains non-finite number: {token}"
        )

    try:
        return json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("route protocol review request must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"route protocol review request is not valid JSON: {exc.msg}"
        ) from exc


def _validate_route_review_strata(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("route protocol review requires a non-empty study_plans list")
    if len(value) > MAX_ROUTE_REVIEW_BUNDLE_WEB_STRATA:
        raise ValueError("route protocol review supports at most 100 browser strata")


# ─── Benchmark graph embed helper ───────────────────────────────────────────
def _bench_graph_b64() -> str:
    """Return base64-encoded PNG of the benchmark graph, or empty string."""
    candidates = [
        Path(__file__).parent.parent / "output" / "benchmark_local_all_models_multibench_common_v5_20suite_evo3h_s5_20260506_post.png",
        Path("output") / "benchmark_local_all_models_multibench_common_v5_20suite_evo3h_s5_20260506_post.png",
        Path.home() / "Desktop" / "benchmark_graph_v46_common_v5_20suite.png",
        Path(__file__).parent.parent / "output" / "v48_benchmark_comparison.png",
        Path("output") / "v48_benchmark_comparison.png",
        Path(__file__).parent.parent / "output" / "v47_benchmark_comparison.png",
        Path("output") / "v47_benchmark_comparison.png",
    ]
    output_roots = [Path(__file__).parent.parent / "output", Path("output")]
    for root in output_roots:
        if root.exists():
            candidates.extend(
                sorted(
                    root.glob("benchmark_local_all_models_multibench*.png"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
    seen = set()
    for p in candidates:
        try:
            key = p.resolve()
        except Exception:
            key = p
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.suffix.lower() == ".png":
            return base64.b64encode(p.read_bytes()).decode()
    return ""


def _latest_benchmark_json_path() -> Optional[Path]:
    candidates = [
        Path(__file__).parent.parent / "output" / "benchmark_local_all_models_multibench_common_v5_20suite_evo3h_s5_20260506_post.json",
        Path("output") / "benchmark_local_all_models_multibench_common_v5_20suite_evo3h_s5_20260506_post.json",
        Path(__file__).parent.parent / "output" / "v48_benchmark_results.json",
        Path("output") / "v48_benchmark_results.json",
        Path(__file__).parent.parent / "output" / "v47_benchmark_results.json",
        Path("output") / "v47_benchmark_results.json",
    ]
    output_roots = [Path(__file__).parent.parent / "output", Path("output")]
    for root in output_roots:
        if root.exists():
            candidates.extend(
                sorted(
                    root.glob("benchmark_local_all_models_multibench*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
    seen = set()
    for p in candidates:
        try:
            key = p.resolve()
        except Exception:
            key = p
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.suffix.lower() == ".json":
            return p
    return None


def _benchmark_rows_for_ui(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rows = data.get("rows") or data.get("models") or []
    models: List[Dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        mean = row.get("common_overall_exact")
        if mean is None:
            mean = row.get("mean")
        if mean is None:
            mean = row.get("recipe_eval_accuracy")
        per_bench = row.get("per_benchmark")
        if mean is None and isinstance(per_bench, dict) and per_bench:
            vals = [float(v) for v in per_bench.values() if isinstance(v, (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
        if mean is None:
            continue
        models.append(
            {
                "key": row.get("model_key") or row.get("key") or row.get("label") or "model",
                "label": row.get("label") or row.get("model_key") or row.get("key") or "model",
                "mean": float(mean),
                "benchmark_count": len(per_bench) if isinstance(per_bench, dict) else None,
                "freshness": row.get("benchmark_freshness") or row.get("score_source") or "",
            }
        )
    return models

# ─── HTML / CSS / JS ─────────────────────────────────────────────────────────
_BENCH_B64 = _bench_graph_b64()

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Studio X - V46 20-Suite Champion</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    /* ── Design Tokens ────────────────────────────────────────────── */
    :root {
      --bg:           #020610;
      --surface:      rgba(15, 23, 42, 0.7);
      --surface-hi:   rgba(30, 41, 59, 0.85);
      --border:       rgba(255, 255, 255, 0.08);
      --border-blue:  rgba(56, 189, 248, 0.4);
      --text:         #f1f5f9;
      --muted:        #94a3b8;
      --blue:         #38bdf8;
      --cyan:         #22d3ee;
      --teal:         #2dd4bf;
      --purple:       #818cf8;
      --amber:        #fbbf24;
      --rose:         #fb7185;
      --green:        #34d399;
      --shadow-deep:  0 32px 96px rgba(0,0,0,0.7);
      --shadow-card:  0 12px 40px rgba(0,0,0,0.5);
      --glass:        blur(24px) saturate(180%);
    }

    /* ── Mesh Background ───────────────────────────────────────────── */
    .mesh-bg {
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      z-index: -1; background: var(--bg); overflow: hidden;
    }
    .mesh-bg::after {
      content: ""; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
      opacity: 0.15; pointer-events: none;
      background-image:
        radial-gradient(circle at 20% 30%, #3b82f6 0%, transparent 40%),
        radial-gradient(circle at 80% 20%, #8b5cf6 0%, transparent 40%),
        radial-gradient(circle at 40% 80%, #14b8a6 0%, transparent 40%),
        radial-gradient(circle at 70% 70%, #f59e0b 0%, transparent 40%);
      filter: blur(80px); animation: meshMove 40s ease-in-out infinite alternate;
    }
    @keyframes meshMove {
      from { transform: translate(0, 0) rotate(0deg); }
      to { transform: translate(-5%, -10.5%) rotate(12deg); }
    }

    /* ── Reset & Base ──────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }
    body, html { height:100%; margin:0; padding:0; background:var(--bg); color:var(--text);
                 font-family:'Inter',system-ui,sans-serif; overflow:hidden;
                 -webkit-font-smoothing:antialiased; }
    button { font-family:inherit; cursor:pointer; border:none; background:none; }
    a { color:var(--blue); text-decoration:none; }

    /* ── Shell Layout ───────────────────────────────────────────────── */
    .shell { display:grid; grid-template-columns:80px 1fr 400px; height:100vh;
             position: relative; overflow: hidden; }

    /* ── Navigation Rail ─────────────────────────────────────────── */
    .rail { background:rgba(2,8,16,0.35); border-right:1px solid var(--border);
            display:flex; flex-direction:column; align-items:center;
            padding:24px 0 28px; gap:8px; backdrop-filter: var(--glass); z-index:50; }

    .rail-logo { width:48px; height:48px; border-radius:14px; margin-bottom:16px;
                 background:linear-gradient(135deg,#0ea5e9,#6366f1);
                 display:flex; align-items:center; justify-content:center;
                 box-shadow:0 0 32px rgba(14,165,233,0.4); cursor:pointer;
                 transition:all 0.4s cubic-bezier(0.34,1.56,0.64,1); }
    .rail-logo:hover { transform:scale(1.1) rotate(5deg); filter:brightness(1.1); }
    
    .rail-item { width:52px; height:52px; border-radius:14px;
                 display:flex; align-items:center; justify-content:center;
                 color:var(--muted); cursor:pointer; position:relative;
                 transition:0.2s; }
    .rail-item:hover { background:rgba(255,255,255,0.06); color:var(--text); }
    .rail-item.on { color:var(--blue); background:rgba(56,189,248,0.12);
                    box-shadow:inset 0 0 0 1px rgba(56,189,248,0.3); }
    .rail-item[title]:hover::after { content:attr(title); position:absolute;
      left:64px; top:50%; transform:translateY(-50%);
      background:var(--surface-hi); border:1px solid var(--border);
      border-radius:8px; padding:6px 12px; font-size:12px; font-weight:600;
      white-space:nowrap; color:var(--text); pointer-events:none; z-index:99; }
    .rail-spacer { flex:1; }

    /* ── Workspace (centre) ─────────────────────────────────────────── */
    .workspace { display:grid; grid-template-rows:72px 1fr auto; height: 100vh; min-width:0; position:relative; overflow: hidden; }

    /* header bar */
    .wk-header { display:flex; align-items:center; justify-content:space-between;
                 padding:0 40px; border-bottom:1px solid var(--border);
                 background:rgba(2,8,16,0.25); backdrop-filter: var(--glass); z-index: 10; }
    .wk-title { font-family:'Outfit',sans-serif; font-size:20px; font-weight:800;
                background:linear-gradient(90deg,#fff 20%,#38bdf8);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                letter-spacing:-0.01em; }
    .model-pill { padding:5px 14px; background:rgba(56,189,248,0.1);
                  border:1px solid rgba(56,189,248,0.25); border-radius:100px;
                  font-size:11px; font-weight:800; color:var(--blue);
                  text-transform:uppercase; letter-spacing:0.08em;
                  transition:0.3s; }
    .model-pill.v47 { background:rgba(45,212,191,0.12);
                      border-color:rgba(45,212,191,0.35); color:var(--teal);
                      box-shadow:0 0 15px rgba(45,212,191,0.1); }
    .model-pill.v48 { background:rgba(244,114,182,0.12);
                      border-color:rgba(244,114,182,0.35); color:#f9a8d4;
                      box-shadow:0 0 15px rgba(244,114,182,0.12); }
    .model-pill.v46 { background:rgba(52,211,153,0.13);
                      border-color:rgba(52,211,153,0.45); color:#86efac;
                      box-shadow:0 0 20px rgba(52,211,153,0.13); }
    .panel-toggle { display:none; align-items:center; gap:8px; min-height:38px; padding:8px 12px;
                    border:1px solid rgba(56,189,248,.28); border-radius:12px;
                    background:rgba(56,189,248,.09); color:#bae6fd; font-size:11px;
                    font-weight:900; letter-spacing:.06em; text-transform:uppercase; }
    .panel-toggle:hover { background:rgba(56,189,248,.16); border-color:rgba(56,189,248,.5); }
    .panel-toggle:focus-visible, .panel-close:focus-visible {
      outline:2px solid var(--blue); outline-offset:2px;
    }

    /* ── Thread ─────────────────────────────────────────────────────── */
    .thread { padding:40px 14%; overflow-y:auto; display:flex;
              flex-direction:column; gap:32px; scroll-behavior:smooth;
              min-height: 0; flex: 1; }
    .thread::-webkit-scrollbar { width:5px; }
    .thread::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.08);
                                       border-radius:100px; }

    /* Message rows */
    .msg { display:flex; flex-direction:column; gap:12px; max-width:85%;
           animation:msgIn 0.45s cubic-bezier(0.16,1,0.3,1); }
    @keyframes msgIn { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:none} }
    .msg.user  { align-self:flex-end; }
    .msg.asst  { align-self:flex-start; }

    .msg-meta { display:flex; align-items:center; gap:10px; font-size:11px;
                color:var(--muted); margin-bottom:4px; font-weight:600; }
    .msg.user .msg-meta { justify-content:flex-end; }
    .msg-avatar { width:24px; height:24px; border-radius:8px; font-size:10px;
                  font-weight:900; display:flex; align-items:center;
                  justify-content:center; }
    .msg.asst .msg-avatar { background:linear-gradient(135deg,#0ea5e9,#6366f1); color:#fff; }
    .msg.user .msg-avatar { background:var(--surface-hi); color:var(--muted); }

    .bubble { padding:22px 28px; border-radius:28px; font-size:16px; line-height:1.7;
              background:var(--surface); border:1px solid var(--border);
              box-shadow:var(--shadow-card); white-space:pre-wrap; word-break:break-word;
              backdrop-filter: var(--glass); transition: transform 0.2s; }
    .msg.user .bubble { background:linear-gradient(135deg,rgba(14,165,233,0.16),rgba(14,165,233,0.08));
                        border-color:rgba(14,165,233,0.3); border-bottom-right-radius:8px; }
    .msg.asst .bubble { border-bottom-left-radius:8px; }
    .bubble:hover { transform: translateY(-1px); }
    .mini-copy { margin-left:8px; padding:4px 8px; border-radius:999px;
                 border:1px solid var(--border); color:var(--muted);
                 background:rgba(255,255,255,.04); font-size:10px;
                 font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
    .mini-copy:hover { color:var(--text); border-color:rgba(56,189,248,.35);
                       background:rgba(56,189,248,.08); }

    .champion-card { padding:24px 28px; border-radius:28px;
                     border:1px solid rgba(52,211,153,.28);
                     background:
                       linear-gradient(135deg,rgba(52,211,153,.12),rgba(14,165,233,.08)),
                       rgba(2,8,16,.72);
                     box-shadow:var(--shadow-card); backdrop-filter:var(--glass); }
    .champion-head { display:flex; align-items:center; justify-content:space-between;
                     gap:16px; margin-bottom:18px; }
    .champion-title { font-family:'Outfit',sans-serif; font-size:20px;
                      font-weight:800; letter-spacing:-.02em; }
    .champion-badge { padding:6px 10px; border-radius:999px;
                      background:rgba(52,211,153,.12);
                      border:1px solid rgba(52,211,153,.35);
                      color:#86efac; font-size:10px; font-weight:900;
                      text-transform:uppercase; letter-spacing:.12em; white-space:nowrap; }
    .signal-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
    .signal { border:1px solid var(--border); border-radius:18px;
              background:rgba(0,0,0,.24); padding:14px 16px; }
    .signal strong { display:block; color:var(--text); font-size:18px; margin-bottom:4px; }
    .signal small { color:var(--muted); font-size:11px; line-height:1.35; }

    /* Typing indicator */
    .typing-dots { display:flex; gap:6px; padding:20px 24px; }
    .typing-dots span { width:7px; height:7px; border-radius:50%;
                        background:var(--muted); animation:dot 1.2s infinite ease-in-out; }
    .typing-dots span:nth-child(2) { animation-delay:.2s; }
    .typing-dots span:nth-child(3) { animation-delay:.4s; }
    @keyframes dot { 0%,80%,100%{opacity:.3;transform:scale(0.85)}
                     40%{opacity:1;transform:scale(1.1)} }

    /* Trace cards */
    .trace { margin-top:12px; border:1px solid rgba(255,255,255,0.08);
             background:rgba(0,0,0,0.3); border-radius:18px; overflow:hidden;
             transition: 0.2s; }
    .trace:hover { border-color: rgba(255,255,255,0.15); }
    .trace-hdr { display:flex; align-items:center; gap:10px;
                 padding:12px 20px; font-size:10px; font-weight:900;
                 text-transform:uppercase; letter-spacing:.14em;
                 border-bottom:1px solid rgba(255,255,255,0.05); cursor:pointer; }
    .trace-body { padding:18px 20px; font-size:13px; font-family:'JetBrains Mono',monospace;
                  color:var(--muted); line-height:1.6; }
    .trace-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px 24px;
                  font-size:12.5px; }
    .trace-kv strong { color:var(--text); font-weight: 700; }
    .trace-step { display:flex; gap:14px; margin-bottom:14px; }
    .trace-step-n { width:24px; height:24px; flex-shrink:0; border-radius:50%;
                    border:1.5px solid var(--teal); display:flex;
                    align-items:center; justify-content:center;
                    font-size:10px; font-weight:800; color:var(--teal); }
    .trace-summary { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .trace-pill { display:inline-flex; margin:0 6px 4px 0;
                  border:1px solid rgba(255,255,255,.10); border-radius:999px;
                  padding:5px 9px; color:var(--muted); background:rgba(255,255,255,.03); }
    .trace-score { color:var(--green); border-color:rgba(52,211,153,.28); }
    .route-feedback { margin-top:12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .route-feedback-label { color:var(--muted); font-size:11px; font-weight:900;
                            text-transform:uppercase; letter-spacing:.12em; margin-right:2px; }
    .route-feedback button { border:1px solid var(--border); border-radius:999px;
                             background:rgba(255,255,255,.04); color:var(--muted);
                             padding:6px 10px; font-size:11px; font-weight:900; }
    .route-feedback button:hover { color:var(--text); border-color:rgba(129,140,248,.4);
                                   background:rgba(129,140,248,.09); }
    .route-feedback button:disabled { opacity:.5; cursor:default; }
    .route-health { margin-top:10px; display:flex; align-items:center; gap:7px;
                    flex-wrap:wrap; color:var(--muted); }
    .route-health span { display:inline-flex; border:1px solid rgba(56,189,248,.18);
                         background:rgba(56,189,248,.06); border-radius:999px;
                         padding:6px 9px; font-size:10.5px; font-weight:900; }
    .policy-lab { margin-top:14px; padding:13px; border:1px solid rgba(129,140,248,.2);
                  border-radius:14px; background:linear-gradient(145deg,rgba(129,140,248,.08),rgba(56,189,248,.035)); }
    .policy-lab-head { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .policy-lab-heading { display:flex; align-items:center; flex-wrap:wrap; gap:7px; min-width:0; }
    .policy-lab-title { font-size:11px; font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
    .policy-lab-source { padding:3px 7px; border:1px solid rgba(56,189,248,.3); border-radius:999px;
                         background:rgba(56,189,248,.08); color:#7dd3fc; font-size:9px; font-weight:800;
                         letter-spacing:.04em; text-transform:uppercase; }
    .policy-lab-controls { display:flex; gap:6px; align-items:center; }
    .policy-lab select,.policy-lab button { border:1px solid var(--border); border-radius:8px;
                                            background:rgba(8,12,25,.75); color:var(--text);
                                            padding:5px 7px; font-size:10.5px; font-weight:800; }
    .policy-lab button:disabled { opacity:.55; cursor:wait; }
    .policy-lab-metrics { margin-top:10px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .policy-lab-metric { border:1px solid rgba(255,255,255,.07); border-radius:9px;
                         padding:8px; background:rgba(0,0,0,.16); }
    .policy-lab-metric b { display:block; font-size:13px; color:var(--text); }
    .policy-lab-metric span { display:block; margin-top:2px; color:var(--muted); font-size:9.5px; }
    .policy-lab-gate { margin-top:9px; font-size:10.5px; line-height:1.45; color:#fbbf24; }
    .policy-lab-readiness { margin-top:9px; padding:9px; border:1px solid rgba(255,255,255,.07);
                            border-radius:10px; background:rgba(0,0,0,.14); }
    .policy-lab-readiness-title { margin-bottom:7px; color:var(--muted); font-size:9px;
                                  font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
    .policy-lab-checks { display:grid; grid-template-columns:1fr; gap:4px; }
    .policy-lab-check { display:flex; align-items:flex-start; gap:6px; padding:5px 6px;
                        border-radius:7px; color:var(--muted); font-size:9.5px; line-height:1.35; }
    .policy-lab-check::before { content:'?'; flex:0 0 14px; height:14px; border-radius:50%;
                                display:inline-flex; align-items:center; justify-content:center;
                                background:rgba(148,163,184,.12); color:#cbd5e1; font-size:8px; font-weight:900; }
    .policy-lab-check[data-state="pass"] { background:rgba(52,211,153,.06); color:#a7f3d0; }
    .policy-lab-check[data-state="pass"]::before { content:'\2713'; background:rgba(52,211,153,.18); color:#6ee7b7; }
    .policy-lab-check[data-state="fail"] { background:rgba(251,113,133,.06); color:#fecdd3; }
    .policy-lab-check[data-state="fail"]::before { content:'\00D7'; background:rgba(251,113,133,.17); color:#fda4af; }
    .policy-lab-blockers { margin-top:7px; color:#fcd34d; font-size:9.5px; line-height:1.45; }
    .policy-lab-warning { margin-top:7px; padding:7px 8px; border:1px solid rgba(251,191,36,.32);
                          border-radius:8px; background:rgba(251,191,36,.08); color:#fde68a;
                          font-size:10px; font-weight:700; line-height:1.45; }
    .policy-lab-note { margin-top:6px; font-size:9.5px; line-height:1.45; color:var(--muted); }
    .route-study { margin-top:12px; padding:13px; border:1px solid rgba(45,212,191,.24);
                   border-radius:14px; background:linear-gradient(145deg,rgba(45,212,191,.075),rgba(56,189,248,.025)); }
    .route-study-head { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
    .route-study-heading { min-width:0; }
    .route-study-title { font-size:11px; font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
    .route-study-badge { display:inline-flex; margin-top:5px; padding:3px 7px; border:1px solid rgba(45,212,191,.32);
                         border-radius:999px; background:rgba(45,212,191,.08); color:#99f6e4;
                         font-size:8.5px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
    .route-study button { border:1px solid rgba(45,212,191,.28); border-radius:8px;
                          background:rgba(8,12,25,.75); color:#ccfbf1; padding:6px 8px;
                          font-size:10.5px; font-weight:900; }
    .route-study button:disabled { opacity:.55; cursor:wait; }
    .route-study-controls { margin-top:10px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .route-study-control { display:flex; flex-direction:column; gap:4px; min-width:0; color:var(--muted);
                           font-size:8.5px; font-weight:900; letter-spacing:.06em; text-transform:uppercase; }
    .route-study-control input,.route-study-control select { width:100%; min-width:0; border:1px solid var(--border);
                                                              border-radius:8px; background:rgba(8,12,25,.75);
                                                              color:var(--text); padding:6px 7px; font-size:10.5px; }
    .route-study-status { margin-top:9px; padding:8px; border:1px solid rgba(251,191,36,.28);
                          border-radius:9px; background:rgba(251,191,36,.065); color:#fde68a;
                          font-size:10px; font-weight:800; line-height:1.45; }
    .route-study-metrics { margin-top:8px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .route-study-metric { padding:8px; border:1px solid rgba(255,255,255,.07); border-radius:9px;
                          background:rgba(0,0,0,.15); min-width:0; }
    .route-study-metric b { display:block; color:var(--text); font-size:12.5px; overflow-wrap:anywhere; }
    .route-study-metric span { display:block; margin-top:2px; color:var(--muted); font-size:9px; line-height:1.35; }
    .route-study-dist { margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; }
    .route-study-chip { padding:5px 7px; border:1px solid rgba(56,189,248,.22); border-radius:999px;
                        background:rgba(56,189,248,.06); color:#bae6fd; font-size:9px; font-weight:800; }
    .route-study-chip[data-state="unresolved"] { border-color:rgba(251,113,133,.3); background:rgba(251,113,133,.07); color:#fecdd3; }
    .route-study-chip[data-state="declared_unvalidated"] { border-color:rgba(251,191,36,.3); background:rgba(251,191,36,.07); color:#fde68a; }
    .route-study-chip[data-state="drafted_unsealed"],.route-study-chip[data-state="drafted_unvalidated"] {
      border-color:rgba(167,139,250,.3); background:rgba(167,139,250,.07); color:#ddd6fe;
    }
    .route-study-note { margin-top:7px; color:var(--muted); font-size:9.5px; line-height:1.45; }
    .route-study-campaign { margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,.08); }
    .route-study-campaign-head { display:flex; align-items:center; justify-content:space-between; gap:8px;
                                 color:var(--text); font-size:10px; font-weight:900; }
    .route-study-actions { margin-top:7px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    .route-study-file-label { display:inline-flex; align-items:center; border:1px solid rgba(45,212,191,.28);
                              border-radius:8px; background:rgba(8,12,25,.75); color:#ccfbf1;
                              padding:6px 8px; font-size:10.5px; font-weight:900; cursor:pointer; }
    .route-study-file-label input { position:absolute; width:1px; height:1px; overflow:hidden;
                                    clip:rect(0,0,0,0); white-space:nowrap; }
    .route-study-inventory { margin-top:7px; display:flex; flex-wrap:wrap; gap:5px; }
    .route-study-inventory button { text-align:left; overflow-wrap:anywhere; }

    /* ── Composer ───────────────────────────────────────────────────── */
    .compose-wrap { padding:0 14% 32px; flex-shrink: 0; z-index: 20;
                    background:linear-gradient(transparent,rgba(2,6,16,.9) 50%); }
    .quickbar { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 12px; }
    .quick-chip { padding:9px 13px; border-radius:999px;
                  border:1px solid var(--border); background:rgba(15,23,42,.72);
                  color:var(--muted); font-size:12px; font-weight:800;
                  backdrop-filter:var(--glass); transition:.2s; }
    .quick-chip:hover { color:var(--text); border-color:rgba(56,189,248,.35);
                        background:rgba(56,189,248,.1); transform:translateY(-1px); }
    .compose-box { background:var(--surface-hi); border:1px solid var(--border);
                   border-radius:26px; padding:8px; backdrop-filter: var(--glass);
                   box-shadow:0 48px 112px rgba(0,0,0,.7);
                   transition:border-color .3s, box-shadow .3s, transform .3s; }
    .compose-box:focus-within { border-color:var(--border-blue);
                                transform: translateY(-2px);
                                box-shadow:0 0 0 4px rgba(56,189,248,.1),
                                           0 48px 112px rgba(0,0,0,.7); }
    textarea#prompt { width:100%; background:transparent; border:none; outline:none;
                      color:var(--text); font-family:'Inter',sans-serif;
                      font-size:16px; line-height:1.6; resize:none;
                      padding:18px 24px; min-height:56px; max-height:300px; }
    .compose-bar { display:flex; align-items:center; justify-content:space-between;
                   padding:4px 14px 12px; }
    .compose-tools { display:flex; align-items:center; gap:6px; }
    .ic-btn { width:42px; height:42px; border-radius:12px; display:flex;
              align-items:center; justify-content:center;
              background:transparent; border:none; color:var(--muted);
              transition:.2s; }
    .ic-btn:hover { background:rgba(255,255,255,.08); color:var(--text); }
    .ic-btn.on { color:var(--blue); background:rgba(56,189,248,.1); }

    .send-btn { display:flex; align-items:center; gap:10px; padding:12px 28px;
                background:var(--blue); color:#fff; border:none; border-radius:16px;
                font-weight:800; font-size:14.5px;
                box-shadow:0 10px 24px rgba(14,165,233,0.35); transition:.3s cubic-bezier(0.16,1,0.3,1); }
    .send-btn:hover { transform:scale(1.04) translateY(-1px); 
                      box-shadow:0 18px 36px rgba(14,165,233,0.5); }
    .send-btn:active { transform:scale(0.98); }
    .send-btn:disabled { opacity:.5; pointer-events:none; filter: grayscale(0.5); }

    /* upload preview bar */
    .upload-bar { display:none; align-items:center; gap:12px; padding:10px 16px;
                  background:rgba(0,0,0,.4); border:1px solid var(--border);
                  border-radius:16px; margin:0 14% 16px; font-size:12px; backdrop-filter: var(--glass); }
    .upload-bar img { width:48px; height:48px; border-radius:10px; object-fit:cover; border:1px solid var(--border); }
    .upload-bar .up-name { color:var(--text); flex:1; font-weight: 500; }
    .upload-bar .up-rm { background:rgba(255,255,255,0.06); border:none; color:var(--muted);
                         font-size:14px; width:28px; height:28px; border-radius:50%; transition: .2s; }
    .upload-bar .up-rm:hover { background:rgba(251,113,133,0.2); color:var(--rose); }

    /* ── Control Panel (right sidebar) ──────────────────────────────── */
    .panel { background:rgba(15, 23, 42, 0.4); border-left:1px solid var(--border);
             backdrop-filter: var(--glass); display:flex; flex-direction:column; overflow:hidden; }
    .panel-tabs { display:flex; border-bottom:1px solid var(--border); background: rgba(0,0,0,0.1); }
    .panel-close { display:none; flex:0 0 48px; align-items:center; justify-content:center;
                   color:var(--muted); border-left:1px solid var(--border); font-size:22px; }
    .panel-close:hover { color:var(--text); background:rgba(255,255,255,.06); }
    .panel-backdrop { display:none; position:fixed; inset:0; z-index:70; padding:0;
                      background:rgba(2,6,23,.68); backdrop-filter:blur(4px); opacity:0;
                      pointer-events:none; transition:opacity .25s ease; }
    .ptab { flex:1; padding:18px 8px; font-size:11px; font-weight:800;
            text-align:center; text-transform:uppercase; letter-spacing:.14em;
            color:var(--muted); cursor:pointer; border:none;
            background:transparent; transition:.25s;
            border-bottom:3px solid transparent; }
    .ptab.on { color:var(--blue); border-bottom-color:var(--blue); background: rgba(56,189,248,0.04); }
    .panel-body { flex:1; overflow-y:auto; padding:32px 28px; display:flex;
                  flex-direction:column; gap:36px; }
    .panel-body::-webkit-scrollbar { width:5px; }
    .panel-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,.06); border-radius:100px; }

    .panel-section h4 { font-family:'Outfit',sans-serif; font-size:11px;
                        text-transform:uppercase; letter-spacing:.18em;
                        color:var(--muted); margin:0 0 20px; font-weight:900; }
    .model-snapshot { margin-top:14px; padding:16px; border-radius:18px;
                      border:1px solid rgba(52,211,153,.24);
                      background:rgba(52,211,153,.07); font-size:12px;
                      color:var(--muted); line-height:1.55; }
    .model-snapshot strong { color:var(--text); font-size:13px; }

    /* select / input inputs */
    select, .cfg-input { width:100%; background:rgba(0,0,0,.4); border:1px solid var(--border);
                         border-radius:12px; color:var(--text); padding:10px 16px;
                         font-family:inherit; font-size:14px; transition: .2s; }
    select:focus, .cfg-input:focus { border-color:var(--border-blue); outline:none; background:rgba(0,0,0,.6); }

    /* Mode cards */
    .modes { display:flex; flex-direction:column; gap:12px; }
    .mode-card { padding:20px; border:1px solid var(--border);
                 border-radius:20px; cursor:pointer;
                 background:rgba(255,255,255,.015); transition:all 0.3s cubic-bezier(0.16,1,0.3,1); }
    .mode-card:hover { background:rgba(255,255,255,.04);
                       border-color:rgba(56,189,248,.35); transform: translateX(4px); }
    .mode-card.on { background:rgba(56,189,248,.08);
                    border-color:rgba(56,189,248,.5);
                    box-shadow:0 8px 24px rgba(0,0,0,0.2), inset 0 0 20px rgba(56,189,248,0.03); }
    .mode-card.on[data-mode="auto"] { background:rgba(129,140,248,.08);
                                       border-color:rgba(129,140,248,.5); }
    .mode-card.on[data-mode="collective"] { background:rgba(45,212,191,.08);
                                             border-color:rgba(45,212,191,.5); }
    .mode-card.on[data-mode="loop"] { background:rgba(245,158,11,.08);
                                       border-color:rgba(245,158,11,.5); }
    .mc-title { font-size:14.5px; font-weight:800; margin-bottom:6px;
                display:flex; align-items:center; gap:10px; }
    .mc-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
    .mc-dot.auto { background:#818cf8; box-shadow:0 0 8px #818cf8; }
    .mc-dot.std  { background:var(--green); box-shadow:0 0 8px var(--green); }
    .mc-dot.col  { background:var(--teal);  box-shadow:0 0 8px var(--teal); }
    .mc-dot.loop { background:var(--amber); box-shadow:0 0 8px var(--amber); }
    .mc-desc { font-size:12.5px; color:var(--muted); line-height:1.5; }

    /* setting rows */
    .cfg-row { display:flex; align-items:center; justify-content:space-between;
               margin-bottom:16px; }
    .cfg-row label { font-size:14px; color:var(--text); font-weight: 500; }

    /* Benchmark graph tab */
    .bench-wrap { border-radius:18px; overflow:hidden; background:rgba(0,0,0,0.4);
                  border:1px solid var(--border); box-shadow:0 12px 32px rgba(0,0,0,0.3); }
    .bench-wrap img { width:100%; display:block; filter: saturate(1.1) brightness(1.05); }
    .bench-note { padding:18px 20px; font-size:12px; color:var(--muted);
                  line-height:1.6; text-align: center; }

    /* Status footer */
    .panel-footer { padding:20px 28px; border-top:1px solid var(--border);
                    font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:var(--muted); line-height:1.8; background:rgba(0,0,0,.4); }

    /* Toasts */
    #toasts { position:fixed; bottom:32px; left:50%; transform:translateX(-50%);
              display:flex; flex-direction:column-reverse; gap:10px; z-index:999; }
    .toast { padding:14px 24px; border-radius:16px; font-size:13.5px; font-weight:700;
             background:var(--surface-hi); border:1px solid var(--border);
             box-shadow:var(--shadow-deep); var(--glass);
             animation:toastIn .4s cubic-bezier(0.16,1,.3,1); }
    .toast.ok   { border-color:rgba(52,211,153,.5); color:var(--green); }
    .toast.err  { border-color:rgba(251,113,133,.5); color:var(--rose); }
    @keyframes toastIn { from{opacity:0;transform:translateY(16px) scale(0.95)} to{opacity:1;transform:none} }

    @media (max-width: 1100px) {
      .shell { grid-template-columns:64px 1fr; }
      .panel-toggle, .panel-close { display:inline-flex; }
      .panel-backdrop { display:block; }
      .panel { position:fixed; right:0; top:0; bottom:0; width:min(390px,92vw); z-index:80;
               background:rgba(8,15,30,.97); box-shadow:-24px 0 70px rgba(0,0,0,.48);
               transform:translateX(105%); opacity:0; visibility:hidden; pointer-events:none;
               transition:transform .28s cubic-bezier(.16,1,.3,1), opacity .2s ease, visibility 0s linear .28s; }
      .panel.is-open { transform:none; opacity:1; visibility:visible; pointer-events:auto;
                       transition-delay:0s; }
      .shell.panel-open .panel-backdrop { opacity:1; pointer-events:auto; }
      .thread, .compose-wrap { padding-left:7%; padding-right:7%; }
    }
    @media (max-width: 760px) {
      .shell { grid-template-columns:1fr; }
      .rail { display:none; }
      .wk-header { padding:0 18px; }
      .wk-header > div:first-child { min-width:0; gap:10px !important; }
      .wk-title { font-size:17px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .model-pill { display:none; }
      .panel-toggle span { display:none; }
      .panel-toggle { min-width:40px; padding:8px; justify-content:center; }
      .thread { padding:24px 18px; gap:22px; }
      .compose-wrap { padding:0 18px 18px; }
      .msg { max-width:100%; }
      .signal-grid { grid-template-columns:1fr; }
      .champion-head { align-items:flex-start; flex-direction:column; }
    }
  </style>
</head>
<body>
<div class="mesh-bg"></div>
<div class="shell" id="shell">

  <!-- ── Rail ── -->
  <nav class="rail">
    <div class="rail-logo" title="Supermix Studio X">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="white">
        <path d="M12,2L4.5,20.29L5.21,21L12,18L18.79,21L19.5,20.29L12,2Z"/>
      </svg>
    </div>
    <div class="rail-item on" data-tab="chat" title="Chat Lab">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20,2H4C2.9,2,2,2.9,2,4v18l4-4h14c1.1,0,2-.9,2-2V4C22,2.9,21.1,2,20,2z"/>
      </svg>
    </div>
    <div class="rail-item" data-tab="bench" title="Benchmarks">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19,3H5C3.9,3,3,3.9,3,5v14c0,1.1,0.9,2,2,2h14c1.1,0,2-.9,2-2V5C21,3.9,20.1,3,19,3z M9,17H7v-7h2V17z M13,17h-2V7h2V17z M17,17h-2v-4h2V17z"/>
      </svg>
    </div>
    <div class="rail-item" data-tab="settings" title="Settings">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14,12.94c.04-.3.06-.61.06-.94s-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94L14.4,2.81A.488.488,0,0,0,13.92,2.4H10.08a.488.488,0,0,0-.47.41L9.25,5.35c-.59.24-1.13.56-1.62.94L5.24,5.33c-.22-.08-.47,0-.59.22L2.72,8.87c-.11.2-.06.47.12.61l2.03,1.58c-.05.3-.07.62-.07.94s.02.64.07.94L2.84,14.53c-.18.14-.23.41-.12.61l1.92,3.32c.12.22.37.29.59.22l2.39-.96c.5.38,1.03.7,1.62.94l.36,2.54c.05.24.24.41.48.41h3.84c.24,0,.44-.17.47-.41l.36-2.54c.59-.24,1.13-.56,1.62-.94l2.39.96c.22.08.47,0,.59-.22l1.92-3.32c.12-.22.07-.47-.12-.61ZM12,15.6A3.6,3.6,0,1,1,15.6,12,3.605,3.605,0,0,1,12,15.6Z"/>
      </svg>
    </div>
    <div class="rail-spacer"></div>
    <div class="rail-item" title="Clear session" id="clearBtn">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19V4ZM6,19a2,2,0,0,0,2,2h8a2,2,0,0,0,2-2V7H6Z"/>
      </svg>
    </div>
  </nav>

  <!-- ── Workspace ── -->
  <main class="workspace">
    <header class="wk-header">
      <div style="display:flex;align-items:center;gap:16px">
        <div class="wk-title">Supermix Studio X</div>
        <div class="model-pill v46" id="activePill">V46 Champion</div>
        <div class="model-pill" id="modePill" style="display:none">Standard</div>
      </div>
      <div class="wk-actions" id="wkActions">
        <button type="button" class="panel-toggle" id="panelToggle" aria-controls="controlPanel" aria-expanded="false" aria-label="Open control panel">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M4,5H20V7H4V5ZM4,11H20V13H4V11ZM4,17H20V19H4V17Z"/>
          </svg>
          <span>Controls</span>
        </button>
      </div>
    </header>

    <div class="thread" id="thread">
      <div class="champion-card">
        <div class="champion-head">
          <div>
            <div class="champion-title">Omni Collective V46 Champion is loaded.</div>
            <div style="color:var(--muted);font-size:13px;margin-top:4px">The promoted 20-suite model is selected by default for normal chat, reasoning, and benchmark-backed testing.</div>
          </div>
          <div class="champion-badge">active frontier</div>
        </div>
        <div class="signal-grid">
          <div class="signal"><strong>1.000</strong><small>20-suite exact benchmark score</small></div>
          <div class="signal"><strong>97 / 97</strong><small>latest local benchmark items passed</small></div>
          <div class="signal"><strong>Guarded</strong><small>normal-chat drift repair enabled</small></div>
        </div>
      </div>
    </div>

    <!-- Upload preview bar -->
    <div class="upload-bar" id="uploadBar">
      <img id="imgThumb" src="" alt="">
      <div class="up-name" id="imgName">image.png</div>
      <button class="up-rm" id="clearUpBtn" title="Remove">&#x2715;</button>
    </div>

    <div class="compose-wrap">
      <div class="quickbar" id="quickbar">
        <button class="quick-chip" data-prompt="Hello. Reply like a normal helpful chat model.">Normal hello</button>
        <button class="quick-chip" data-prompt="What model is active, and what benchmark score is it using?">Model status</button>
        <button class="quick-chip" data-prompt="Give a concise step-by-step answer: if a train leaves at 3pm and takes 2 hours 35 minutes, when does it arrive?">Reasoning test</button>
        <button class="quick-chip" data-prompt="Explain the benchmark graph in plain English.">Benchmark summary</button>
      </div>
      <div class="compose-box">
        <textarea id="prompt" rows="1" placeholder="Message Omni V46 Champion..."></textarea>
        <div class="compose-bar">
          <div class="compose-tools">
            <button class="ic-btn" title="Attach image" id="imgBtn">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                <path d="M21,19V5a2,2,0,0,0-2-2H5A2,2,0,0,0,3,5V19a2,2,0,0,0,2,2H19A2,2,0,0,0,21,19ZM8.5,13.5l2.5,3L14.5,12l4.5,6H5Z"/>
              </svg>
            </button>
            <button class="ic-btn" title="Web search" id="webBtn">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                <path d="M15.5,14h-.79l-.28-.27A6.471,6.471,0,0,0,16,9.5,6.5,6.5,0,1,0,9.5,16a6.471,6.471,0,0,0,4.23-1.57l.27.28v.79l5,4.99L20.49,19Zm-6,0a4.5,4.5,0,1,1,4.5-4.5A4.494,4.494,0,0,1,9.5,14Z"/>
              </svg>
            </button>
            <button class="ic-btn" title="Preview route plan" id="routePlanBtn">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9,3h6l1.5,3H21v6H3V6H7.5L9,3Zm1.24,2L9.22,7H5v3H19V8h-3.72L14.26,5H10.24ZM5,14h14v2H5V14Zm0,4h9v2H5V18Z"/>
              </svg>
            </button>
          </div>
          <button class="send-btn" id="sendBtn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01,21L23,12,2.01,3,2,10l15,2L2,14Z"/>
            </svg>
            Send
          </button>
        </div>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none">
    </div>
  </main>

  <!-- ── Right Panel ── -->
  <aside class="panel" id="controlPanel" aria-label="Model, mode, and benchmark controls">
    <div class="panel-tabs">
      <button class="ptab on" data-ptab="model">Model</button>
      <button class="ptab" data-ptab="mode">Mode</button>
      <button class="ptab" data-ptab="bench">Bench</button>
      <button type="button" class="panel-close" id="panelClose" aria-label="Close control panel">&times;</button>
    </div>

    <!-- MODEL tab -->
    <div class="panel-body" id="ptab-model">
      <div class="panel-section">
        <h4>Active Model</h4>
        <select id="modelSelect"></select>
        <div class="model-snapshot" id="modelSnapshot">Loading champion manifest...</div>
      </div>
      <div class="panel-section">
        <h4>V46 Champion System</h4>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div style="display:flex;align-items:center;gap:12px;font-size:13px">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--teal);box-shadow:0 0 10px var(--teal)"></div>
            Graph-of-Thoughts synthesis
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:13px">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--purple);box-shadow:0 0 10px var(--purple)"></div>
            Mixture-of-Depths routing
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:13px">
            <div style="width:10px;height:10px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)"></div>
            Continuous Latent C-CoT
          </div>
        </div>
      </div>
      <div class="panel-section">
        <h4>Inference Settings</h4>
        <div class="cfg-row">
          <label>Loop budget</label>
          <input class="cfg-input" type="number" id="loopBudget" value="4" min="2" max="16" style="width:70px">
        </div>
        <div class="cfg-row">
          <label>Auto budget</label>
          <select class="cfg-input" id="autoBudget" style="width:116px">
            <option value="fast">Fast</option>
            <option value="balanced" selected>Balanced</option>
            <option value="deep">Deep</option>
            <option value="max">Max</option>
          </select>
        </div>
        <div class="cfg-row">
          <label>Reasoning cycles</label>
          <select class="cfg-input" id="reasoningCycles" style="width:116px" title="Model uses checkpoint metadata; Prompt auto estimates a budget from the request">
            <option value="model" selected>Model / Route</option>
            <option value="auto">Prompt auto</option>
            <option value="1">1 cycle</option>
            <option value="3">3 cycles</option>
            <option value="8">8 cycles</option>
            <option value="16">16 cycles</option>
          </select>
        </div>
        <div class="cfg-row">
          <label>Adaptive compute</label>
          <select class="cfg-input" id="adaptiveCompute" style="width:116px" title="Let supported checkpoints stop early when predictions stabilize">
            <option value="model" selected>Model / Route</option>
            <option value="on">Enabled</option>
            <option value="off">Disabled</option>
          </select>
        </div>
        <div class="cfg-row">
          <label>Session budget</label>
          <input class="cfg-input" type="number" id="sessionBudget" value="0" min="0" max="100000" step="0.5" style="width:90px" title="0 disables session cost pacing">
        </div>
        <div class="cfg-row">
          <label>Budget horizon</label>
          <input class="cfg-input" type="number" id="sessionBudgetTargetRoutes" value="0" min="0" max="10000" step="1" style="width:90px" title="0 uses only the remaining session budget">
        </div>
        <div class="cfg-row">
          <label>Neural Memory</label>
          <select class="cfg-input" id="memToggle" style="width:90px">
            <option value="on">Enabled</option>
            <option value="off">Disabled</option>
          </select>
        </div>
        <div class="cfg-row">
          <label>Web Access</label>
          <select class="cfg-input" id="webToggle" style="width:90px">
            <option value="off">Local Only</option>
            <option value="on">Hybrid Search</option>
          </select>
        </div>
      </div>
    </div>

    <!-- MODE tab -->
    <div class="panel-body" id="ptab-mode" style="display:none">
      <div class="panel-section">
        <h4>Operational Mode</h4>
        <div class="modes">
          <div class="mode-card on" data-mode="auto">
            <div class="mc-title"><div class="mc-dot auto"></div>Adaptive Router</div>
            <div class="mc-desc">Prompt-aware orchestration. Chooses standard, collective, or loop depth from task complexity and budget.</div>
          </div>
          <div class="mode-card" data-mode="off">
            <div class="mc-title"><div class="mc-dot std"></div>Standard Case</div>
            <div class="mc-desc">Optimal for direct queries and creative generation. High-speed single-pass.</div>
          </div>
          <div class="mode-card" data-mode="collective">
            <div class="mc-title"><div class="mc-dot col"></div>Collective Synthesis</div>
            <div class="mc-desc">Ensemble reasoning. V46 consults sub-experts before delivering a unified response.</div>
          </div>
          <div class="mode-card" data-mode="loop">
            <div class="mc-title"><div class="mc-dot loop"></div>Autonomous Frontier</div>
            <div class="mc-desc">Recursive loop for complex workflows. Self-correcting multi-step planner.</div>
          </div>
        </div>
      </div>
      <div class="panel-section" id="loopPanel" style="display:block">
        <h4>Loop Observation</h4>
        <div class="loop-steps" id="loopSteps"></div>
        <div class="route-feedback" id="routeFeedback" style="display:none">
          <div class="route-feedback-label" id="routeFeedbackLabel">Route</div>
          <button type="button" id="routeGoodBtn" title="Prefer this Auto route next time">Good route</button>
          <button type="button" id="routeBadBtn" title="The route produced a poor answer">Bad quality</button>
          <button type="button" id="routeDeeperBtn" title="Use a deeper Auto route for similar prompts">Needs deeper</button>
          <button type="button" id="routeCostBtn" title="Prefer a lower-cost Auto route for similar prompts">Too costly</button>
          <button type="button" id="routeSlowBtn" title="Prefer a lower-latency Auto route for similar prompts">Too slow</button>
        </div>
        <div class="route-health" id="routeHealth" style="display:none">
          <span id="routeHealthCount">Routes 0</span>
          <span id="routeHealthQuality">Quality -</span>
          <span id="routeHealthConfidence">Recent evidence -</span>
          <span id="routeHealthPreference">Preference -</span>
          <span id="routeHealthCost">Avg cost -</span>
          <span id="routeHealthLatency">Avg ms -</span>
        </div>
        <div class="policy-lab" id="policyLab">
          <div class="policy-lab-head">
            <div class="policy-lab-title">Route Policy Lab</div>
            <div class="policy-lab-controls">
              <select id="policyLabProfile" title="Shadow threshold profile">
                <option value="balanced">Balanced</option>
                <option value="efficiency">Efficiency</option>
                <option value="quality_first">Quality first</option>
              </select>
              <button type="button" id="policyLabRefresh">Replay</button>
            </div>
          </div>
          <div class="policy-lab-metrics">
            <div class="policy-lab-metric"><b id="policyLabJoined">0 / 0</b><span>exact usage-feedback joins</span></div>
            <div class="policy-lab-metric"><b id="policyLabAgreement">-</b><span>candidate action agreement</span></div>
            <div class="policy-lab-metric"><b id="policyLabApproval">-</b><span>matched observed approval</span></div>
            <div class="policy-lab-metric"><b id="policyLabEconomics">-</b><span>matched cost / latency</span></div>
            <div class="policy-lab-metric"><b id="policyLabLifecycle">0 / 0 / 0</b><span>completed / failed / in flight</span></div>
            <div class="policy-lab-metric"><b id="policyLabFeedbackCoverage">-</b><span>terminal feedback coverage</span></div>
            <div class="policy-lab-metric"><b id="policyLabOverlapEss">ESS 0.0 / 20</b><span>target-policy overlap</span></div>
            <div class="policy-lab-metric"><b id="policyLabWeakestAction">-</b><span>weakest target action</span></div>
            <div class="policy-lab-metric"><b id="policyLabReadinessChecks">0 / 12</b><span>readiness checks passed</span></div>
            <div class="policy-lab-metric"><b id="policyLabOutcomeCoverage">0 / 0</b><span>quality outcomes observed</span></div>
            <div class="policy-lab-metric"><b id="policyLabContractCoverage">0 / 0</b><span>routes with precommitted outcome set</span></div>
            <div class="policy-lab-metric"><b id="policyLabEvidenceSource">waiting</b><span>evidence source</span></div>
          </div>
          <div class="policy-lab-gate" id="policyLabGate">Shadow only - waiting for joined evidence.</div>
          <div class="policy-lab-readiness">
            <div class="policy-lab-readiness-title">Readiness matrix</div>
            <div class="policy-lab-checks" id="policyLabChecks" aria-label="Readiness checks"></div>
            <div class="policy-lab-blockers" id="policyLabBlockers">Waiting for durable evidence.</div>
          </div>
          <div class="policy-lab-note" id="policyLabNote">Associational replay only. Changed actions receive no imputed reward.</div>
        </div>
        <div class="route-study" id="routeStudy">
          <div class="route-study-head">
            <div class="route-study-heading">
              <div class="route-study-title">Bounded Exposure Rehearsal</div>
              <div class="route-study-badge">Rehearsal only - execution off</div>
            </div>
            <button type="button" id="routeStudyPreview">Rehearse</button>
          </div>
          <div class="route-study-controls">
            <label class="route-study-control" for="routeStudyHorizon">Route horizon
              <input id="routeStudyHorizon" type="number" value="2000" min="20" max="100000" step="20">
            </label>
            <label class="route-study-control" for="routeStudyEpsilon">Alternate mass
              <select id="routeStudyEpsilon">
                <option value="0.10" selected>10%</option>
                <option value="0.15">15%</option>
                <option value="0.20">20%</option>
              </select>
            </label>
            <label class="route-study-control" for="routeStudyResponseRate">Rating response
              <select id="routeStudyResponseRate">
                <option value="0.10">10% scenario</option>
                <option value="0.30" selected>30% scenario</option>
                <option value="0.50">50% scenario</option>
              </select>
            </label>
            <label class="route-study-control" for="routeStudyTargetLabels">Target labels
              <input id="routeStudyTargetLabels" type="number" value="20" min="1" max="1000" step="1">
            </label>
            <label class="route-study-control" for="routeProtocolTarget">Target policy class
              <select id="routeProtocolTarget">
                <option value="efficiency">Efficiency</option>
                <option value="balanced" selected>Balanced</option>
                <option value="quality_first">Quality first</option>
              </select>
            </label>
            <label class="route-study-control" for="routeProtocolDesign">Stateful design
              <select id="routeProtocolDesign">
                <option value="sticky_session_cluster" selected>Sticky session cluster</option>
                <option value="clustered_switchback">Clustered switchback</option>
              </select>
            </label>
            <label class="route-study-control" for="routeProtocolCarryover">Carryover declaration
              <select id="routeProtocolCarryover">
                <option value="unknown" selected>Unknown</option>
                <option value="none_declared">None declared</option>
                <option value="within_session">Within session</option>
                <option value="cross_session">Cross session</option>
              </select>
            </label>
            <label class="route-study-control" for="routeProtocolInterference">Interference declaration
              <select id="routeProtocolInterference">
                <option value="unknown" selected>Unknown</option>
                <option value="none_declared">None declared</option>
                <option value="shared_resource">Shared resource</option>
                <option value="cross_cluster">Cross cluster</option>
              </select>
            </label>
            <label class="route-study-control" for="routeProtocolTemporal">Temporal variation
              <select id="routeProtocolTemporal">
                <option value="unknown" selected>Unknown</option>
                <option value="stable_declared">Stable declared</option>
                <option value="nonstationary">Nonstationary</option>
              </select>
            </label>
            <label class="route-study-control" for="routeProtocolClusters">Cluster ceiling
              <input id="routeProtocolClusters" type="number" value="200" min="2" max="1000000" step="10">
            </label>
            <label class="route-study-control" for="routeProtocolBlock">Switchback block routes
              <input id="routeProtocolBlock" type="number" value="20" min="2" max="10000" step="1">
            </label>
            <label class="route-study-control" for="routeProtocolWashout">Switchback washout routes
              <input id="routeProtocolWashout" type="number" value="0" min="0" max="9999" step="1">
            </label>
          </div>
          <div class="route-study-status" id="routeStudyStatus" role="status" aria-live="polite">
            Preview a prompt to rehearse exact post-filter propensities. No route will run and no evidence will be written.
          </div>
          <div class="route-study-metrics">
            <div class="route-study-metric"><b id="routeStudyBaseline">-</b><span>incumbent route</span></div>
            <div class="route-study-metric"><b id="routeStudyFloor">-</b><span>minimum alternate propensity</span></div>
            <div class="route-study-metric"><b id="routeStudyTraffic">-</b><span>routes for target on every alternate</span></div>
            <div class="route-study-metric"><b id="routeStudyCost">-</b><span>same-stratum expected cost units</span></div>
            <div class="route-study-metric"><b id="routeStudyLatency">-</b><span>latency-tier envelope</span></div>
            <div class="route-study-metric"><b id="routeStudyCharter">-</b><span>draft charter fingerprint</span></div>
            <div class="route-study-metric"><b id="routeProtocolMode">-</b><span>stateful design screen</span></div>
            <div class="route-study-metric"><b id="routeProtocolPolicy">-</b><span>frozen target-policy draft</span></div>
            <div class="route-study-metric"><b id="routeProtocolReview">-</b><span>independent-review state</span></div>
            <div class="route-study-metric"><b id="routeProtocolHash">-</b><span>protocol draft fingerprint</span></div>
          </div>
          <div class="route-study-dist" id="routeStudyDistribution" aria-label="Rehearsed route probabilities"></div>
          <div class="route-study-dist" id="routeProtocolBlockers" aria-label="Protocol activation blocker register"></div>
          <div class="route-study-campaign">
            <div class="route-study-campaign-head">
              <span>Multi-stratum semantic review</span>
              <span id="routeBundleCount">0 strata</span>
            </div>
            <div class="route-study-actions">
              <button type="button" id="routeBundleAdd" disabled>Add current stratum</button>
              <button type="button" id="routeBundleBuild" disabled>Build review bundle</button>
              <button type="button" id="routeBundleDownload" disabled>Download bundle</button>
              <label class="route-study-file-label" for="routeBundleImport">Import and verify
                <input id="routeBundleImport" type="file" accept="application/json,.json">
              </label>
              <button type="button" id="routeBundleClear" disabled>Clear inventory</button>
            </div>
            <div class="route-study-inventory" id="routeBundleInventory" aria-label="Prompt-free support stratum inventory"></div>
            <div class="route-study-metrics">
              <div class="route-study-metric"><b id="routeBundleVerification">not built</b><span>semantic verification</span></div>
              <div class="route-study-metric"><b id="routeBundleHash">-</b><span>review bundle fingerprint</span></div>
            </div>
            <div class="route-study-note" id="routeBundleStatus">
              Inventory only. Strata are never pooled, weighted, assigned, executed, or promoted.
            </div>
          </div>
          <div class="route-study-campaign" id="routeShadowRegistry">
            <div class="route-study-campaign-head">
              <span>Shadow assignment registry - read only</span>
              <button type="button" id="routeShadowRegistryRefresh">Refresh</button>
            </div>
            <div class="route-study-metrics">
              <div class="route-study-metric"><b id="routeShadowRegistryCount">0</b><span>sealed campaigns</span></div>
              <div class="route-study-metric"><b id="routeShadowRegistryChain">not loaded</b><span>append-only event chain</span></div>
              <div class="route-study-metric"><b id="routeShadowRegistryAssignments">0 / 0</b><span>verified / committed assignments</span></div>
              <div class="route-study-metric"><b id="routeShadowRegistryState">not initialized</b><span>campaign states</span></div>
            </div>
            <div class="route-study-inventory" id="routeShadowRegistryCampaigns" aria-label="Read-only shadow campaign status"></div>
            <div class="route-study-note" id="routeShadowRegistryStatus" role="status" aria-live="polite">
              Refresh to inspect the isolated local registry. This browser cannot seal, assign, reveal, activate, or promote a route policy.
            </div>
          </div>
          <div class="route-study-note" id="routeStudyNote">
            Same-support rehearsal only. ESS, policy value, live assignment, and promotion remain unavailable.
          </div>
        </div>
      </div>
    </div>

    <!-- BENCH tab -->
    <div class="panel-body" id="ptab-bench" style="display:none">
      <div class="panel-section">
        <h4>V46 20-Suite Benchmarks</h4>
        <div class="bench-wrap" id="benchWrap">
          <img id="benchImg" src="" alt="Benchmark comparison" style="display:none">
          <div class="bench-note" id="benchNote">Initializing frontier telemetry...</div>
        </div>
        <div style="margin-top:24px;display:flex;flex-direction:column;gap:12px" id="benchScores"></div>
      </div>
    </div>

    <div class="panel-footer" id="panelStatus">system: active  |  accelerator: auto  |  v: 46.20</div>
  </aside>
  <div class="panel-backdrop" id="panelBackdrop" aria-hidden="true"></div>
</div>

<div id="toasts"></div>

<script>
(function() {
  'use strict';

  // ── Helpers ──────────────────────────────────────────────────────────
  const el   = id => document.getElementById(id);
  const qs   = (sel, root=document) => root.querySelector(sel);
  const qsa  = (sel, root=document) => [...root.querySelectorAll(sel)];
  const sessionId = ([1e7]+-1e3+-4e3+-8e2+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c/4).toString(16));

  let agentMode   = 'auto';
  let currentUpload = null;
  let currentUpUrl  = '';
  let loopStep = 0;
  let catalogByKey = {};
  let lastRouteFeedback = null;
  let latestRouteStudy = null;
  let routeStudyStrata = [];
  let latestRouteReviewBundle = null;

  async function api(path, body=null) {
    const opts = body
      ? { method:'POST', body:JSON.stringify(body),
          headers:{'Content-Type':'application/json'} }
      : {};
    const r = await fetch(path, opts);
    if (!r.ok) {
      let message = await r.text();
      try {
        const parsed = JSON.parse(message);
        if (parsed && parsed.error) message = parsed.error;
      } catch (_) {}
      throw new Error(message);
    }
    return r.json();
  }

  function toast(type, msg) {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    el('toasts').prepend(t);
    setTimeout(() => t.remove(), 4000);
  }

  // Smart Scrolling Logic
  function scrollToBottom(force = false) {
    const thread = el('thread');
    const threshold = 120; // px from bottom
    const isAtBottom = thread.scrollHeight - thread.scrollTop <= thread.clientHeight + threshold;
    if (force || isAtBottom) {
      thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' });
    }
  }

  const compactPanelMedia = window.matchMedia('(max-width: 1100px)');

  function setPanelOpen(open, options = {}) {
    const panel = el('controlPanel');
    const toggle = el('panelToggle');
    const compact = compactPanelMedia.matches;
    const nextOpen = compact && Boolean(open);
    el('shell').classList.toggle('panel-open', nextOpen);
    panel.classList.toggle('is-open', nextOpen);
    panel.setAttribute('aria-hidden', compact && !nextOpen ? 'true' : 'false');
    panel.inert = compact && !nextOpen;
    toggle.setAttribute('aria-expanded', String(nextOpen));
    toggle.setAttribute('aria-label', nextOpen ? 'Close control panel' : 'Open control panel');
    if (nextOpen && options.focus !== false) {
      const activeTab = panel.querySelector('.ptab.on') || panel.querySelector('.ptab');
      if (activeTab) activeTab.focus();
    } else if (!nextOpen && options.restoreFocus) {
      toggle.focus();
    }
  }

  function openPanelTab(name) {
    switchPtab(name);
    setPanelOpen(true);
  }

  // ── Tabs (rail) ─────────────────────────────────────────────────────
  qsa('.rail-item[data-tab]').forEach(btn => {
    btn.onclick = () => {
      qsa('.rail-item[data-tab]').forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      const ptab = btn.dataset.tab === 'bench' ? 'bench'
                 : btn.dataset.tab === 'settings' ? 'model'
                 : 'model';
      if (btn.dataset.tab === 'bench' || btn.dataset.tab === 'settings') openPanelTab(ptab);
      else {
        switchPtab(ptab);
        setPanelOpen(false);
      }
    };
  });

  // ── Panel tabs ───────────────────────────────────────────────────────
  qsa('.ptab').forEach(btn => {
    btn.onclick = () => switchPtab(btn.dataset.ptab);
  });

  el('panelToggle').onclick = () => setPanelOpen(!el('controlPanel').classList.contains('is-open'));
  el('panelClose').onclick = () => setPanelOpen(false, { restoreFocus: true });
  el('panelBackdrop').onclick = () => setPanelOpen(false, { restoreFocus: true });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && el('controlPanel').classList.contains('is-open')) {
      setPanelOpen(false, { restoreFocus: true });
    }
  });
  compactPanelMedia.addEventListener('change', () => setPanelOpen(false, { focus: false }));
  setPanelOpen(false, { focus: false });

  function switchPtab(name) {
    qsa('.ptab').forEach(b => b.classList.toggle('on', b.dataset.ptab === name));
    ['model','mode','bench'].forEach(t =>
      el(`ptab-${t}`).style.display = t===name ? 'flex' : 'none');
    if (name === 'bench') loadBenchData();
  }

  // ── Mode cards ───────────────────────────────────────────────────────
  qsa('.mode-card').forEach(c => {
    c.onclick = () => {
      qsa('.mode-card').forEach(x => x.classList.remove('on'));
      c.classList.add('on');
      agentMode = c.dataset.mode;
      el('loopPanel').style.display = ['auto','loop','collective_loop'].includes(agentMode) ? 'block' : 'none';
      el('loopSteps').innerHTML = '';
      loopStep = 0;
      updateModePill();
      toast('ok', `Switched to ${agentMode} mode`);
    };
  });

  function updateModePill() {
    const pill = el('modePill');
    const labels = { auto:'Auto', off:'Standard', collective:'Collective', loop:'Autonomous', collective_loop:'Collective Loop' };
    if (agentMode === 'off') { pill.style.display='none'; return; }
    pill.style.display='block';
    pill.textContent = labels[agentMode] || agentMode;
    const isCollective = agentMode==='collective' || agentMode==='collective_loop';
    const isAuto = agentMode==='auto';
    pill.style.color = isAuto ? '#818cf8' : isCollective ? 'var(--teal)' : 'var(--amber)';
    pill.style.borderColor = isAuto ? 'rgba(129,140,248,.4)' : isCollective
      ? 'rgba(45,212,191,.4)' : 'rgba(245,158,11,.4)';
    pill.style.background = isAuto ? 'rgba(129,140,248,.08)' : isCollective
      ? 'rgba(45,212,191,.08)' : 'rgba(245,158,11,.08)';
  }

  // ── Thread ───────────────────────────────────────────────────────────
  function addMsg(role, text, trace=null, extra='') {
    const row = document.createElement('div');
    row.className = `msg ${role}`;

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.textContent = role==='user' ? 'U' : 'SX';
    meta.append(av, document.createTextNode(role==='user' ? 'You' : 'Omni V46 Champion'));
    if (role !== 'user') {
      const copy = document.createElement('button');
      copy.className = 'mini-copy';
      copy.type = 'button';
      copy.textContent = 'Copy';
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(String(text || ''));
          toast('ok', 'Copied response');
        } catch (_) {
          toast('err', 'Copy failed');
        }
      };
      meta.appendChild(copy);
    }
    row.appendChild(meta);

    const bub = document.createElement('div');
    bub.className = 'bubble';
    bub.innerHTML = escHtml(text).replace(/\n/g,'<br>');

    if (extra) {
      const eDiv = document.createElement('div');
      eDiv.style.cssText = 'margin-top:10px;';
      eDiv.innerHTML = extra;
      bub.appendChild(eDiv);
    }
    row.appendChild(bub);

    if (trace) {
      row.appendChild(buildTrace(trace));
    }
    el('thread').appendChild(row);
    scrollToBottom(role === 'user'); // Force scroll for user, smart scroll for bot
    return row;
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function scorePct(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    const bounded = Math.max(0, Math.min(1, n));
    return Math.round(bounded * 100);
  }

  function stopReasonLabel(code) {
    const labels = {
      reviewer_complete: 'Reviewer complete',
      score_threshold: 'Score threshold',
      budget_exhausted: 'Budget exhausted',
      reviewer_continue: 'Reviewer continue',
      score_below_threshold: 'Score below threshold'
    };
    return labels[code] || String(code || '').replace(/_/g, ' ');
  }

  function autoPolicyPills(policy) {
    if (!policy) return [];
    const selected = policy.selected_agent_mode || policy.resolved_agent_mode || 'off';
    const bits = [`<span class="trace-pill">Auto ${escHtml(selected)}</span>`];
    if (policy.budget_profile) bits.push(`<span class="trace-pill">Budget ${escHtml(policy.budget_profile)}</span>`);
    if (policy.session_budget) {
      const b = policy.session_budget;
      bits.push(`<span class="trace-pill">Session ${escHtml(b.remaining_cost_units)} / ${escHtml(b.limit_cost_units)}</span>`);
    }
    if (policy.score !== undefined) bits.push(`<span class="trace-pill">Difficulty ${escHtml(policy.score)}</span>`);
    if (policy.score_before_budget !== undefined && policy.score_before_budget !== policy.score) {
      bits.push(`<span class="trace-pill">Score ${escHtml(policy.score_before_budget)} -> ${escHtml(policy.score)}</span>`);
    }
    if (policy.reason) bits.push(`<span class="trace-pill">${escHtml(stopReasonLabel(policy.reason))}</span>`);
    if (Array.isArray(policy.reasons) && policy.reasons.length) {
      bits.push(`<span class="trace-pill">${escHtml(policy.reasons.slice(0,3).join(', '))}</span>`);
    }
    if (policy.feedback_adjustment) {
      const adj = policy.feedback_adjustment;
      const label = adj.reason === 'recent_weighted_feedback_regression'
        ? 'Adaptive'
        : (adj.reason === 'adaptive_quality_cost_preferred_neighbor' ? 'Pareto' : 'Feedback');
      bits.push(`<span class="trace-pill">${label} ${escHtml(adj.from || '')} -> ${escHtml(adj.to || '')}</span>`);
    } else if (policy.feedback_summary && policy.feedback_summary.total_feedback) {
      bits.push(`<span class="trace-pill">Feedback ${escHtml(policy.feedback_summary.total_feedback)}</span>`);
    }
    if (policy.uncertainty_adjustment) {
      const adj = policy.uncertainty_adjustment;
      bits.push(`<span class="trace-pill">Uncertain ${escHtml(adj.from || '')} -> ${escHtml(adj.to || '')}</span>`);
    }
    if (policy.session_budget_adjustment) {
      const adj = policy.session_budget_adjustment;
      bits.push(`<span class="trace-pill">Paced ${escHtml(adj.from || '')} -> ${escHtml(adj.to || '')}</span>`);
    }
    return bits;
  }

  function routeEconomicsPills(economics) {
    if (!economics) return [];
    const estimate = economics.estimate || {};
    const actual = economics.actual || {};
    const bits = [];
    if (estimate.estimated_cost_units !== undefined) {
      bits.push(`<span class="trace-pill">Cost ~${escHtml(estimate.estimated_cost_units)}</span>`);
    }
    if (estimate.estimated_model_calls !== undefined) {
      bits.push(`<span class="trace-pill">Planned calls ${escHtml(estimate.estimated_model_calls)}</span>`);
    }
    if (actual.elapsed_ms !== undefined) {
      bits.push(`<span class="trace-pill">Elapsed ${escHtml(actual.elapsed_ms)}ms</span>`);
    }
    if (actual.model_calls !== undefined) {
      bits.push(`<span class="trace-pill">Calls ${escHtml(actual.model_calls)}</span>`);
    }
    return bits;
  }

  function computePills(compute) {
    if (!compute || !Object.keys(compute).length) return [];
    const bits = [];
    const requested = compute.requested_reasoning_cycles ?? compute.selected_reasoning_cycles;
    const used = compute.cycles_used;
    if (requested !== undefined && requested !== null) {
      bits.push(`<span class="trace-pill">Compute ${escHtml(requested)} requested</span>`);
    }
    if (used !== undefined && used !== null) {
      bits.push(`<span class="trace-pill trace-score">${escHtml(used)} cycles used</span>`);
    }
    if (compute.reasoning_budget_mode) {
      bits.push(`<span class="trace-pill">${escHtml(compute.reasoning_budget_mode)} budget</span>`);
    }
    if (compute.adaptive_compute !== undefined) {
      bits.push(`<span class="trace-pill">Adaptive ${compute.adaptive_compute ? 'on' : 'off'}</span>`);
    }
    if (compute.exit_reason) {
      bits.push(`<span class="trace-pill">Exit ${escHtml(stopReasonLabel(compute.exit_reason))}</span>`);
    }
    if (compute.prediction_confidence_delta !== undefined && compute.prediction_confidence_delta !== null) {
      bits.push(`<span class="trace-pill">Prediction drift ${escHtml(compute.prediction_confidence_delta)}</span>`);
    }
    if (compute.applied === false && compute.supported === false) {
      bits.push('<span class="trace-pill">Compute controls unsupported</span>');
    }
    return bits;
  }

  function buildTrace(trace) {
    const wrapper = document.createElement('div');
    wrapper.className = 'trace';

    if (trace.loop_steps && trace.loop_steps.length) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--amber)';
      hdr.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12,4V1L8,5l4,4V6c3.31,0,6,2.69,6,6a5.987,5.987,0,0,1-.7,2.8l1.46,1.46A7.93,7.93,0,0,0,20,12C20,7.58,16.42,4,12,4Zm0,14c-3.31,0-6-2.69-6-6a5.987,5.987,0,0,1,.7-2.8L5.24,7.74A7.93,7.93,0,0,0,4,12c0,4.42,3.58,8,8,8v3l4-4-4-4Z"/></svg> Autonomous Logic Chain — ${trace.loop_steps.length} cycles`;
      wrapper.appendChild(hdr);

      const summaryBits = autoPolicyPills(trace.auto_agent_policy);
      summaryBits.push(...routeEconomicsPills(trace.route_economics));
      summaryBits.push(...computePills(trace.compute));
      const stopScore = scorePct(trace.loop_stop_score);
      if (trace.loop_stop_reason_code) summaryBits.push(`<span class="trace-pill">${escHtml(stopReasonLabel(trace.loop_stop_reason_code))}</span>`);
      if (stopScore != null) summaryBits.push(`<span class="trace-pill trace-score">Stop score ${stopScore}%</span>`);
      if (trace.loop_budget != null) summaryBits.push(`<span class="trace-pill">Budget ${escHtml(trace.loop_steps.length)}/${escHtml(trace.loop_budget)}</span>`);
      if (trace.loop_completion_reason) summaryBits.push(`<span class="trace-pill">${escHtml(String(trace.loop_completion_reason).slice(0,160))}</span>`);
      if (summaryBits.length) {
        const summary = document.createElement('div');
        summary.className = 'trace-body trace-summary';
        summary.innerHTML = summaryBits.join('');
        wrapper.appendChild(summary);
      }

      const body = document.createElement('div');
      body.className = 'trace-body';
      trace.loop_steps.forEach(s => {
        const stepScore = scorePct(s.review_score ?? s.loop_score);
        const scoreHtml = stepScore == null ? '' : `<span class="trace-pill trace-score">Score ${stepScore}%</span>`;
        const stopHtml = s.stop_decision === 'stop'
          ? `<span class="trace-pill">${escHtml(stopReasonLabel(s.stop_reason_code))}</span>`
          : '';
        const note = s.review_note || s.completion_evidence || s.next_step || '';
        body.innerHTML += `<div class="trace-step">
          <div class="trace-step-n">${s.step}</div>
          <div><strong style="color:var(--text)">${escHtml(s.goal||'Strategy Initialization')}</strong><br>
          <span style="color:var(--muted);font-size:11px">${escHtml((s.worker_excerpt||'').slice(0,140))}...</span>
          <div style="margin-top:6px">${scoreHtml}${stopHtml}</div>
          ${note ? `<div style="color:var(--muted);font-size:11px;margin-top:4px">${escHtml(String(note).slice(0,160))}</div>` : ''}</div>
        </div>`;
      });
      wrapper.appendChild(body);

    } else if (trace.reasoning_passes != null) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--teal)';
      hdr.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12,2A10,10,0,1,0,22,12,10.011,10.011,0,0,0,12,2Zm1,15H11V11h2Zm0-8H11V7h2Z"/></svg> V46 Reasoning Telemetry`;
      wrapper.appendChild(hdr);
      const body = document.createElement('div');
      body.className = 'trace-body';
      body.innerHTML = `<div class="trace-grid">
        <div class="trace-kv">Reasoning Passes: <strong>${trace.reasoning_passes}</strong></div>
        <div class="trace-kv">GoT Synthesis: <strong style="color:${trace.graph_synthesis_applied?'var(--teal)':'var(--muted)'}">${trace.graph_synthesis_applied?'Enabled':'Bypassed'}</strong></div>
        <div class="trace-kv">MoD Routed: <strong>${trace.mixture_of_depths_skipped===0?'Full':'Optimized'}</strong></div>
        <div class="trace-kv">C-CoT Latent: <strong style="color:${trace.continuous_latent_active?'var(--cyan)':'var(--muted)'}">${trace.continuous_latent_active?'Fluid':'Static'}</strong></div>
      </div>`;
      wrapper.appendChild(body);

    } else if (trace.consulted_models && trace.consulted_models.length) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--purple)';
      hdr.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16,11c1.66,0,2.99-1.34,2.99-3S17.66,5,16,5s-3,1.34-3,3S14.34,11,16,11Zm-8,0c1.66,0,2.99-1.34,2.99-3S9.66,5,8,5S5,6.34,5,8,6.34,11,8,11Zm0,2c-2.33,0-7,1.17-7,3.5V19H15V16.5C15,14.17,10.33,13,8,13Zm8,0c-.29,0-.62,.02-.97,.05C16.52,14.3,17,15.77,17,17.5V19H23V16.5C23,14.17,18.33,13,16,13Z"/></svg> Ensemble Consultation — ${trace.consulted_models.length} Nodes`;
      wrapper.appendChild(hdr);
      const body = document.createElement('div');
      body.className = 'trace-body';
      body.style.fontSize = '12px';
      const policyBits = autoPolicyPills(trace.auto_agent_policy)
        .concat(routeEconomicsPills(trace.route_economics), computePills(trace.compute));
      body.innerHTML = (policyBits.length ? `<div class="trace-summary" style="margin-bottom:10px">${policyBits.join('')}</div>` : '') +
        '<span style="color:var(--muted)">Expert weights synthesized from:</span> ' + trace.consulted_models.join(', ');
      wrapper.appendChild(body);
    } else if (trace.auto_agent_policy) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = '#818cf8';
      hdr.textContent = 'Adaptive Router';
      wrapper.appendChild(hdr);
      const body = document.createElement('div');
      body.className = 'trace-body trace-summary';
      body.innerHTML = autoPolicyPills(trace.auto_agent_policy)
        .concat(routeEconomicsPills(trace.route_economics), computePills(trace.compute)).join('');
      wrapper.appendChild(body);
    } else if (trace.route_economics) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--blue)';
      hdr.textContent = 'Route Economics';
      wrapper.appendChild(hdr);
      const body = document.createElement('div');
      body.className = 'trace-body trace-summary';
      body.innerHTML = routeEconomicsPills(trace.route_economics).concat(computePills(trace.compute)).join('');
      wrapper.appendChild(body);
    } else if (trace.compute && Object.keys(trace.compute).length) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--teal)';
      hdr.textContent = 'Adaptive Compute';
      wrapper.appendChild(hdr);
      const body = document.createElement('div');
      body.className = 'trace-body trace-summary';
      body.innerHTML = computePills(trace.compute).join('');
      wrapper.appendChild(body);
    }

    if (!wrapper.children.length) return document.createTextNode('');
    return wrapper;
  }

  // ── Loop step UI ─────────────────────────────────────────────────────
  function addLoopStep(n, title, sub, state='active') {
    const steps = el('loopSteps');
    loopStep = n;
    const item = document.createElement('div');
    item.className = 'lstep';
    item.id = `lstep-${n}`;
    item.innerHTML = `<div class="lstep-n ${state}">${n}</div>
      <div class="lstep-info">
        <div class="lstep-title">${escHtml(title)}</div>
        <div class="lstep-sub">${escHtml(sub)}</div>
      </div>`;
    steps.appendChild(item);
    if (el('ptab-mode').style.display !== 'none') {
      item.scrollIntoView({ behavior:'smooth', block:'nearest' });
    }
  }

  function finaliseLoopSteps() {
    qsa('.lstep-n').forEach(n => {
      n.classList.remove('active');
      n.classList.add('done');
    });
  }

  function setRouteFeedbackVisible(show, label='Route') {
    const box = el('routeFeedback');
    if (!box) return;
    box.style.display = show ? 'flex' : 'none';
    const labelEl = el('routeFeedbackLabel');
    if (labelEl) labelEl.textContent = label;
    ['routeGoodBtn','routeBadBtn','routeDeeperBtn','routeCostBtn','routeSlowBtn'].forEach(id => {
      const btn = el(id);
      if (btn) btn.disabled = false;
    });
  }

  function routeMetricText(value, digits=1) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    return n.toFixed(digits).replace(/\.0$/, '');
  }

  function routeQualityText(adaptive) {
    if (!adaptive) return 'Quality -';
    const q = Number(adaptive.quality_score);
    if (!Number.isFinite(q)) return 'Quality -';
    const suffix = adaptive.regression_signal ? ' risk' : '';
    return `Quality ${Math.round(Math.max(0, Math.min(1, q)) * 100)}%${suffix}`;
  }

  function routeConfidenceText(adaptive) {
    if (!adaptive) return 'Recent evidence -';
    const lowerRaw = adaptive.quality_lower_bound;
    const upperRaw = adaptive.quality_upper_bound;
    if (lowerRaw === null || lowerRaw === undefined || upperRaw === null || upperRaw === undefined) return 'Recent evidence -';
    const lower = Number(lowerRaw);
    const upper = Number(upperRaw);
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) return 'Recent evidence -';
    const status = adaptive.confidence_status ? ` ${adaptive.confidence_status}` : '';
    return `Heuristic ${Math.round(Math.max(0, lower) * 100)}-${Math.round(Math.min(1, upper) * 100)}%${status}`;
  }

  function routePreferenceText(adaptive) {
    if (!adaptive) return 'Preference -';
    if (adaptive.preference_direction === 'deeper') return 'Preference deeper';
    if (adaptive.preference_direction === 'shallower') {
      const cost = Number(adaptive.weighted_cost_pressure) || 0;
      const latency = Number(adaptive.weighted_latency_pressure) || 0;
      return latency > cost ? 'Preference faster' : 'Preference cheaper';
    }
    return 'Preference neutral';
  }

  function renderRouteHealth(summary) {
    const box = el('routeHealth');
    if (!box) return;
    const usage = summary && summary.route_usage ? summary.route_usage : {};
    const usageEconomics = usage && usage.economics ? usage.economics : {};
    const feedbackEconomics = summary && summary.economics ? summary.economics : {};
    const economics = usageEconomics.sample_count ? usageEconomics : feedbackEconomics;
    const adaptive = summary && summary.adaptive ? summary.adaptive : {};
    const total = summary && summary.total_feedback !== undefined ? Number(summary.total_feedback) : NaN;
    const routeTotal = usage && usage.total_routes !== undefined ? Number(usage.total_routes) : NaN;
    const samples = economics.sample_count ? Number(economics.sample_count) : 0;
    const shownTotal = Number.isFinite(routeTotal) && routeTotal ? routeTotal : Number.isFinite(total) ? total : samples;
    if (!shownTotal && !samples) {
      box.style.display = 'none';
      return;
    }
    box.style.display = 'flex';
    el('routeHealthCount').textContent = `Routes ${shownTotal}`;
    el('routeHealthQuality').textContent = routeQualityText(adaptive);
    el('routeHealthConfidence').textContent = routeConfidenceText(adaptive);
    el('routeHealthPreference').textContent = routePreferenceText(adaptive);
    el('routeHealthCost').textContent = `Avg cost ${routeMetricText(economics.avg_cost_units, 2)}`;
    el('routeHealthLatency').textContent = `Avg ms ${routeMetricText(economics.avg_elapsed_ms, 0)}`;
  }

  async function refreshRouteHealth() {
    try {
      const result = await api('/api/route_health', { session_id: sessionId });
      renderRouteHealth(result.route_health || result.summary || {});
    } catch (_) {}
  }

  function policyLabPercent(value) {
    if (value === null || value === undefined || value === '') return '-';
    const n = Number(value);
    return Number.isFinite(n) ? `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%` : '-';
  }

  function policyLabLabel(value) {
    return String(value || '').replace(/_/g, ' ').trim();
  }

  const policyLabCheckLabels = {
    candidate_delta_present: 'Candidate changes at least one action',
    population_integrity_complete: 'Durable population is unique and evaluable',
    execution_integrity_complete: 'Execution state and chosen action reconcile',
    logging_integrity_complete: 'Logging metadata and decision fingerprint verify',
    minimum_overlap_routes_met: 'Minimum overlap route count',
    target_probability_floor_met: 'Target probability floor',
    global_overlap_ess_met: 'Global overlap ESS floor',
    per_action_overlap_met: 'Per-action overlap ESS floor',
    outcome_evidence_integrity: 'Outcome contracts and eligible quality evidence verify',
    quality_observation_ready: 'Quality observation process',
    durable_lifecycle_present: 'Durable lifecycle evidence',
    lifecycle_reconciled: 'Lifecycle fully reconciled',
  };

  function renderPolicyLabChecks(checks, blockers) {
    const container = el('policyLabChecks');
    container.replaceChildren();
    const knownChecks = checks && typeof checks === 'object' ? checks : {};
    Object.entries(policyLabCheckLabels).forEach(([key, label]) => {
      const item = document.createElement('div');
      const known = Object.prototype.hasOwnProperty.call(knownChecks, key);
      item.className = 'policy-lab-check';
      item.dataset.state = known ? (knownChecks[key] ? 'pass' : 'fail') : 'unknown';
      item.textContent = label;
      container.appendChild(item);
    });
    const blockerList = Array.isArray(blockers) ? blockers.filter(Boolean) : [];
    const blockerBox = el('policyLabBlockers');
    if (blockerList.length) {
      blockerBox.textContent = `Blockers: ${blockerList.map(policyLabLabel).join(' | ')}`;
    } else if (Object.keys(knownChecks).length) {
      blockerBox.textContent = 'No readiness blockers; validated external OPE is still required.';
    } else {
      blockerBox.textContent = 'Blockers unavailable until replay succeeds.';
    }
  }

  function renderPolicyLabUnavailable(message) {
    const defaults = {
      policyLabJoined: '0 / 0', policyLabAgreement: '-', policyLabApproval: '-',
      policyLabEconomics: '-', policyLabLifecycle: '0 / 0 / 0', policyLabFeedbackCoverage: '-',
      policyLabOverlapEss: 'ESS - / -', policyLabWeakestAction: '-',
      policyLabReadinessChecks: `0 / ${Object.keys(policyLabCheckLabels).length}`,
      policyLabOutcomeCoverage: '0 / 0',
      policyLabContractCoverage: '0 / 0',
      policyLabEvidenceSource: 'unavailable',
    };
    Object.entries(defaults).forEach(([id, value]) => { el(id).textContent = value; });
    el('policyLabGate').textContent = `Policy replay unavailable: ${message}`;
    el('policyLabNote').textContent = 'No Policy Lab metrics are current. Shadow-only gating remains in force.';
    renderPolicyLabChecks({}, []);
  }

  function renderPolicyLab(report) {
    const support = report && report.support ? report.support : {};
    const usage = support.usage || {};
    const agreement = report && report.candidate_action_agreement ? report.candidate_action_agreement : {};
    const matched = report && report.matched_observed ? report.matched_observed : {};
    const gate = report && report.promotion_gate ? report.promotion_gate : {};
    const propensity = report && report.propensity_readiness ? report.propensity_readiness : {};
    const readiness = report && report.evaluation_readiness ? report.evaluation_readiness : {};
    const thresholds = readiness.thresholds || {};
    const overlap = readiness.target_overlap || {};
    const outcome = readiness.outcome_observation || {};
    const maturity = report && report.outcome_contract_maturity && typeof report.outcome_contract_maturity === 'object'
      ? report.outcome_contract_maturity
      : {};
    const maturityByOutcome = maturity.by_outcome && typeof maturity.by_outcome === 'object'
      ? maturity.by_outcome
      : {};
    const qualityMaturity = maturityByOutcome.user_quality_rating && typeof maturityByOutcome.user_quality_rating === 'object'
      ? maturityByOutcome.user_quality_rating
      : {};
    const durable = report && report.durable_ledger ? report.durable_ledger : {};
    const lifecycle = durable.counts || {};
    const feedbackCoverage = durable.feedback_coverage || {};
    const evidenceSource = String(report && report.evidence_source ? report.evidence_source : 'unknown');
    const durableEvidence = evidenceSource === 'durable_sqlite_ledger';
    el('policyLabJoined').textContent = `${Number(support.exact_joined_route_ids) || 0} / ${Number(usage.unique_route_ids) || 0}`;
    el('policyLabAgreement').textContent = policyLabPercent(agreement.agreement_rate);
    const approval = policyLabPercent(matched.approval_rate);
    el('policyLabApproval').textContent = matched.quality_sample_count ? `${approval} (n=${matched.quality_sample_count})` : '-';
    const hasCost = matched.avg_cost_units !== null && matched.avg_cost_units !== undefined && matched.avg_cost_units !== '';
    const hasLatency = matched.avg_elapsed_ms !== null && matched.avg_elapsed_ms !== undefined && matched.avg_elapsed_ms !== '';
    const cost = hasCost ? Number(matched.avg_cost_units) : NaN;
    const latency = hasLatency ? Number(matched.avg_elapsed_ms) : NaN;
    el('policyLabEconomics').textContent = Number.isFinite(cost) || Number.isFinite(latency)
      ? `${Number.isFinite(cost) ? cost.toFixed(2) : '-'} / ${Number.isFinite(latency) ? Math.round(latency) + ' ms' : '-'}`
      : '-';
    el('policyLabLifecycle').textContent =
      `${Number(lifecycle.completed) || 0} / ${Number(lifecycle.failed) || 0} / ${Number(lifecycle.inflight) || 0}`;
    el('policyLabFeedbackCoverage').textContent = policyLabPercent(feedbackCoverage.terminal_coverage_rate);
    const overlapEss = Number(overlap.effective_sample_size) || 0;
    const globalEssFloor = Number(thresholds.minimum_global_effective_sample_size) || 20;
    const actionEssFloor = Number(thresholds.minimum_per_action_effective_sample_size) || 10;
    const weakestAction = policyLabLabel(overlap.weakest_target_action || 'none');
    const weakestEss = Number(overlap.weakest_action_effective_sample_size) || 0;
    el('policyLabOverlapEss').textContent = `ESS ${overlapEss.toFixed(1)} / ${globalEssFloor.toFixed(0)}`;
    el('policyLabWeakestAction').textContent = overlap.weakest_target_action
      ? `${weakestAction} ${weakestEss.toFixed(1)} / ${actionEssFloor.toFixed(0)}`
      : '-';
    const totalChecks = Number(gate.total_checks) || Object.keys(policyLabCheckLabels).length;
    el('policyLabReadinessChecks').textContent = `${Number(gate.passed_checks) || 0} / ${totalChecks}`;
    el('policyLabOutcomeCoverage').textContent =
      `${Number(outcome.quality_observed_routes) || 0} / ${Number(outcome.evaluable_routes) || 0}`;
    el('policyLabContractCoverage').textContent =
      `${Number(maturity.precommitted_routes) || 0} / ${Number(maturity.included_routes) || 0}`;
    el('policyLabEvidenceSource').textContent = durableEvidence ? 'durable SQLite' : policyLabLabel(evidenceSource);
    const gateStatus = policyLabLabel(gate.status || 'blocked');
    const gateReason = policyLabLabel(gate.reason_code || 'no evidence');
    const blockers = Array.isArray(gate.blocking_reason_codes) ? gate.blocking_reason_codes : [];
    const moreBlockers = Math.max(0, blockers.length - 1);
    renderPolicyLabChecks(gate.checks || {}, blockers);
    el('policyLabGate').textContent =
      `${gateStatus} · ${Number(gate.passed_checks) || 0}/${totalChecks} checks · ${gateReason}` +
      `${moreBlockers ? ` + ${moreBlockers} more` : ''} · ${policyLabLabel(gate.deployment || 'shadow only')}`;
    const valid = Number(propensity.valid_routes) || 0;
    const checked = Number(propensity.checked_evaluable_usage_routes) || 0;
    const started = Number(lifecycle.started) || 0;
    const minTarget = overlap.minimum_target_probability === null || overlap.minimum_target_probability === undefined
      ? '-'
      : Number(overlap.minimum_target_probability).toFixed(3);
    el('policyLabNote').textContent =
      `${durableEvidence ? 'Durable ledger evidence.' : 'Nondurable compatibility evidence; readiness is blocked.'} ` +
      `Durable routes ${started}; target overlap ESS ${overlapEss.toFixed(1)}/${globalEssFloor.toFixed(0)}; ` +
      `minimum target propensity ${minTarget}; quality observed ${Number(outcome.quality_observed_routes) || 0}/${Number(outcome.evaluable_routes) || 0}. ` +
      `Outcome contracts ${Number(maturity.precommitted_routes) || 0}/${Number(maturity.included_routes) || 0} precommitted; ` +
      `quality events ${Number(qualityMaturity.observed_event_count) || 0}, mature contracts ${Number(qualityMaturity.mature_contract_count) || 0}. ` +
      `Missing feedback stays unknown; maturity is diagnostic only; ${valid}/${checked} usage rows validate logging metadata. No policy value was estimated.`;
  }

  async function refreshPolicyLab() {
    const button = el('policyLabRefresh');
    if (button) button.disabled = true;
    try {
      const result = await api('/api/route_policy_lab', {
        session_id: sessionId,
        profile: el('policyLabProfile').value,
      });
      renderPolicyLab(result.policy_lab || {});
    } catch (err) {
      renderPolicyLabUnavailable(err.message);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function routeStudyNumber(value, digits = 0) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '-';
    return number.toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function renderRouteStudyUnavailable(message) {
    latestRouteStudy = null;
    el('routeBundleAdd').disabled = true;
    const defaults = {
      routeStudyBaseline: '-', routeStudyFloor: '-', routeStudyTraffic: '-',
      routeStudyCost: '-', routeStudyLatency: '-', routeStudyCharter: '-',
      routeProtocolMode: '-', routeProtocolPolicy: '-', routeProtocolReview: '-',
      routeProtocolHash: '-',
    };
    Object.entries(defaults).forEach(([id, value]) => { el(id).textContent = value; });
    el('routeStudyDistribution').replaceChildren();
    el('routeProtocolBlockers').replaceChildren();
    el('routeStudyStatus').textContent = message;
    el('routeStudyNote').textContent =
      'No study probabilities are current. Execution, evidence writes, policy value, and promotion remain unavailable.';
  }

  function renderRouteShadowRegistry(snapshot) {
    const campaigns = Array.isArray(snapshot && snapshot.campaigns) ? snapshot.campaigns : [];
    const chain = snapshot && snapshot.event_chain ? snapshot.event_chain : null;
    const available = snapshot && snapshot.available === true;
    const committed = campaigns.reduce((total, row) => total + (Number(row.commitment_count) || 0), 0);
    const matched = campaigns.reduce((total, row) => total + (Number(row.matched_assignment_count) || 0), 0);
    const processed = campaigns.reduce((total, row) => total + (Number(row.processed_reveal_count) || 0), 0);
    const mismatched = campaigns.reduce((total, row) => total + (Number(row.mismatched_assignment_count) || 0), 0);
    const states = [...new Set(campaigns.map(row => policyLabLabel((row || {}).state || 'unknown')))];
    el('routeShadowRegistryCount').textContent = routeStudyNumber(snapshot && snapshot.campaign_count, 0);
    el('routeShadowRegistryChain').textContent = chain
      ? `${chain.ok ? 'verified' : 'failed'} - ${routeStudyNumber(chain.verified_events, 0)} events`
      : 'not initialized';
    el('routeShadowRegistryAssignments').textContent = `${routeStudyNumber(matched, 0)} / ${routeStudyNumber(committed, 0)}`;
    el('routeShadowRegistryState').textContent = states.length ? states.join(', ') : 'not initialized';

    const inventory = el('routeShadowRegistryCampaigns');
    inventory.replaceChildren();
    campaigns.forEach(row => {
      const chip = document.createElement('span');
      chip.className = 'route-study-chip';
      chip.dataset.state = Number(row.mismatched_assignment_count) > 0 ? 'unresolved' : '';
      chip.textContent = `${row.campaign_id || 'campaign'} - ${policyLabLabel(row.state || 'unknown')} - ${routeStudyNumber(row.matched_assignment_count, 0)}/${routeStudyNumber(row.commitment_count, 0)} matched`;
      inventory.appendChild(chip);
    });

    if (!available) {
      el('routeShadowRegistryStatus').textContent =
        `No shadow registry exists at ${snapshot && snapshot.registry_location ? snapshot.registry_location : 'the canonical memory path'}. ` +
        'Browser access is read-only; execution, activation, and automatic promotion remain unavailable.';
      return;
    }
    el('routeShadowRegistryStatus').textContent =
      `${snapshot.ok ? 'Registry verification passed.' : 'Registry verification failed.'} ` +
      `${campaigns.length} campaign(s), ${committed} opaque commitment(s), ${matched} matched assignment(s), ${processed} processed reveal(s), ${mismatched} mismatch(es). ` +
      'Local chain verification is not an external transparency anchor. Browser access is read-only; execution, activation, and promotion remain unavailable.';
  }

  async function refreshRouteShadowRegistry() {
    const button = el('routeShadowRegistryRefresh');
    button.disabled = true;
    el('routeShadowRegistryStatus').textContent = 'Reading and verifying the local shadow registry...';
    try {
      const result = await api('/api/route_shadow_registry/status');
      renderRouteShadowRegistry(result.route_shadow_registry || {});
    } catch (err) {
      renderRouteShadowRegistry({available:false, campaign_count:0, campaigns:[]});
      el('routeShadowRegistryStatus').textContent = `Registry status error: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  }

  function renderRouteStudy(payload) {
    const study = payload && payload.route_study ? payload.route_study : {};
    const charter = study.charter || {};
    const enrollment = charter.enrollment || {};
    const design = charter.probability_design || {};
    const traffic = charter.traffic_scenario || {};
    const labelScenario = traffic.observed_label_scenario || {};
    const exact = labelScenario.exact_simultaneous_target || {};
    const resources = charter.resource_forecast || {};
    const boundaries = charter.causal_boundaries || {};
    const protocol = payload && payload.route_protocol_preflight
      ? payload.route_protocol_preflight : null;
    const protocolCharter = protocol ? (protocol.charter || {}) : {};
    const stateful = protocolCharter.stateful_design || {};
    const targetClass = protocolCharter.target_policy_class || {};
    const protocolMeta = protocol ? (protocol.protocol || {}) : {};
    const blockerRegister = Array.isArray(protocolCharter.blocker_register)
      ? protocolCharter.blocker_register : [];
    const totals = resources.expected_for_planned_routes || {};
    const probabilities = design.action_probabilities || {};
    const baseline = enrollment.baseline_action || payload.baseline_agent_mode || '-';
    const floor = design.minimum_positive_exploration_probability;
    const expectedByAlternate = labelScenario.expected_routes_for_target_by_alternate_action || {};
    const expectedRouteValues = Object.values(expectedByAlternate)
      .map(Number).filter(Number.isFinite);
    const expectedRoutes = expectedRouteValues.length ? Math.max(...expectedRouteValues) : null;
    const confidenceRoutes = exact.minimum_routes_for_target_on_every_alternate_action;
    const confidencePct = Number(exact.confidence_level) * 100;
    const targetLabels = Number(labelScenario.target_observed_labels_per_alternate_action) || 0;
    const eligible = enrollment.eligible === true;
    latestRouteStudy = eligible ? JSON.parse(JSON.stringify(study)) : null;
    el('routeBundleAdd').disabled = !latestRouteStudy;
    el('routeStudyBaseline').textContent = policyLabLabel(baseline) || '-';
    el('routeStudyFloor').textContent = Number.isFinite(Number(floor))
      ? `${Math.round(Number(floor) * 100)}%`
      : 'not enrolled';
    el('routeStudyTraffic').textContent = expectedRoutes && confidenceRoutes
      ? `~${routeStudyNumber(expectedRoutes)} exp each / ${routeStudyNumber(confidenceRoutes)} @${routeStudyNumber(confidencePct)}% joint`
      : 'no alternate support';
    el('routeStudyCost').textContent = routeStudyNumber(totals.cost_units, 1);
    const tierOrder = ['low', 'moderate', 'high', 'frontier', 'unknown'];
    const tiers = Object.values(resources.by_action || {})
      .map(row => String((row || {}).latency_tier || 'unknown'))
      .filter((tier, index, values) => values.indexOf(tier) === index)
      .sort((a, b) => tierOrder.indexOf(a) - tierOrder.indexOf(b));
    el('routeStudyLatency').textContent = tiers.length
      ? `${policyLabLabel(tiers[0])}${tiers.length > 1 ? ' to ' + policyLabLabel(tiers[tiers.length - 1]) : ''}`
      : '-';
    el('routeStudyCharter').textContent = study.design_hash ? String(study.design_hash).slice(0, 12) : '-';
    el('routeProtocolMode').textContent = protocol
      ? policyLabLabel(stateful.selected_design_mode || 'not screened') : 'not enrolled';
    el('routeProtocolPolicy').textContent = protocol
      ? policyLabLabel(targetClass.profile_name || 'not frozen') : '-';
    el('routeProtocolReview').textContent = protocol
      ? policyLabLabel(stateful.selected_design_status || protocolMeta.state || 'blocked')
      : policyLabLabel(payload.route_protocol_preflight_reason || 'unavailable');
    el('routeProtocolHash').textContent = protocol && protocol.protocol_hash
      ? String(protocol.protocol_hash).slice(0, 12) : '-';

    const distribution = el('routeStudyDistribution');
    distribution.replaceChildren();
    Object.entries(probabilities).forEach(([action, probability]) => {
      const chip = document.createElement('span');
      chip.className = 'route-study-chip';
      chip.textContent = `${policyLabLabel(action)} ${Math.round(Number(probability) * 100)}%`;
      distribution.appendChild(chip);
    });
    const blockerList = el('routeProtocolBlockers');
    blockerList.replaceChildren();
    blockerRegister.forEach(row => {
      const chip = document.createElement('span');
      chip.className = 'route-study-chip';
      chip.dataset.state = String((row || {}).status || 'unresolved');
      chip.textContent = `${policyLabLabel((row || {}).code || 'blocker')} - ${policyLabLabel((row || {}).status || 'unresolved')}`;
      blockerList.appendChild(chip);
    });
    const adjacent = Array.isArray(enrollment.adjacent_feasible_actions)
      ? enrollment.adjacent_feasible_actions.map(policyLabLabel).join(', ')
      : '';
    el('routeStudyStatus').textContent = eligible
      ? `Rehearsal ready - ${adjacent || 'adjacent support'} - protocol ${policyLabLabel(protocolMeta.state || 'unavailable')} - activation, assignment, and execution remain off.`
      : `Not enrolled - ${policyLabLabel(enrollment.reason || 'no feasible adjacent action')} - baseline remains deterministic.`;
    const blockers = Array.isArray(boundaries.activation_blockers)
      ? boundaries.activation_blockers.map(item => String(item).replaceAll('_', ' '))
      : [];
    const trafficMethod = String(exact.method || 'not applicable').replaceAll('_', ' ');
    const designReasons = Array.isArray(stateful.selected_design_blocking_reasons)
      ? stateful.selected_design_blocking_reasons.map(item => String(item).replaceAll('_', ' '))
      : [];
    el('routeStudyNote').textContent =
      `Hypothetical repetition of this prompt-specific support: ${routeStudyNumber(traffic.planned_routes)} routes; ` +
      `target ${routeStudyNumber(targetLabels)} observed ratings on every alternate; ` +
      `assumed response ${routeStudyNumber(Number(labelScenario.assumed_feedback_rate) * 100)}%. ` +
      `${trafficMethod}. This is not power, an observation model, OPE, or a promotion decision. ` +
      `Stateful preflight: ${policyLabLabel(stateful.selected_design_status || 'not available')}; ` +
      `${designReasons.length ? 'design review reasons: ' + designReasons.join('; ') + '. ' : ''}` +
      `Declarations are not validation, route-level campaign assignment is screened out, and no seed is sealed. ` +
      `Activation blockers (${blockers.length}): ${blockers.join('; ') || 'live integration unavailable'}. ` +
      `No route was assigned, executed, or written.`;
  }

  function routeProtocolBuildInput() {
    const plannedClusters = parseInt(el('routeProtocolClusters').value, 10);
    return {
      study_plans: routeStudyStrata.map(plan => JSON.parse(JSON.stringify(plan))),
      target_policy_profile: el('routeProtocolTarget').value,
      design_mode: el('routeProtocolDesign').value,
      carryover_scope: el('routeProtocolCarryover').value,
      interference_scope: el('routeProtocolInterference').value,
      temporal_variation: el('routeProtocolTemporal').value,
      population_rule_id: 'interactive-auto-route-opt-in',
      population_rule_version: '1',
      cluster_key_schema_version: 'session-hash-v1',
      planned_clusters: plannedClusters,
      max_routes_per_cluster: 20,
      analysis_every_clusters: Math.min(plannedClusters, 50),
      block_length_routes: parseInt(el('routeProtocolBlock').value, 10),
      washout_routes: parseInt(el('routeProtocolWashout').value, 10),
      seed_commitment: null,
      external_estimator_id: null,
      external_reviewer_id: null,
    };
  }

  function invalidateRouteReviewBundle(message) {
    latestRouteReviewBundle = null;
    el('routeBundleVerification').textContent = 'not built';
    el('routeBundleHash').textContent = '-';
    el('routeBundleDownload').disabled = true;
    el('routeBundleStatus').textContent = message ||
      'Inventory changed. Rebuild for full source-bound semantic verification.';
  }

  function renderRouteBundleInventory() {
    const inventory = el('routeBundleInventory');
    inventory.replaceChildren();
    routeStudyStrata.forEach((plan, index) => {
      const charter = (plan || {}).charter || {};
      const enrollment = charter.enrollment || {};
      const design = charter.probability_design || {};
      const hash = String((plan || {}).design_hash || '').slice(0, 10) || 'unhashed';
      const baseline = policyLabLabel(enrollment.baseline_action || 'unknown');
      const actions = Array.isArray(design.eligible_actions)
        ? design.eligible_actions.map(policyLabLabel).join('/') : 'unknown support';
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'route-study-chip';
      remove.textContent = `${index + 1}. ${hash} - ${baseline} - ${actions} - remove`;
      remove.setAttribute('aria-label', `Remove support stratum ${index + 1}, ${hash}`);
      remove.onclick = () => {
        routeStudyStrata.splice(index, 1);
        invalidateRouteReviewBundle('Support stratum removed. Rebuild before review.');
        renderRouteBundleInventory();
      };
      inventory.appendChild(remove);
    });
    const count = routeStudyStrata.length;
    el('routeBundleCount').textContent = `${count} ${count === 1 ? 'stratum' : 'strata'}`;
    el('routeBundleBuild').disabled = count === 0;
    el('routeBundleClear').disabled = count === 0;
  }

  function addCurrentRouteStudyStratum() {
    if (!latestRouteStudy) {
      toast('err', 'Rehearse an eligible prompt-specific support stratum first');
      return;
    }
    const designHash = String(latestRouteStudy.design_hash || '');
    if (!designHash) {
      toast('err', 'Current rehearsal has no canonical design hash');
      return;
    }
    if (routeStudyStrata.some(plan => String((plan || {}).design_hash || '') === designHash)) {
      toast('err', 'That support stratum is already in the review inventory');
      return;
    }
    if (routeStudyStrata.length >= 100) {
      toast('err', 'Browser review inventory is limited to 100 support strata');
      return;
    }
    routeStudyStrata.push(JSON.parse(JSON.stringify(latestRouteStudy)));
    invalidateRouteReviewBundle('Support stratum added. Build the bundle for semantic verification.');
    renderRouteBundleInventory();
    toast('ok', 'Prompt-free support stratum added to the review inventory');
  }

  function renderRouteBundleVerification(verification) {
    const checked = verification || {};
    const full = checked.verification_level === 'full_source_bound_reconstruction' &&
      checked.source_plan_reconstruction_performed === true;
    el('routeBundleVerification').textContent = full ? 'full reconstruction' : 'verification failed';
    el('routeBundleHash').textContent = checked.bundle_hash
      ? String(checked.bundle_hash).slice(0, 12) : '-';
    el('routeBundleDownload').disabled = !full || !latestRouteReviewBundle;
    el('routeBundleStatus').textContent = full
      ? `Verified ${checked.support_stratum_count || 0} canonical source strata by rebuilding the protocol. ` +
        'This is semantic conformance only: no signature, trusted timestamp, causal validation, assignment, or activation.'
      : 'Full source-bound reconstruction did not complete; the bundle is not review-ready.';
  }

  async function buildRouteReviewBundle() {
    if (!routeStudyStrata.length) {
      toast('err', 'Add at least one support stratum first');
      return;
    }
    const button = el('routeBundleBuild');
    button.disabled = true;
    el('routeBundleStatus').textContent = 'Reconstructing the campaign protocol from canonical prompt-free source plans...';
    try {
      const result = await api('/api/route_study_protocol_bundle', routeProtocolBuildInput());
      latestRouteReviewBundle = result.route_protocol_review_bundle || null;
      renderRouteBundleVerification(result.verification || {});
      toast('ok', 'Multi-stratum review bundle passed full source reconstruction');
    } catch (err) {
      invalidateRouteReviewBundle(`Review bundle unavailable: ${err.message}`);
      toast('err', err.message);
    } finally {
      renderRouteBundleInventory();
    }
  }

  function downloadRouteReviewBundle() {
    if (!latestRouteReviewBundle) return;
    const rendered = JSON.stringify(latestRouteReviewBundle, null, 2) + '\n';
    const blob = new Blob([rendered], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const hash = String(latestRouteReviewBundle.bundle_hash || 'unverified').slice(0, 12);
    link.href = url;
    link.download = `supermix-route-review-${hash}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  async function importRouteReviewFile(event) {
    const input = event.target;
    const file = input.files && input.files[0];
    if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) {
        throw new Error('Review file exceeds the 2 MiB browser limit');
      }
      const parsed = JSON.parse(await file.text());
      let result;
      if (parsed && parsed.schema_version === 'route-study-review-bundle-v1') {
        result = await api('/api/route_study_protocol_bundle/audit', {bundle: parsed});
        latestRouteReviewBundle = parsed;
      } else if (parsed && Array.isArray(parsed.study_plans)) {
        result = await api('/api/route_study_protocol_bundle', parsed);
        latestRouteReviewBundle = result.route_protocol_review_bundle || null;
      } else {
        throw new Error('Import must be a review bundle or closed prompt-free protocol build input');
      }
      routeStudyStrata = JSON.parse(JSON.stringify(
        (latestRouteReviewBundle || {}).source_study_plans || []
      ));
      renderRouteBundleInventory();
      renderRouteBundleVerification(result.verification || {});
      toast('ok', 'Review file verified by full source-bound reconstruction');
    } catch (err) {
      invalidateRouteReviewBundle(`Review import rejected: ${err.message}`);
      toast('err', err.message);
    } finally {
      input.value = '';
    }
  }

  function clearRouteReviewInventory() {
    routeStudyStrata = [];
    invalidateRouteReviewBundle(
      'Inventory cleared. Strata were client-side only; no ledger or memory records were written.'
    );
    renderRouteBundleInventory();
  }

  async function previewRouteStudy() {
    const button = el('routeStudyPreview');
    const text = el('prompt').value.trim();
    if (!text) {
      renderRouteStudyUnavailable('Enter a prompt first; rehearsal needs the final prompt-specific route support.');
      return;
    }
    if (agentMode !== 'auto') {
      renderRouteStudyUnavailable('Select Adaptive Router before rehearsing the adjacent-route study.');
      return;
    }
    button.disabled = true;
    el('routeStudyStatus').textContent = 'Rehearsing post-filter support without assigning or running a route...';
    try {
      const payload = buildRoutePayload(text);
      Object.assign(payload, {
        exploration_rate: Number(el('routeStudyEpsilon').value),
        planned_routes: parseInt(el('routeStudyHorizon').value, 10),
        scenario_confidence: 0.95,
        assumed_feedback_rate: Number(el('routeStudyResponseRate').value),
        target_observed_labels: parseInt(el('routeStudyTargetLabels').value, 10),
        target_policy_profile: el('routeProtocolTarget').value,
        protocol_design_mode: el('routeProtocolDesign').value,
        carryover_scope: el('routeProtocolCarryover').value,
        interference_scope: el('routeProtocolInterference').value,
        temporal_variation: el('routeProtocolTemporal').value,
        planned_clusters: parseInt(el('routeProtocolClusters').value, 10),
        max_routes_per_cluster: 20,
        analysis_every_clusters: Math.min(
          parseInt(el('routeProtocolClusters').value, 10), 50
        ),
        block_length_routes: parseInt(el('routeProtocolBlock').value, 10),
        washout_routes: parseInt(el('routeProtocolWashout').value, 10),
      });
      const result = await api('/api/route_study_plan', payload);
      renderRouteStudy(result);
      toast('ok', 'Adjacent-route rehearsal ready; execution stayed off');
    } catch (err) {
      renderRouteStudyUnavailable(`Study rehearsal unavailable: ${err.message}`);
      toast('err', err.message);
    } finally {
      button.disabled = false;
    }
  }

  async function sendRouteFeedback(rating, feedbackIntent, reason) {
    if (!lastRouteFeedback) return;
    const buttons = ['routeGoodBtn','routeBadBtn','routeDeeperBtn','routeCostBtn','routeSlowBtn']
      .map(id => el(id)).filter(Boolean);
    buttons.forEach(btn => btn.disabled = true);
    try {
      const result = await api('/api/route_feedback', {
        ...lastRouteFeedback,
        rating,
        feedback_intent: feedbackIntent,
        reason: reason || feedbackIntent,
      });
      const total = result.summary ? result.summary.total_feedback : '';
      setRouteFeedbackVisible(true, total ? `Route ${total}` : 'Route saved');
      if (result.summary) renderRouteHealth(result.summary);
      refreshPolicyLab();
      const message = feedbackIntent === 'needs_deeper' ? 'Deeper-route preference saved'
        : feedbackIntent === 'too_costly' ? 'Lower-cost preference saved'
        : feedbackIntent === 'too_slow' ? 'Lower-latency preference saved'
        : rating === 'up' ? 'Route preference saved' : 'Quality correction saved';
      toast('ok', message);
    } catch (err) {
      buttons.forEach(btn => btn.disabled = false);
      toast('err', err.message);
    }
  }

  // ── Typing indicator ─────────────────────────────────────────────────
  function buildRoutePayload(text) {
    const settings = {
      agent_mode: agentMode,
      auto_agent_budget: el('autoBudget').value,
      auto_session_budget_units: parseFloat(el('sessionBudget').value) || 0,
      auto_session_budget_target_routes: parseInt(el('sessionBudgetTargetRoutes').value) || 0,
      loop_max_steps: parseInt(el('loopBudget').value),
      memory_enabled: el('memToggle').value === 'on',
      web_search_enabled: el('webToggle').value === 'on',
      uploaded_image_path: currentUpload
    };
    const requestedCycles = el('reasoningCycles').value;
    const requestedAdaptive = el('adaptiveCompute').value;
    if (requestedCycles !== 'model') settings.reasoning_cycles = requestedCycles;
    if (requestedAdaptive !== 'model') settings.adaptive_compute = requestedAdaptive === 'on';
    return {
      session_id: sessionId,
      message: text,
      model_key: el('modelSelect').value,
      action_mode: 'text',
      settings
    };
  }

  el('routePlanBtn').onclick = async () => {
    const text = el('prompt').value.trim();
    if (!text) return;
    const btn = el('routePlanBtn');
    btn.disabled = true;
    try {
      const result = await api('/api/route_plan', buildRoutePayload(text));
      const estimate = result.route_economics_estimate || {};
      const cost = estimate.estimated_cost_units !== undefined ? `cost ${estimate.estimated_cost_units}` : 'cost -';
      const selected = result.selected_agent_mode || 'off';
      const frontier = result.route_frontier || {};
      const recommendation = frontier.recommended_agent_mode
        ? `rec ${frontier.recommended_agent_mode}${frontier.selected_matches_recommendation ? '' : ' != selected'}`
        : '';
      const recommendedQualityCost = frontier.recommended_estimated_quality_cost_score !== undefined
        && frontier.recommended_estimated_quality_cost_score !== null
        ? `rec qc ${frontier.recommended_estimated_quality_cost_score}`
        : '';
      const blocker = frontier.selected_budget_blocker || frontier.recommended_budget_blocker;
      const blockerText = blocker ? `block ${blocker}` : '';
      const cap = frontier.budget_cap_cost_units !== undefined && frontier.budget_cap_cost_units !== null
        ? `cap ${frontier.budget_cap_cost_units}`
        : '';
      const effective = frontier.effective_cap_cost_units !== undefined && frontier.effective_cap_cost_units !== null
        ? `effective ${frontier.effective_cap_cost_units}`
        : '';
      const remaining = frontier.remaining_cost_units !== undefined && frontier.remaining_cost_units !== null
        ? `remain ${frontier.remaining_cost_units}`
        : '';
      const pacing = frontier.pacing_cap_cost_units !== undefined && frontier.pacing_cap_cost_units !== null
        ? `pace ${frontier.pacing_cap_cost_units}`
        : '';
      const budgetPareto = Array.isArray(frontier.budget_feasible_pareto_modes) && frontier.budget_feasible_pareto_modes.length
        ? `bpareto ${frontier.budget_feasible_pareto_modes.join(',')}`
        : '';
      const alternatives = Array.isArray(result.route_alternatives)
        ? result.route_alternatives
          .slice()
          .sort((a,b) => Number(a.frontier_rank || 99) - Number(b.frontier_rank || 99))
          .slice(0, 4)
          .map(row => {
            const value = row.estimated_cost_units !== undefined ? row.estimated_cost_units : '-';
            const rank = row.frontier_rank ? `#${row.frontier_rank}` : '';
            const qualityCost = row.estimated_quality_cost_score !== undefined && row.estimated_quality_cost_score !== null
              ? `/qc${row.estimated_quality_cost_score}`
              : '';
            const evidence = row.quality_source === 'adaptive_feedback'
              ? '/adaptive'
              : (row.quality_evidence_status && row.quality_evidence_status !== 'heuristic_prior' ? `/${row.quality_evidence_status}` : '');
            const lowerRaw = row.estimated_quality_lower_bound;
            const upperRaw = row.estimated_quality_upper_bound;
            const lower = Number(lowerRaw);
            const upper = Number(upperRaw);
            const confidence = lowerRaw !== null && lowerRaw !== undefined
              && upperRaw !== null && upperRaw !== undefined
              && Number.isFinite(lower) && Number.isFinite(upper)
              ? `/q90:${Math.round(lower * 100)}-${Math.round(upper * 100)}`
              : '';
            const riskRaw = row.risk_adjusted_quality_cost_score;
            const riskQc = Number(riskRaw);
            const risk = riskRaw !== null && riskRaw !== undefined
              && Number.isFinite(riskQc) && row.confidence_status === 'established'
              ? `/riskqc${riskQc}`
              : '';
            const pareto = row.budget_feasible_pareto_frontier ? '/bpf' : (row.pareto_frontier ? '/pf' : '');
            return `${rank}${row.selected_agent_mode}${row.is_selected ? '*' : ''}:${value}${qualityCost}${evidence}${confidence}${risk}${pareto}`;
          })
          .join(' ')
        : '';
      const frontierBits = [
        recommendation,
        recommendedQualityCost,
        blockerText,
        cap,
        effective,
        remaining,
        pacing,
        budgetPareto,
        alternatives,
      ].filter(Boolean).join(' | ');
      addLoopStep(1, 'Route Plan', `${selected} | ${cost}${frontierBits ? ' | ' + frontierBits : ''}`, 'done');
      openPanelTab('mode');
      toast('ok', `Route plan: ${selected}`);
    } catch (err) {
      toast('err', err.message);
    } finally {
      btn.disabled = false;
    }
  };

  function addTyping() {
    const row = document.createElement('div');
    row.className = 'msg asst';
    row.id = 'typing';
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.textContent = 'SX';
    meta.append(av, document.createTextNode('V46 synthesis...'));
    row.appendChild(meta);
    const dots = document.createElement('div');
    dots.className = 'typing-dots bubble';
    dots.innerHTML = '<span></span><span></span><span></span>';
    row.appendChild(dots);
    el('thread').appendChild(row);
    scrollToBottom();
  }

  function removeTyping() {
    const t = el('typing');
    if (t) t.remove();
  }

  // ── Auto-resize textarea ─────────────────────────────────────────────
  const textarea = el('prompt');
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 360) + 'px';
  });
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el('sendBtn').click(); }
  });
  qsa('.quick-chip').forEach(btn => {
    btn.onclick = () => {
      textarea.value = btn.dataset.prompt || btn.textContent.trim();
      textarea.dispatchEvent(new Event('input'));
      textarea.focus();
    };
  });

  // ── Image upload ─────────────────────────────────────────────────────
  el('imgBtn').onclick = () => el('fileInput').click();
  el('fileInput').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const fd = new FormData();
      fd.append('session_id', sessionId);
      fd.append('file', file);
      const r = await fetch('/api/upload_image', { method:'POST', body:fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Upload failed');
      currentUpload = data.saved_path;
      currentUpUrl  = data.image_url;
      el('imgThumb').src  = data.image_url;
      el('imgName').textContent = file.name;
      el('uploadBar').style.display = 'flex';
      el('imgBtn').classList.add('on');
      toast('ok', 'Data artifact attached');
    } catch(err) { toast('err', err.message); }
  };

  el('clearUpBtn').onclick = () => {
    currentUpload = null; currentUpUrl = '';
    el('fileInput').value = '';
    el('uploadBar').style.display = 'none';
    el('imgBtn').classList.remove('on');
  };

  // ── Send ─────────────────────────────────────────────────────────────
  el('sendBtn').onclick = async () => {
    const text = textarea.value.trim();
    if (!text && !currentUpload) return;

    let extra = '';
    if (currentUpUrl) extra = `<img src="${currentUpUrl}" style="margin-top:10px;width:120px;height:120px;object-fit:cover;border-radius:12px;border:1px solid var(--border)">`;
    addMsg('user', text || 'Cognitive analysis of artifact.', null, extra);

    textarea.value = '';
    textarea.style.height = 'auto';
    el('sendBtn').disabled = true;
    addTyping();

    if (agentMode === 'auto' || agentMode === 'loop' || agentMode === 'collective_loop') {
      el('loopSteps').innerHTML = '';
      loopStep = 0;
      lastRouteFeedback = null;
      setRouteFeedbackVisible(false);
      addLoopStep(
        1,
        agentMode === 'auto' ? 'Adaptive Routing' : 'Target Initialization',
        agentMode === 'auto' ? 'Scoring task complexity...' : 'Constructing reasoning graph...',
        'active'
      );
      switchPtab('mode');
    }

    const payload = buildRoutePayload(text);

    currentUpload = null; currentUpUrl = '';
    el('uploadBar').style.display = 'none';
    el('imgBtn').classList.remove('on');

    try {
      const data = await api('/api/chat', payload);
      removeTyping();

      if (data.agent_trace && data.agent_trace.loop_steps) {
        el('loopSteps').innerHTML = '';
        data.agent_trace.loop_steps.forEach((s, i) => {
          const score = scorePct(s.review_score ?? s.loop_score);
          const bits = [];
          if (s.worker_excerpt) bits.push(s.worker_excerpt);
          if (score != null) bits.push(`Score ${score}%`);
          if (s.stop_decision === 'stop') bits.push(stopReasonLabel(s.stop_reason_code));
          addLoopStep(i+1, s.goal || `Phase ${i+1}`, bits.join(' | '), 'done');
        });
        finaliseLoopSteps();
      } else if (data.agent_trace && data.agent_trace.auto_agent_policy) {
        const p = data.agent_trace.auto_agent_policy;
        el('loopSteps').innerHTML = '';
        addLoopStep(1, `Auto selected ${p.selected_agent_mode || 'off'}`, (p.reasons || []).slice(0,3).join(' | '), 'done');
        finaliseLoopSteps();
      }

      if (data.agent_trace && data.agent_trace.auto_agent_policy) {
        const routeId = data.route_id || data.agent_trace.route_id || '';
        lastRouteFeedback = {
          session_id: sessionId,
          route_id: routeId,
        };
        setRouteFeedbackVisible(Boolean(routeId));
      } else {
        lastRouteFeedback = null;
        setRouteFeedbackVisible(false);
      }
      refreshRouteHealth();
      refreshPolicyLab();

      addMsg('assistant', data.response || '(Inference finalized)', data.agent_trace);
      updateStatus(data);
    } catch(err) {
      removeTyping();
      addMsg('assistant', 'System Fault: ' + err.message);
      toast('err', err.message);
    } finally {
      el('sendBtn').disabled = false;
    }
  };

  // ── Clear ─────────────────────────────────────────────────────────────
  el('clearBtn').onclick = async () => {
    try {
      await api('/api/clear', { session_id: sessionId });
      el('thread').innerHTML = '';
      el('loopSteps').innerHTML = '';
      lastRouteFeedback = null;
      setRouteFeedbackVisible(false);
      renderRouteHealth({});
      renderPolicyLab({});
      addMsg('assistant', 'Session memory cleared. Omni V46 is ready for the next message.');
      toast('ok', 'System memory purged');
    } catch(e) { toast('err', e.message); }
  };

  // ── Model init ────────────────────────────────────────────────────────
  async function initModels() {
    try {
      const data = await api('/api/catalog');
      const sel  = el('modelSelect');
      sel.innerHTML = '';
      catalogByKey = {};
      let preferred = null;
      let auto = null;
      (data.models || []).forEach(m => {
        catalogByKey[m.key] = m;
        const opt = document.createElement('option');
        opt.value = m.key;
        const score = m.common_overall_exact != null ? m.common_overall_exact : m.recipe_eval_accuracy;
        const scoreText = score != null ? ` - ${(Number(score) * 100).toFixed(1)}%` : '';
        opt.textContent = m.key === 'auto' ? 'Auto Router' : `${m.label}${scoreText}`;
        if (m.key === 'omni_collective_v46') preferred = opt;
        if (m.key === 'auto') auto = opt;
        sel.appendChild(opt);
      });
      if (preferred) preferred.selected = true;
      else if (auto) auto.selected = true;
      sel.onchange = () => updateStatus();
      updateStatus();
    } catch(e) { console.warn('Catalog failure', e); }
  }

  async function updateStatus(data) {
    try {
      const s = await api('/api/status');
      const st = s.status || {};
      const sel = el('modelSelect');
      const selected = catalogByKey[sel.value] || {};
      const selectedLabel = selected.label || (sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : '');
      const label = st.active_model_label || selectedLabel || 'Auto Router';
      const score = selected.common_overall_exact != null ? selected.common_overall_exact : selected.recipe_eval_accuracy;
      const scoreText = score != null ? `${(Number(score) * 100).toFixed(1)}%` : 'pending';
      const policy = data && data.agent_trace ? data.agent_trace.auto_agent_policy : null;
      const budgetText = policy && policy.budget_profile ? `/${policy.budget_profile}` : '';
      const modeText = policy ? `${agentMode}${budgetText} -> ${policy.selected_agent_mode || 'off'}` : agentMode;
      el('panelStatus').textContent =
        `model: ${label || '-'}\ndevice: ${st.device || '-'}\nbenchmark: ${scoreText}\nmode: ${modeText}`;
      el('activePill').textContent = String(label).toLowerCase().includes('v46') ? 'V46 Champion' : (label || 'Auto');
      const lowered = label.toLowerCase();
      const pillClass = lowered.includes('v46') ? ' v46' : (lowered.includes('v48') ? ' v48' : (lowered.includes('v47') ? ' v47' : ''));
      el('activePill').className = 'model-pill' + pillClass;
      const snap = el('modelSnapshot');
      if (snap) {
        const benchCount = selected.per_benchmark ? Object.keys(selected.per_benchmark).length : selected.benchmark_count;
        snap.innerHTML = `<strong>${escHtml(label || 'Selected model')}</strong><br>` +
          `Benchmark: ${escHtml(scoreText)}${benchCount ? ` across ${benchCount} suites` : ''}<br>` +
          `Source: ${escHtml(selected.selection_policy || selected.score_source || 'runtime catalog')}`;
      }
    } catch(_) {}
  }

  el('routeGoodBtn').onclick = () => sendRouteFeedback('up', 'good', 'satisfied');
  el('routeBadBtn').onclick = () => sendRouteFeedback('down', 'bad_quality', 'bad quality');
  el('routeDeeperBtn').onclick = () => sendRouteFeedback('down', 'needs_deeper', 'needs deeper reasoning');
  el('routeCostBtn').onclick = () => sendRouteFeedback('down', 'too_costly', 'too costly');
  el('routeSlowBtn').onclick = () => sendRouteFeedback('down', 'too_slow', 'too slow');
  el('policyLabRefresh').onclick = () => refreshPolicyLab();
  el('policyLabProfile').onchange = () => refreshPolicyLab();
  el('routeStudyPreview').onclick = () => previewRouteStudy();
  el('routeShadowRegistryRefresh').onclick = () => refreshRouteShadowRegistry();
  el('routeBundleAdd').onclick = () => addCurrentRouteStudyStratum();
  el('routeBundleBuild').onclick = () => buildRouteReviewBundle();
  el('routeBundleDownload').onclick = () => downloadRouteReviewBundle();
  el('routeBundleImport').onchange = event => importRouteReviewFile(event);
  el('routeBundleClear').onclick = () => clearRouteReviewInventory();
  [
    'routeProtocolTarget', 'routeProtocolDesign', 'routeProtocolCarryover',
    'routeProtocolInterference', 'routeProtocolTemporal', 'routeProtocolClusters',
    'routeProtocolBlock', 'routeProtocolWashout',
  ].forEach(id => el(id).addEventListener('change', () => {
    if (latestRouteReviewBundle) {
      invalidateRouteReviewBundle('Protocol declarations changed. Rebuild before review.');
    }
  }));
  renderRouteBundleInventory();

  // ── Benchmark tab ─────────────────────────────────────────────────────
  async function loadBenchData() {
    try {
      const r = await api('/api/benchmark');
      const nota  = el('benchNote');
      const img   = el('benchImg');
      const scores = el('benchScores');

      if (r.graph_b64) {
        img.src = 'data:image/png;base64,' + r.graph_b64;
        img.style.display = 'block';
        nota.style.display = 'none';
      } else {
        nota.textContent = 'No benchmark graph was found yet.';
      }

      if (r.models) {
        scores.innerHTML = '';
        const topMean = Math.max(...r.models.map(m => Number(m.mean) || 0), 0);
        r.models.forEach(m => {
          const raw = Number(m.mean) || 0;
          const pctNum = raw <= 1.001 ? raw * 100 : raw;
          const pct = Math.max(0, Math.min(100, pctNum)).toFixed(1);
          const loweredLabel = String(m.label || '').toLowerCase();
          const isTop = Math.abs((Number(m.mean) || 0) - topMean) < 0.001 || loweredLabel.includes('v46');
          const bar = document.createElement('div');
          bar.style.cssText = 'margin-bottom:10px';
          bar.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
            <span style="color:${isTop?'#86efac':'var(--text)'};font-weight:${isTop?700:400}">${escHtml(m.label)}</span>
            <span style="color:${isTop?'#86efac':'var(--muted)'}">${pct}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:4px;overflow:hidden">
            <div style="height:100%;width:${pct}%;background:${isTop?'#34d399':'var(--blue)'};border-radius:4px;transition:.6s"></div>
          </div>`;
          scores.appendChild(bar);
        });
      }
    } catch(e) {
      el('benchNote').textContent = 'Benchmark data unavailable.';
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────
  initModels();
  refreshRouteHealth();
  refreshPolicyLab();
  refreshRouteShadowRegistry();
})();
</script>

<!-- ═══ Studio X Discovery & Compose Scaffold ═══════════════════════════ -->
<div id="appShell" style="display:none">

  <!-- Discovery Panel -->
  <input  id="modelSearch" type="text" placeholder="Search models…">
  <select id="capabilityFilter"><option value="">All</option></select>
  <div id="quickPickChips"></div>
  <div id="discoveryNote"></div>

  <!-- Session Planner -->
  <textarea id="sessionObjective" placeholder="Session objective…"></textarea>
  <div id="deliverableTarget"></div>
  <div id="successChecks"></div>
  <div id="riskBox"></div>

  <!-- Drafts & Context Bank -->
  <div id="savedDrafts"></div>
  <div id="contextBankList"></div>
  <button id="captureLastReplyBtn">Capture</button>

  <!-- Thread Navigation -->
  <div id="threadBookmarks"></div>
  <div id="compareSummary"></div>
  <div id="dispatchPreview"></div>

  <!-- Model Store Panel -->
  <div id="modelStoreList"></div>
  <button id="refreshStoreBtn">Refresh Store</button>

  <!-- Compose Toolbar -->
  <div id="composeScroll">
    <button id="composeQuickBtn">Quick</button>
    <button id="composeMediaBtn">Media</button>
    <button id="composeWorkbenchBtn">Workbench</button>
  </div>

  <!-- Mode Selector with Loop Agent modes -->
  <select id="modeSelector">
    <option value="text">Text</option>
    <option value="auto">Auto Router</option>
    <option value="loop">Loop Agent</option>
    <option value="collective_loop">Collective + Loop</option>
  </select>

  <!-- Layout Controls -->
  <button id="toggleSidebarBtn">Toggle Sidebar</button>
  <button id="toggleThreadDensityBtn">Density</button>

  <!-- Response Deck -->
  <div id="responseDeck"></div>

  <!-- Structured Reasoning Controls -->
  <div id="confidenceMode"></div>
  <div id="evidenceMode"></div>
  <div id="clarifyMode"></div>
  <div id="assumptionMode"></div>

  <!-- Refinement Deck -->
  <div id="refinementDeck">
    <button id="refineLastReplyBtn">Refine</button>
    <button id="challengeLastReplyBtn">Challenge</button>
  </div>

</div>

</body>
</html>
"""

# ─── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/api/status")
def api_status():
    return jsonify({"status": manager.status()})

@app.route("/api/catalog")
def api_catalog():
    from multimodel_catalog import models_to_json
    return jsonify({"models": models_to_json(manager.records)})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    try:
        result = manager.handle_prompt(
            session_id=data.get("session_id", "default"),
            prompt=data.get("message", ""),
            model_key=data.get("model_key", "auto"),
            action_mode=data.get("action_mode", "text"),
            settings=data.get("settings", {})
        )
        if hasattr(result, "to_dict"):
            return jsonify(result.to_dict())
        return jsonify(result)
    except Exception as exc:
        logging.exception("Chat request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/route_plan", methods=["POST"])
def api_route_plan():
    data = request.json or {}
    try:
        result = manager.preview_route_plan(
            session_id=data.get("session_id", "default"),
            prompt=data.get("message", ""),
            model_key=data.get("model_key", "auto"),
            action_mode=data.get("action_mode", "text"),
            settings=data.get("settings", {}),
        )
        return jsonify(result)
    except (KeyError, ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route plan request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_study_plan", methods=["POST"])
def api_route_study_plan():
    data = request.json or {}
    try:
        result = manager.preview_route_study(
            session_id=data.get("session_id", "default"),
            prompt=data.get("message", ""),
            model_key=data.get("model_key", "auto"),
            action_mode=data.get("action_mode", "text"),
            settings=data.get("settings", {}),
            exploration_rate=data.get("exploration_rate", 0.10),
            planned_routes=data.get("planned_routes", 2_000),
            scenario_confidence=data.get("scenario_confidence", 0.95),
            assumed_feedback_rate=data.get("assumed_feedback_rate", 0.30),
            target_observed_labels=data.get("target_observed_labels", 20),
            target_policy_profile=data.get("target_policy_profile", "balanced"),
            protocol_design_mode=data.get(
                "protocol_design_mode", "sticky_session_cluster"
            ),
            carryover_scope=data.get("carryover_scope", "unknown"),
            interference_scope=data.get("interference_scope", "unknown"),
            temporal_variation=data.get("temporal_variation", "unknown"),
            planned_clusters=data.get("planned_clusters", 200),
            max_routes_per_cluster=data.get("max_routes_per_cluster", 20),
            analysis_every_clusters=data.get("analysis_every_clusters", 50),
            block_length_routes=data.get("block_length_routes", 20),
            washout_routes=data.get("washout_routes", 0),
        )
        return jsonify(result)
    except (KeyError, ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route study rehearsal failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_study_protocol_bundle", methods=["POST"])
def api_route_study_protocol_bundle():
    try:
        data = _read_strict_route_review_json()
        if not isinstance(data, dict):
            raise ValueError("route protocol review build input must be a JSON object")
        _validate_route_review_request_size(data)
        _validate_route_review_strata(data.get("study_plans"))
        return jsonify(manager.build_route_protocol_review_bundle(data))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route protocol review bundle build failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_study_protocol_bundle/audit", methods=["POST"])
def api_route_study_protocol_bundle_audit():
    try:
        data = _read_strict_route_review_json()
        if not isinstance(data, dict) or set(data) != {"bundle"}:
            raise ValueError("route protocol review audit accepts only a bundle object")
        _validate_route_review_request_size(data)
        bundle = data.get("bundle")
        if not isinstance(bundle, dict):
            raise ValueError("route protocol review bundle must be a JSON object")
        _validate_route_review_strata(bundle.get("source_study_plans"))
        return jsonify(manager.audit_route_protocol_review_bundle(bundle))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route protocol review bundle audit failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_feedback", methods=["POST"])
def api_route_feedback():
    data = request.json or {}
    session_id = str(data.get("session_id") or "default")
    try:
        if not str(data.get("route_id") or "").strip():
            raise ValueError("route_id is required for route feedback")
        result = manager.record_route_feedback(
            session_id=session_id,
            feedback={
                "route_id": data.get("route_id") or "",
                "rating": data.get("rating") or "",
                "feedback_intent": data.get("feedback_intent") or data.get("intent") or "",
                "feedback_tags": data.get("feedback_tags") or [],
                "reason": data.get("reason") or "",
            },
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route feedback request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/route_health", methods=["POST"])
def api_route_health():
    data = request.json or {}
    session_id = str(data.get("session_id") or "default")
    try:
        return jsonify({"ok": True, "route_health": manager.route_health_snapshot(session_id)})
    except Exception as exc:
        logging.exception("Route health request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_policy_lab", methods=["POST"])
def api_route_policy_lab():
    data = request.json or {}
    session_id = str(data.get("session_id") or "default")
    profile = str(data.get("profile") or "balanced")
    try:
        return jsonify(
            {
                "ok": True,
                "policy_lab": manager.route_policy_lab_snapshot(session_id, profile=profile),
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Route policy lab request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/route_shadow_registry/status", methods=["GET"])
def api_route_shadow_registry_status():
    try:
        response = jsonify(
            {
                "ok": True,
                "route_shadow_registry": manager.route_shadow_registry_snapshot(),
            }
        )
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:
        logging.exception("Route shadow registry status failed")
        response = jsonify({"ok": False, "error": str(exc)})
        response.headers["Cache-Control"] = "no-store"
        return response, 500


@app.route("/api/clear", methods=["POST"])
def api_clear():
    data = request.json or {}
    manager.clear(data.get("session_id", "default"))
    return jsonify({"ok": True})

@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    session_id = request.form.get("session_id", "default")
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    raw_bytes = file.read()
    filename = file.filename or "upload.png"
    result = manager.store_uploaded_image(
        session_id=session_id, filename=filename, raw_bytes=raw_bytes
    )
    return jsonify(result)

@app.route("/api/benchmark")
def api_benchmark():
    """Serve the benchmark graph (base64) and scores JSON."""
    b64 = _bench_graph_b64()
    models = []
    source = ""
    generated_at = ""
    benchmark_path = _latest_benchmark_json_path()
    if benchmark_path:
        try:
            data = json.loads(benchmark_path.read_text(encoding="utf-8"))
            models = _benchmark_rows_for_ui(data)
            source = str(benchmark_path)
            generated_at = data.get("created_at", "")
        except Exception:
            logging.exception("Benchmark JSON load failed")
    return jsonify(
        {
            "graph_b64": b64,
            "models": models,
            "source": source,
            "generated_at": generated_at,
        }
    )

@app.route("/uploads/<session_slug>/<filename>")
def serve_upload(session_slug, filename):
    safe_slug = "".join(c for c in session_slug if c.isalnum() or c in ("-", "_"))
    return send_from_directory(manager.uploads_dir / safe_slug, filename)


# ─── 3D Model View & Downloads ───────────────────────────────────────────────

@app.route("/api/three_d_model_view")
def api_three_d_model_view():
    try:
        model_data = manager.three_d_model_view()
        model_data["download_zip_url"] = "/download/three_d_model_zip"
        model_data["download_summary_url"] = "/download/three_d_model_summary"
        return jsonify({"ok": True, "model": model_data})
    except Exception as exc:
        logging.exception("3D model view failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/download/three_d_model_zip")
def download_three_d_model_zip():
    try:
        model_data = manager.three_d_model_view()
        file_path = Path(model_data["zip_path"])
        return send_from_directory(str(file_path.parent), file_path.name, as_attachment=True)
    except Exception as exc:
        logging.exception("3D zip download failed")
        return str(exc), 500


@app.route("/download/three_d_model_summary")
def download_three_d_model_summary():
    try:
        model_data = manager.three_d_model_view()
        file_path = Path(model_data["summary_path"])
        return send_from_directory(str(file_path.parent), file_path.name, as_attachment=True)
    except Exception as exc:
        logging.exception("3D summary download failed")
        return str(exc), 500


# ─── Model Store API ─────────────────────────────────────────────────────────

@app.route("/api/model_store")
def api_model_store():
    try:
        force_refresh = request.args.get("force_refresh", "false").lower() == "true"
        catalog = manager.model_store_catalog(force_refresh=force_refresh)
        return jsonify({"ok": True, **catalog})
    except Exception as exc:
        logging.exception("Model store catalog failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/model_store/jobs")
def api_model_store_jobs():
    try:
        jobs_data = manager.model_store_jobs()
        return jsonify({"ok": True, **jobs_data})
    except Exception as exc:
        logging.exception("Model store jobs failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/model_store/install", methods=["POST"])
def api_model_store_install():
    try:
        data = request.json or {}
        file_name = data.get("file_name")
        if not file_name:
            return jsonify({"ok": False, "error": "file_name is required"}), 400
        job = manager.install_model_store_artifact(file_name)
        return jsonify({"ok": True, "job": job})
    except Exception as exc:
        logging.exception("Model store install failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supermix Studio X - V46 20-Suite Champion")
    parser.add_argument("--port",   type=int, default=5000, help="Port to listen on")
    parser.add_argument("--models", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Path to local models directory")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Interface to bind (default: loopback only; opt in explicitly for remote access)",
    )
    return parser


def main():
    global manager
    args = build_arg_parser().parse_args()

    manager = UnifiedModelManager(
        records=discover_model_records(Path(args.models)),
        extraction_root=Path("tmp/ext"),
        generated_dir=Path("tmp/gen"),
        models_dir=Path(args.models),
    )
    print(f"[Supermix Studio X] starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

