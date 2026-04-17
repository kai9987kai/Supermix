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

def build_app(unified_manager: UnifiedModelManager) -> Flask:
    global manager
    manager = unified_manager
    return app


# ─── Benchmark graph embed helper ───────────────────────────────────────────
def _bench_graph_b64() -> str:
    """Return base64-encoded PNG of the benchmark graph, or empty string."""
    candidates = [
        Path(__file__).parent.parent / "output" / "v48_benchmark_comparison.png",
        Path("output") / "v48_benchmark_comparison.png",
        Path(__file__).parent.parent / "output" / "v47_benchmark_comparison.png",
        Path("output") / "v47_benchmark_comparison.png",
    ]
    for p in candidates:
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""

# ─── HTML / CSS / JS ─────────────────────────────────────────────────────────
_BENCH_B64 = _bench_graph_b64()

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Studio X - v48 Frontier</title>
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

    /* ── Composer ───────────────────────────────────────────────────── */
    .compose-wrap { padding:0 14% 32px; flex-shrink: 0; z-index: 20;
                    background:linear-gradient(transparent,rgba(2,6,16,.9) 50%); }
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
    .mode-card.on[data-mode="collective"] { background:rgba(45,212,191,.08);
                                             border-color:rgba(45,212,191,.5); }
    .mode-card.on[data-mode="loop"] { background:rgba(245,158,11,.08);
                                       border-color:rgba(245,158,11,.5); }
    .mc-title { font-size:14.5px; font-weight:800; margin-bottom:6px;
                display:flex; align-items:center; gap:10px; }
    .mc-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
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
        <div class="model-pill v48" id="activePill">v48 Frontier</div>
        <div class="model-pill" id="modePill" style="display:none">Standard</div>
      </div>
      <div class="wk-actions" id="wkActions"></div>
    </header>

    <div class="thread" id="thread">
      <div class="msg asst">
        <div class="msg-meta">
          <div class="msg-avatar">SX</div>
          <span>Supermix Studio X</span>
        </div>
        <div class="bubble"><strong>v48 Frontier is online.</strong>
Powered by Adaptive Graph-of-Thoughts, Hierarchical Mixture-of-Experts routing, and frontier-tuned multimodal reasoning.

Select an operational mode in the right panel to begin.</div>
      </div>
    </div>

    <!-- Upload preview bar -->
    <div class="upload-bar" id="uploadBar">
      <img id="imgThumb" src="" alt="">
      <div class="up-name" id="imgName">image.png</div>
      <button class="up-rm" id="clearUpBtn" title="Remove">&#x2715;</button>
    </div>

    <div class="compose-wrap">
      <div class="compose-box">
        <textarea id="prompt" rows="1" placeholder="Type a message to Studio X..."></textarea>
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
  <aside class="panel">
    <div class="panel-tabs">
      <button class="ptab on" data-ptab="model">Model</button>
      <button class="ptab" data-ptab="mode">Mode</button>
      <button class="ptab" data-ptab="bench">Bench</button>
    </div>

    <!-- MODEL tab -->
    <div class="panel-body" id="ptab-model">
      <div class="panel-section">
        <h4>Active Model</h4>
        <select id="modelSelect"></select>
      </div>
      <div class="panel-section">
        <h4>v48 Frontier Series</h4>
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
          <div class="mode-card on" data-mode="off">
            <div class="mc-title"><div class="mc-dot std"></div>Standard Case</div>
            <div class="mc-desc">Optimal for direct queries and creative generation. High-speed single-pass.</div>
          </div>
          <div class="mode-card" data-mode="collective">
            <div class="mc-title"><div class="mc-dot col"></div>Collective Synthesis</div>
            <div class="mc-desc">Ensemble reasoning. V48 consults sub-experts before delivering a unified response.</div>
          </div>
          <div class="mode-card" data-mode="loop">
            <div class="mc-title"><div class="mc-dot loop"></div>Autonomous Frontier</div>
            <div class="mc-desc">Recursive loop for complex workflows. Self-correcting multi-step planner.</div>
          </div>
        </div>
      </div>
      <div class="panel-section" id="loopPanel" style="display:none">
        <h4>Loop Observation</h4>
        <div class="loop-steps" id="loopSteps"></div>
      </div>
    </div>

    <!-- BENCH tab -->
    <div class="panel-body" id="ptab-bench" style="display:none">
      <div class="panel-section">
        <h4>V48 Comparative Benchmarks</h4>
        <div class="bench-wrap" id="benchWrap">
          <img id="benchImg" src="" alt="Benchmark comparison" style="display:none">
          <div class="bench-note" id="benchNote">Initializing frontier telemetry...</div>
        </div>
        <div style="margin-top:24px;display:flex;flex-direction:column;gap:12px" id="benchScores"></div>
      </div>
    </div>

    <div class="panel-footer" id="panelStatus">system: active  |  accelerator: auto  |  v: 48.0.0</div>
  </aside>
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

  let agentMode   = 'off';
  let currentUpload = null;
  let currentUpUrl  = '';
  let loopStep = 0;

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

  // ── Tabs (rail) ─────────────────────────────────────────────────────
  qsa('.rail-item[data-tab]').forEach(btn => {
    btn.onclick = () => {
      qsa('.rail-item[data-tab]').forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      const ptab = btn.dataset.tab === 'bench' ? 'bench'
                 : btn.dataset.tab === 'settings' ? 'model'
                 : 'model';
      switchPtab(ptab);
    };
  });

  // ── Panel tabs ───────────────────────────────────────────────────────
  qsa('.ptab').forEach(btn => {
    btn.onclick = () => switchPtab(btn.dataset.ptab);
  });

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
      el('loopPanel').style.display = agentMode==='loop' ? 'block' : 'none';
      el('loopSteps').innerHTML = '';
      loopStep = 0;
      updateModePill();
      toast('ok', `Switched to ${agentMode} mode`);
    };
  });

  function updateModePill() {
    const pill = el('modePill');
    const labels = { off:'Standard', collective:'Collective', loop:'Autonomous' };
    if (agentMode === 'off') { pill.style.display='none'; return; }
    pill.style.display='block';
    pill.textContent = labels[agentMode] || agentMode;
    pill.style.color = agentMode==='collective' ? 'var(--teal)' : 'var(--amber)';
    pill.style.borderColor = agentMode==='collective'
      ? 'rgba(45,212,191,.4)' : 'rgba(245,158,11,.4)';
    pill.style.background = agentMode==='collective'
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
    meta.append(av, document.createTextNode(role==='user' ? 'Frontier User' : 'Studio X - V48'));
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

  function buildTrace(trace) {
    const wrapper = document.createElement('div');
    wrapper.className = 'trace';

    if (trace.loop_steps && trace.loop_steps.length) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--amber)';
      hdr.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12,4V1L8,5l4,4V6c3.31,0,6,2.69,6,6a5.987,5.987,0,0,1-.7,2.8l1.46,1.46A7.93,7.93,0,0,0,20,12C20,7.58,16.42,4,12,4Zm0,14c-3.31,0-6-2.69-6-6a5.987,5.987,0,0,1,.7-2.8L5.24,7.74A7.93,7.93,0,0,0,4,12c0,4.42,3.58,8,8,8v3l4-4-4-4Z"/></svg> Autonomous Logic Chain — ${trace.loop_steps.length} cycles`;
      wrapper.appendChild(hdr);

      const body = document.createElement('div');
      body.className = 'trace-body';
      trace.loop_steps.forEach(s => {
        body.innerHTML += `<div class="trace-step">
          <div class="trace-step-n">${s.step}</div>
          <div><strong style="color:var(--text)">${escHtml(s.goal||'Strategy Initialization')}</strong><br>
          <span style="color:var(--muted);font-size:11px">${escHtml((s.worker_excerpt||'').slice(0,140))}...</span></div>
        </div>`;
      });
      wrapper.appendChild(body);

    } else if (trace.reasoning_passes != null) {
      const hdr = document.createElement('div');
      hdr.className = 'trace-hdr';
      hdr.style.color = 'var(--teal)';
      hdr.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12,2A10,10,0,1,0,22,12,10.011,10.011,0,0,0,12,2Zm1,15H11V11h2Zm0-8H11V7h2Z"/></svg> V48 Reasoning Telemetry`;
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
      body.innerHTML = '<span style="color:var(--muted)">Expert weights synthesized from:</span> ' + trace.consulted_models.join(', ');
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

  // ── Typing indicator ─────────────────────────────────────────────────
  function addTyping() {
    const row = document.createElement('div');
    row.className = 'msg asst';
    row.id = 'typing';
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.textContent = 'SX';
    meta.append(av, document.createTextNode('Studio X Synthesis...'));
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

    if (agentMode === 'loop') {
      el('loopSteps').innerHTML = '';
      loopStep = 0;
      addLoopStep(1, 'Target Initialization', 'Constructing reasoning graph...', 'active');
      switchPtab('mode');
    }

    const payload = {
      session_id: sessionId,
      message: text,
      model_key: el('modelSelect').value,
      action_mode: 'text',
      settings: {
        agent_mode: agentMode,
        loop_max_steps: parseInt(el('loopBudget').value),
        memory_enabled: el('memToggle').value === 'on',
        web_search_enabled: el('webToggle').value === 'on',
        uploaded_image_path: currentUpload
      }
    };

    currentUpload = null; currentUpUrl = '';
    el('uploadBar').style.display = 'none';
    el('imgBtn').classList.remove('on');

    try {
      const data = await api('/api/chat', payload);
      removeTyping();

      if (agentMode === 'loop' && data.agent_trace && data.agent_trace.loop_steps) {
        el('loopSteps').innerHTML = '';
        data.agent_trace.loop_steps.forEach((s, i) => {
          addLoopStep(i+1, s.goal || `Phase ${i+1}`, s.worker_excerpt || '', 'done');
        });
        finaliseLoopSteps();
      }

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
      addMsg('assistant', 'Inference memory cleared. Ready for next frontier case.');
      toast('ok', 'System memory purged');
    } catch(e) { toast('err', e.message); }
  };

  // ── Model init ────────────────────────────────────────────────────────
  async function initModels() {
    try {
      const data = await api('/api/catalog');
      const sel  = el('modelSelect');
      (data.models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.key;
        opt.textContent = m.key === 'auto' ? '⌘ Auto Router'
          : `${m.label}${m.recipe_eval_accuracy ? ' — ' + (m.recipe_eval_accuracy*100).toFixed(1) + '%' : ''}`;
        if (m.key === 'auto') opt.selected = true;
        sel.appendChild(opt);
      });
    } catch(e) { console.warn('Catalog failure', e); }
  }

  async function updateStatus(data) {
    try {
      const s = await api('/api/status');
      const st = s.status || {};
      el('panelStatus').textContent =
        `model: ${st.active_model_label || '—'}\ndevice: ${st.device || '—'}\nmode: ${agentMode}`;
      const label = st.active_model_label || '';
      el('activePill').textContent = label || 'Auto';
      const lowered = label.toLowerCase();
      const pillClass = lowered.includes('v48') ? ' v48' : (lowered.includes('v47') ? ' v47' : '');
      el('activePill').className = 'model-pill' + pillClass;
    } catch(_) {}
  }

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
        nota.textContent = 'Run python source/benchmark_v48.py to generate the graph.';
      }

      if (r.models) {
        scores.innerHTML = '';
        const topMean = Math.max(...r.models.map(m => Number(m.mean) || 0), 0);
        r.models.forEach(m => {
          const pct = (m.mean * 1).toFixed(1);
          const loweredLabel = String(m.label || '').toLowerCase();
          const isTop = Math.abs((Number(m.mean) || 0) - topMean) < 0.001 || loweredLabel.includes('v48');
          const bar = document.createElement('div');
          bar.style.cssText = 'margin-bottom:10px';
          bar.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
            <span style="color:${isTop?'#f9a8d4':'var(--text)'};font-weight:${isTop?700:400}">${escHtml(m.label)}</span>
            <span style="color:${isTop?'#f9a8d4':'var(--muted)'}">${pct}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:4px;overflow:hidden">
            <div style="height:100%;width:${pct}%;background:${isTop?'#f472b6':'var(--blue)'};border-radius:4px;transition:.6s"></div>
          </div>`;
          scores.appendChild(bar);
        });
      }
    } catch(e) {
      el('benchNote').textContent = 'Benchmark data unavailable. Run benchmark_v48.py first.';
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────
  initModels();
  updateStatus();
})();
</script>
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
    scores_path_candidates = [
        Path(__file__).parent.parent / "output" / "v48_benchmark_results.json",
        Path("output") / "v48_benchmark_results.json",
        Path(__file__).parent.parent / "output" / "v47_benchmark_results.json",
        Path("output") / "v47_benchmark_results.json",
    ]
    models = []
    for p in scores_path_candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                models = data.get("models", [])
            except Exception:
                pass
            break
    return jsonify({"graph_b64": b64, "models": models})

@app.route("/uploads/<session_slug>/<filename>")
def serve_upload(session_slug, filename):
    safe_slug = "".join(c for c in session_slug if c.isalnum() or c in ("-", "_"))
    return send_from_directory(manager.uploads_dir / safe_slug, filename)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    global manager
    parser = argparse.ArgumentParser(description="Supermix Studio X - v48 Frontier")
    parser.add_argument("--port",   type=int, default=5000, help="Port to listen on")
    parser.add_argument("--models", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Path to local models directory")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    args = parser.parse_args()

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

