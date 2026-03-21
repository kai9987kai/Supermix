import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flask import Flask, jsonify, request
from peft import PeftConfig, get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer


DEFAULT_BASE_MODEL = (
    r"C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_BASE_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_OVERRIDE_ENV = "SUPERMIX_QWEN_BASE_MODEL_DIR"
MODEL_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+$")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
DEFAULT_SYSTEM_PROMPT = (
    "You are Supermix Qwen v26, a practical local assistant. "
    "Answer directly, stay grounded in the user's request, avoid bracketed training tags, "
    "and do not narrate hidden reasoning."
)
PRESET_HINTS = {
    "direct": "Be brief and decisive.",
    "balanced": "Give a direct answer with only the detail needed to be useful.",
    "reasoning": "When analysis matters, explain the answer in a compact stepwise structure.",
    "creative": "Use more vivid language and richer examples when the prompt invites it.",
    "coding": "Prioritize correctness, concrete debugging steps, and executable code.",
}
PRESET_GENERATION = {
    "direct": {"max_new_tokens": 72, "temperature": 0.05, "top_p": 0.82},
    "balanced": {"max_new_tokens": 112, "temperature": 0.20, "top_p": 0.92},
    "reasoning": {"max_new_tokens": 176, "temperature": 0.14, "top_p": 0.90},
    "creative": {"max_new_tokens": 200, "temperature": 0.58, "top_p": 0.97},
    "coding": {"max_new_tokens": 160, "temperature": 0.12, "top_p": 0.88},
}
VALID_PRESETS = tuple(PRESET_GENERATION.keys())

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Qwen v26</title>
  <style>
    :root{
      --bg:#08121e;--panel:rgba(12,24,38,.9);--panel-2:rgba(16,30,46,.96);--panel-3:rgba(9,18,29,.92);--line:rgba(151,180,214,.18);
      --text:#ecf3fb;--muted:#9db2cc;--blue:#68b4ff;--amber:#ffb163;--up:#8dd8a8;--down:#f2b3b3;--rose:#ff8f8f;
      --r-xl:26px;--r-lg:18px;--r-md:14px;--r-sm:10px;--shadow:0 32px 80px rgba(0,0,0,.35);
    }
    *{box-sizing:border-box} html,body{height:100%}
    body{
      margin:0;color:var(--text);font-family:"Aptos","Segoe UI Variable Text","Segoe UI",sans-serif;
      background:
        radial-gradient(circle at 14% 12%,rgba(104,180,255,.18),transparent 30%),
        radial-gradient(circle at 88% 84%,rgba(255,177,99,.16),transparent 32%),
        linear-gradient(160deg,#06101a 0%,var(--bg) 48%,#08101a 100%);
      overflow:hidden
    }
    .shell{display:grid;grid-template-columns:390px 1fr;gap:20px;width:min(1480px,calc(100vw - 28px));height:calc(100vh - 28px);margin:14px auto}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-xl);box-shadow:var(--shadow);backdrop-filter:blur(18px);overflow:hidden}
    .rail{display:grid;grid-template-rows:auto auto auto auto 1fr;gap:14px;padding:18px;overflow-y:auto}
    .rail::-webkit-scrollbar,.thread::-webkit-scrollbar,.status-box::-webkit-scrollbar{width:8px}
    .rail::-webkit-scrollbar-thumb,.thread::-webkit-scrollbar-thumb,.status-box::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:999px}
    .hero,.card{border-radius:var(--r-lg);border:1px solid var(--line);background:var(--panel-2)}
    .hero{padding:18px;background:linear-gradient(140deg,rgba(25,50,79,.94),rgba(13,24,37,.97))}
    .eyebrow{display:inline-flex;align-items:center;gap:10px;color:var(--blue);font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;margin-bottom:12px}
    .eyebrow::before{content:"";width:10px;height:10px;border-radius:999px;background:var(--blue);box-shadow:0 0 16px rgba(104,180,255,.8)}
    h1{margin:0;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif;font-size:32px;line-height:1.04}
    .hero p,.card p{margin:12px 0 0;color:var(--muted);line-height:1.55;font-size:14px}
    .release-strip,.preset-row,.chip-row,.action-row{display:flex;flex-wrap:wrap;gap:8px}
    .release-strip{margin-top:14px}
    .pill,.preset-btn,.chip,.ghost-btn,.primary-btn{
      display:inline-flex;align-items:center;gap:8px;padding:9px 12px;border-radius:999px;border:1px solid var(--line);
      background:rgba(255,255,255,.04);color:var(--text);cursor:pointer;transition:transform .14s ease,border-color .18s ease,background .18s ease
    }
    .pill{font-size:12px}
    .preset-btn.active{border-color:rgba(104,180,255,.48);background:rgba(104,180,255,.12)}
    .primary-btn{background:linear-gradient(135deg,#2d74c4,var(--blue));font-weight:700}
    .card{padding:16px}
    .card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
    .command-hint{margin:12px 0 0;color:var(--muted);font-size:12px;line-height:1.6}
    .metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
    .metric{padding:12px;border-radius:var(--r-md);background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.05)}
    .metric .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}
    .metric .v{font-family:"Bahnschrift","Segoe UI Semibold",sans-serif;font-size:26px;line-height:1}
    .metric .d{margin-top:7px;font-size:12px;color:var(--muted)}
    .up{color:var(--up)} .down{color:var(--down)}
    .control-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
    .field{display:grid;gap:6px;margin-bottom:12px}
    .field label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;font-weight:700}
    input,textarea{width:100%;color:var(--text);background:rgba(3,10,17,.8);border:1px solid var(--line);border-radius:var(--r-sm);padding:12px 13px;font:inherit}
    textarea{resize:vertical;min-height:88px;max-height:220px;line-height:1.45}
    input:focus,textarea:focus{outline:none;border-color:rgba(104,180,255,.56);box-shadow:0 0 0 3px rgba(104,180,255,.12)}
    .status-box{
      margin-top:12px;max-height:180px;overflow:auto;padding:12px;border-radius:var(--r-md);background:rgba(2,9,15,.82);
      border:1px solid rgba(255,255,255,.05);color:#bed0e5;font-family:Consolas,"Cascadia Code",monospace;font-size:12px;line-height:1.5;white-space:pre-wrap
    }
    .chat{display:grid;grid-template-rows:auto auto 1fr auto;min-height:0}
    .chat-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;padding:18px 22px;border-bottom:1px solid var(--line);background:rgba(15,29,45,.96)}
    .chat-head-main{min-width:0}
    .chat-head-side{display:grid;gap:10px;justify-items:end}
    .chat-head h3{margin:0;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif;font-size:24px;line-height:1.05}
    .chat-sub{margin-top:8px;color:var(--muted);font-size:13px;line-height:1.55}
    .runtime-pill{justify-self:end;border-color:rgba(255,177,99,.22);background:rgba(255,177,99,.08);color:#ffd4ab}
    .summary-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;padding:14px 22px;border-bottom:1px solid var(--line);background:rgba(10,21,33,.82)}
    .summary-card{padding:12px 14px;border-radius:var(--r-md);border:1px solid rgba(255,255,255,.06);background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02))}
    .summary-k{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700}
    .summary-v{margin-top:7px;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif;font-size:24px;line-height:1}
    .summary-d{margin-top:8px;font-size:12px;line-height:1.45;color:var(--muted)}
    .thread{min-height:0;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:16px;background:
      radial-gradient(circle at top right,rgba(104,180,255,.08),transparent 28%),
      linear-gradient(180deg,rgba(6,12,20,.45),rgba(10,19,31,.88))}
    .welcome{padding:18px;border-radius:var(--r-lg);border:1px solid rgba(104,180,255,.12);background:rgba(104,180,255,.06);color:var(--muted);line-height:1.65}
    .msg{max-width:min(860px,84%);border-radius:20px;padding:15px 16px 14px;border:1px solid var(--line);background:rgba(255,255,255,.03);box-shadow:0 10px 26px rgba(0,0,0,.18)}
    .msg.user{align-self:flex-end;background:linear-gradient(145deg,rgba(34,93,158,.92),rgba(18,53,91,.94));border-color:rgba(104,180,255,.32)}
    .msg.bot{align-self:flex-start}
    .msg.pending{opacity:.72;border-style:dashed}
    .msg-top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px}
    .who{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700}
    .msg-time{font-size:12px;color:var(--muted)}
    .body{display:grid;gap:12px;line-height:1.62;font-size:15px}
    .body p,.body blockquote{margin:0}
    .body ul,.body ol{margin:0;padding-left:22px}
    .body li+li{margin-top:6px}
    .body code{font-family:Consolas,"Cascadia Code",monospace;font-size:.92em;padding:2px 6px;border-radius:8px;background:rgba(255,255,255,.08)}
    .code-block{border-radius:16px;border:1px solid rgba(104,180,255,.12);background:rgba(2,8,14,.88);overflow:hidden}
    .code-head{padding:8px 12px;border-bottom:1px solid rgba(104,180,255,.10);font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#a7c3e6;background:rgba(255,255,255,.03)}
    .code-block pre{margin:0;padding:14px 16px;overflow-x:auto}
    .code-block pre code{display:block;padding:0;border-radius:0;background:none}
    .body blockquote{padding:10px 14px;border-left:3px solid rgba(104,180,255,.42);background:rgba(104,180,255,.06);border-radius:0 12px 12px 0;color:#d8e7f9}
    .meta{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:12px;color:var(--muted)}
    .msg-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px;opacity:0;transition:opacity .16s ease}
    .msg:hover .msg-actions,.msg:focus-within .msg-actions{opacity:1}
    .mini-btn,.followup-btn{
      display:inline-flex;align-items:center;gap:8px;padding:8px 11px;border-radius:999px;border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.04);color:var(--text);cursor:pointer;font:inherit;font-size:12px
    }
    .followup-row{display:flex;flex-wrap:wrap;gap:8px}
    .followup-btn{background:rgba(104,180,255,.08);border-color:rgba(104,180,255,.16)}
    .composer{display:grid;grid-template-columns:1fr auto;gap:12px;padding:18px 22px 22px;border-top:1px solid var(--line);background:var(--panel-3)}
    .composer-main{display:grid;gap:10px}
    .composer textarea{min-height:76px;max-height:220px;font-size:15px}
    .send-col{display:grid;gap:10px;align-content:end;min-width:138px}
    .send-note{color:var(--muted);font-size:12px;text-align:right;line-height:1.4}
    .toast-rack{position:fixed;right:18px;bottom:18px;display:grid;gap:10px;z-index:40;pointer-events:none}
    .toast{min-width:240px;max-width:min(380px,calc(100vw - 28px));padding:12px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(7,15,24,.94);box-shadow:0 16px 40px rgba(0,0,0,.28);color:var(--text)}
    .toast.ok{border-color:rgba(141,216,168,.24)}
    .toast.err{border-color:rgba(255,143,143,.30)}
    @media (max-width:1080px){body{overflow:auto}.shell{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 28px)}.msg{max-width:100%}.summary-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media (max-width:700px){.shell{width:calc(100vw - 18px);margin:9px auto;gap:12px}.rail,.composer,.chat-head,.summary-strip,.thread{padding-left:14px;padding-right:14px}.control-grid,.metric-grid,.summary-strip{grid-template-columns:1fr 1fr}.chat-head{flex-direction:column}.chat-head-side{justify-items:start}.composer{grid-template-columns:1fr}.send-col{min-width:0}.send-note{text-align:left}}
    @media (max-width:560px){.summary-strip{grid-template-columns:1fr}.toast-rack{left:12px;right:12px;bottom:12px}}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel rail">
      <section class="hero">
        <div class="eyebrow">Benchmark-backed local model</div>
        <h1>Supermix Qwen v26</h1>
        <p>Local Qwen 2.5 0.5B with the latest cognitive-activation LoRA adapter, benchmark summary, and practical desktop controls.</p>
        <div class="release-strip">
          <span class="pill" id="releaseLabel">Resolving adapter...</span>
          <span class="pill" id="deviceLabel">Waiting for runtime...</span>
        </div>
      </section>

      <section class="card">
        <h2>Model Snapshot</h2>
        <div class="metric-grid">
          <div class="metric"><div class="k">Token F1</div><div class="v" id="metricF1">-</div><div class="d" id="metricF1Delta">No benchmark yet</div></div>
          <div class="metric"><div class="k">Perplexity</div><div class="v" id="metricPpl">-</div><div class="d" id="metricPplDelta">No benchmark yet</div></div>
          <div class="metric"><div class="k">Avg Gen</div><div class="v" id="metricGen">-</div><div class="d" id="metricGenDelta">No benchmark yet</div></div>
          <div class="metric"><div class="k">Train Time</div><div class="v" id="metricTrain">-</div><div class="d" id="metricSamples">Awaiting adapter metadata</div></div>
        </div>
      </section>

      <section class="card">
        <h2>Generation Controls</h2>
        <div class="field">
          <label>Preset</label>
          <div class="preset-row">
            <button class="preset-btn" data-preset="direct">Direct</button>
            <button class="preset-btn" data-preset="balanced">Balanced</button>
            <button class="preset-btn" data-preset="reasoning">Reasoning</button>
            <button class="preset-btn" data-preset="creative">Creative</button>
            <button class="preset-btn" data-preset="coding">Coding</button>
          </div>
        </div>
        <div class="control-grid">
          <div class="field"><label>Max Tokens</label><input id="maxNew" type="number" min="24" max="512" step="1" value="112"></div>
          <div class="field"><label>Temperature</label><input id="temp" type="number" min="0" max="1.4" step="0.01" value="0.20"></div>
          <div class="field"><label>Top P</label><input id="topP" type="number" min="0.1" max="1.0" step="0.01" value="0.92"></div>
        </div>
        <div class="field">
          <label>Session Steering</label>
          <textarea id="systemHint" placeholder="Optional guidance for this session, for example: prefer concise answers, focus on debugging, or ask one clarifying question when details are missing."></textarea>
        </div>
        <p id="adapterMetaLine">Loading adapter metadata...</p>
      </section>
"""

HTML += r"""
      <section class="card">
        <h2>Quick Prompts</h2>
        <div class="chip-row" id="starterChips"></div>
      </section>

      <section class="card">
        <h2>Session Tools</h2>
        <div class="action-row">
          <button class="ghost-btn" id="refreshBtn">Refresh Status</button>
          <button class="ghost-btn" id="newSessionBtn">New Session</button>
          <button class="ghost-btn" id="copyLastBtn">Copy Last Reply</button>
          <button class="ghost-btn" id="exportBtn">Export Chat</button>
          <button class="primary-btn" id="clearBtn">Clear Session</button>
        </div>
        <p class="command-hint">Slash commands: <code>/help</code>, <code>/preset coding</code>, <code>/steer ...</code>, <code>/new</code>, <code>/clear</code>, <code>/export</code>, <code>/status</code></p>
        <div class="status-box" id="statusBox">Waiting for runtime status...</div>
      </section>
    </aside>

    <main class="panel chat">
      <header class="chat-head">
        <div class="chat-head-main">
          <h3 id="chatTitle">Preparing local runtime...</h3>
          <div class="chat-sub" id="chatSub">The desktop launcher will open this view after the local model server reports ready.</div>
        </div>
        <div class="chat-head-side">
          <div class="pill" id="sessionBadge">session pending</div>
          <div class="pill runtime-pill" id="runtimeBadge">syncing runtime</div>
        </div>
      </header>

      <section class="summary-strip">
        <div class="summary-card">
          <div class="summary-k">Turns</div>
          <div class="summary-v" id="turnCount">0</div>
          <div class="summary-d" id="turnDetail">No conversation yet</div>
        </div>
        <div class="summary-card">
          <div class="summary-k">Last Speed</div>
          <div class="summary-v" id="speedValue">-</div>
          <div class="summary-d" id="speedDetail">Awaiting first reply</div>
        </div>
        <div class="summary-card">
          <div class="summary-k">Last Output</div>
          <div class="summary-v" id="outputValue">-</div>
          <div class="summary-d" id="outputDetail">No assistant output yet</div>
        </div>
        <div class="summary-card">
          <div class="summary-k">Context</div>
          <div class="summary-v" id="contextValue">0</div>
          <div class="summary-d" id="contextDetail">Steering off</div>
        </div>
      </section>

      <section class="thread" id="thread">
        <div class="welcome" id="welcomeCard">
          Ask for debugging help, explanations, brainstorming, summaries, or code. The preset buttons tune style and generation,
          while Session Steering lets you bias the whole conversation without editing every prompt.
        </div>
      </section>

      <footer class="composer">
        <div class="composer-main">
          <textarea id="prompt" placeholder="Type a message. Press Enter to send, Shift+Enter for a new line."></textarea>
          <div class="followup-row" id="followupRow"></div>
        </div>
        <div class="send-col">
          <button class="primary-btn" id="sendBtn">Send</button>
          <div class="send-note" id="sendNote">Preset: balanced</div>
        </div>
      </footer>
    </main>
  </div>
  <div class="toast-rack" id="toastRack"></div>

  <script>
    const SETTINGS_KEY = "supermix-qwen-v26-settings";
    const SESSION_KEY = "supermix-qwen-v26-session";
    const HISTORY_KEY_PREFIX = "supermix-qwen-v26-history-";
    const DRAFT_KEY_PREFIX = "supermix-qwen-v26-draft-";
    const PRESETS = {
      direct: { maxNew: 72, temp: 0.05, topP: 0.82 },
      balanced: { maxNew: 112, temp: 0.20, topP: 0.92 },
      reasoning: { maxNew: 176, temp: 0.14, topP: 0.90 },
      creative: { maxNew: 200, temp: 0.58, topP: 0.97 },
      coding: { maxNew: 160, temp: 0.12, topP: 0.88 }
    };
    const FOLLOW_UPS = [
      { label: "Shorter", prompt: "Rewrite your last answer so it is shorter and more direct." },
      { label: "Deeper", prompt: "Go one level deeper and explain the tradeoffs clearly." },
      { label: "Runnable Code", prompt: "Turn your last answer into a small runnable example." },
      { label: "Checklist", prompt: "Turn your last answer into a practical checklist." }
    ];
    const STARTERS = [
      "Debug this stack trace and tell me the most likely root cause.",
      "Explain the tradeoffs between two ways to solve this problem.",
      "Rewrite this draft to sound more direct and professional.",
      "Give me a concise summary and the next action to take.",
      "Help me think through this idea before we implement it.",
      "Write a small runnable example I can test locally."
    ];

    const els = {
      thread: document.getElementById("thread"),
      prompt: document.getElementById("prompt"),
      sendBtn: document.getElementById("sendBtn"),
      sendNote: document.getElementById("sendNote"),
      maxNew: document.getElementById("maxNew"),
      temp: document.getElementById("temp"),
      topP: document.getElementById("topP"),
      systemHint: document.getElementById("systemHint"),
      starterChips: document.getElementById("starterChips"),
      statusBox: document.getElementById("statusBox"),
      releaseLabel: document.getElementById("releaseLabel"),
      deviceLabel: document.getElementById("deviceLabel"),
      adapterMetaLine: document.getElementById("adapterMetaLine"),
      chatTitle: document.getElementById("chatTitle"),
      chatSub: document.getElementById("chatSub"),
      sessionBadge: document.getElementById("sessionBadge"),
      runtimeBadge: document.getElementById("runtimeBadge"),
      metricF1: document.getElementById("metricF1"),
      metricF1Delta: document.getElementById("metricF1Delta"),
      metricPpl: document.getElementById("metricPpl"),
      metricPplDelta: document.getElementById("metricPplDelta"),
      metricGen: document.getElementById("metricGen"),
      metricGenDelta: document.getElementById("metricGenDelta"),
      metricTrain: document.getElementById("metricTrain"),
      metricSamples: document.getElementById("metricSamples"),
      welcomeCard: document.getElementById("welcomeCard"),
      turnCount: document.getElementById("turnCount"),
      turnDetail: document.getElementById("turnDetail"),
      speedValue: document.getElementById("speedValue"),
      speedDetail: document.getElementById("speedDetail"),
      outputValue: document.getElementById("outputValue"),
      outputDetail: document.getElementById("outputDetail"),
      contextValue: document.getElementById("contextValue"),
      contextDetail: document.getElementById("contextDetail"),
      followupRow: document.getElementById("followupRow"),
      toastRack: document.getElementById("toastRack")
    };

    const state = {
      sessionId: "",
      settings: { preset: "balanced", maxNew: 112, temp: 0.20, topP: 0.92, systemHint: "" },
      status: null,
      messages: [],
      lastBotText: "",
      pendingNode: null,
      sending: false
    };

    function makeSessionId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
      }
      return String(Date.now()) + "-" + Math.random().toString(16).slice(2, 10);
    }

    function historyStorageKey(sessionId) {
      return HISTORY_KEY_PREFIX + String(sessionId || "");
    }

    function draftStorageKey(sessionId) {
      return DRAFT_KEY_PREFIX + String(sessionId || "");
    }

    function currentDraftText() {
      return String((els.prompt && els.prompt.value) || "").trim();
    }

    function saveDraft() {
      try {
        const value = els.prompt ? String(els.prompt.value || "") : "";
        if (value) {
          localStorage.setItem(draftStorageKey(state.sessionId), value);
        } else {
          localStorage.removeItem(draftStorageKey(state.sessionId));
        }
      } catch (err) {
        /* localStorage can fail in hardened browser contexts */
      }
      updateConversationSummary();
    }

    function restoreDraft() {
      let value = "";
      try {
        value = localStorage.getItem(draftStorageKey(state.sessionId)) || "";
      } catch (err) {
        value = "";
      }
      els.prompt.value = value;
      autoResizePrompt();
      updateConversationSummary();
    }

    function formatClock(value) {
      const date = value ? new Date(value) : new Date();
      if (Number.isNaN(date.getTime())) return "";
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function sanitizeStoredMessages(value) {
      if (!Array.isArray(value)) return [];
      return value.map((item) => {
        if (!item || typeof item !== "object") return null;
        const kind = item.kind === "user" ? "user" : item.kind === "bot" ? "bot" : "";
        const text = String(item.text || "").trim();
        if (!kind || !text) return null;
        return {
          kind: kind,
          text: text,
          meta: item.meta && typeof item.meta === "object" ? item.meta : {},
          at: String(item.at || new Date().toISOString())
        };
      }).filter(Boolean).slice(-24);
    }

    function mountWelcomeCard() {
      const welcome = document.createElement("div");
      welcome.className = "welcome";
      welcome.id = "welcomeCard";
      welcome.textContent =
        "Ask for debugging help, explanations, brainstorming, summaries, or code. The preset buttons tune style and generation, " +
        "while Session Steering lets you bias the whole conversation without editing every prompt.";
      els.thread.appendChild(welcome);
      els.welcomeCard = welcome;
    }

    function renderConversation() {
      els.thread.innerHTML = "";
      els.welcomeCard = null;
      if (!state.messages.length) {
        mountWelcomeCard();
        updateConversationSummary();
        return;
      }
      state.messages.forEach((message) => addMessage(message.kind, message.text, message.meta || {}, { at: message.at }));
      updateConversationSummary();
    }

    function saveConversation() {
      if (!state.messages.length) {
        localStorage.removeItem(historyStorageKey(state.sessionId));
        return;
      }
      localStorage.setItem(historyStorageKey(state.sessionId), JSON.stringify(state.messages.slice(-24)));
    }

    function restoreConversation() {
      let raw = null;
      try {
        raw = JSON.parse(localStorage.getItem(historyStorageKey(state.sessionId)) || "[]");
      } catch (err) {
        raw = [];
      }
      state.messages = sanitizeStoredMessages(raw);
      state.lastBotText = "";
      for (let index = state.messages.length - 1; index >= 0; index -= 1) {
        if (state.messages[index].kind === "bot" && !(state.messages[index].meta && state.messages[index].meta.local_only)) {
          state.lastBotText = state.messages[index].text;
          break;
        }
      }
      renderConversation();
    }

    function showToast(text, kind) {
      const node = document.createElement("div");
      node.className = "toast " + (kind || "");
      node.textContent = text;
      els.toastRack.appendChild(node);
      window.setTimeout(() => {
        if (node.parentNode) node.parentNode.removeChild(node);
      }, 2600);
    }

    async function copyText(text, successMessage) {
      const value = String(text || "");
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(value);
      } else {
        const helper = document.createElement("textarea");
        helper.value = value;
        helper.setAttribute("readonly", "readonly");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        document.body.appendChild(helper);
        helper.select();
        document.execCommand("copy");
        document.body.removeChild(helper);
      }
      showToast(successMessage, "ok");
    }

    function setComposerPrompt(text, append) {
      const next = append && els.prompt.value.trim()
        ? els.prompt.value.trimEnd() + "\n\n" + text
        : text;
      els.prompt.value = next;
      autoResizePrompt();
      saveDraft();
      els.prompt.focus();
    }

    function quoteMessage(text) {
      const quoted = String(text || "")
        .split("\n")
        .map((line) => "> " + line)
        .join("\n");
      setComposerPrompt(quoted + "\n", true);
      showToast("Quoted into composer.", "ok");
    }

    function buildHistoryPayload() {
      return state.messages
        .filter((message) => (message.kind === "user" || message.kind === "bot") && !(message.meta && message.meta.local_only))
        .slice(-20)
        .map((message) => ({
          role: message.kind === "bot" ? "assistant" : "user",
          content: message.text
        }));
    }

    function updateConversationSummary() {
      const userTurns = state.messages.filter((message) => message.kind === "user").length;
      const lastBot = [...state.messages].reverse().find((message) => message.kind === "bot" && !(message.meta && message.meta.local_only)) || null;
      const historyDepth = buildHistoryPayload().length;

      els.turnCount.textContent = String(userTurns);
      els.turnDetail.textContent = userTurns ? ("session " + state.sessionId.slice(0, 8)) : "No conversation yet";

      els.speedValue.textContent =
        lastBot && Number.isFinite(Number(lastBot.meta && lastBot.meta.tokens_per_sec))
          ? Number(lastBot.meta.tokens_per_sec).toFixed(2) + "/s"
          : "-";
      els.speedDetail.textContent =
        lastBot && lastBot.meta && lastBot.meta.total_ms
          ? ("last " + lastBot.meta.total_ms + " ms")
          : "Awaiting first reply";

      els.outputValue.textContent =
        lastBot && lastBot.meta && lastBot.meta.output_tokens
          ? String(lastBot.meta.output_tokens)
          : "-";
      els.outputDetail.textContent =
        lastBot
          ? ("preset " + String((lastBot.meta && lastBot.meta.preset_used) || state.settings.preset))
          : "No assistant output yet";

      els.contextValue.textContent = String(historyDepth);
      els.contextDetail.textContent = [
        state.settings.systemHint.trim() ? "Steering on" : "Steering off",
        currentDraftText() ? "draft saved" : "draft empty"
      ].join(" · ");
    }

    function ensureSessionId() {
      const existing = localStorage.getItem(SESSION_KEY);
      state.sessionId = existing || makeSessionId();
      localStorage.setItem(SESSION_KEY, state.sessionId);
      els.sessionBadge.textContent = "session " + state.sessionId.slice(0, 8);
      updateConversationSummary();
    }

    function syncPresetButtons() {
      document.querySelectorAll("[data-preset]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.preset === state.settings.preset);
      });
    }

    function updateSendNote() {
      const profile = PRESETS[state.settings.preset] || PRESETS.balanced;
      els.sendNote.textContent =
        "Preset: " + state.settings.preset +
        " | " + profile.maxNew + " max tokens" +
        " | T=" + Number(state.settings.temp).toFixed(2) +
        " | top-p " + Number(state.settings.topP).toFixed(2);
      updateConversationSummary();
    }

    function loadSettings() {
      let raw = null;
      try { raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch (err) { raw = null; }
      if (raw && typeof raw === "object") {
        state.settings.preset = PRESETS[raw.preset] ? raw.preset : state.settings.preset;
        state.settings.maxNew = Number(raw.maxNew) || state.settings.maxNew;
        state.settings.temp = Number.isFinite(Number(raw.temp)) ? Number(raw.temp) : state.settings.temp;
        state.settings.topP = Number.isFinite(Number(raw.topP)) ? Number(raw.topP) : state.settings.topP;
        state.settings.systemHint = String(raw.systemHint || "");
      }
      els.maxNew.value = String(state.settings.maxNew);
      els.temp.value = state.settings.temp.toFixed(2);
      els.topP.value = state.settings.topP.toFixed(2);
      els.systemHint.value = state.settings.systemHint;
      syncPresetButtons();
      updateSendNote();
    }

    function saveSettings() {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
    }

    function applyPreset(name, overwriteValues) {
      if (!PRESETS[name]) return;
      state.settings.preset = name;
      if (overwriteValues) {
        state.settings.maxNew = PRESETS[name].maxNew;
        state.settings.temp = PRESETS[name].temp;
        state.settings.topP = PRESETS[name].topP;
        els.maxNew.value = String(state.settings.maxNew);
        els.temp.value = state.settings.temp.toFixed(2);
        els.topP.value = state.settings.topP.toFixed(2);
      }
      syncPresetButtons();
      updateSendNote();
      saveSettings();
    }
"""

HTML += r"""
    function setSending(flag) {
      state.sending = !!flag;
      els.sendBtn.disabled = state.sending;
      els.prompt.disabled = state.sending;
      els.sendBtn.textContent = state.sending ? "Generating..." : "Send";
    }

    function formatSigned(value, digits, invert) {
      if (!Number.isFinite(value)) return "n/a";
      const positive = invert ? value < 0 : value > 0;
      const cls = positive ? "up" : value === 0 ? "" : "down";
      const sign = value > 0 ? "+" : value < 0 ? "-" : "";
      return "<span class='" + cls + "'>" + sign + Math.abs(value).toFixed(digits) + "</span>";
    }

    function setMetricValue(target, value, suffix, digits) {
      if (!Number.isFinite(value)) { target.textContent = "-"; return; }
      target.textContent = Number(value).toFixed(digits) + (suffix || "");
    }

    function summarizeStatus(status) {
      const profile = status.profile || {};
      const benchmark = profile.benchmark || {};
      const targetModules = Array.isArray(profile.target_modules) ? profile.target_modules.length : 0;
      const runtimeState = status.loaded ? ((status.device || "cpu") + " | adapter " + (status.adapter_loaded ? "ready" : "missing")) : "runtime not loaded";
      els.releaseLabel.textContent = profile.label || "Latest adapter";
      els.deviceLabel.textContent = runtimeState;
      els.runtimeBadge.textContent = status.loaded ? runtimeState : "syncing runtime";
      els.chatTitle.textContent = status.loaded ? ((profile.label || "Supermix Qwen") + " ready") : "Runtime not ready";
      els.chatSub.textContent = status.loaded
        ? (
          "Base model: " + (profile.base_model || "unknown") +
          " | sessions: " + String(status.sessions || 0) +
          " | targets: " + String(targetModules) +
          (benchmark.available ? (" | eval F1 " + Number(benchmark.token_f1 || 0).toFixed(3)) : "")
        )
        : "The local runtime is still starting or failed to load.";
      els.adapterMetaLine.textContent =
        "LoRA r=" + String(profile.lora_rank || "-") +
        " | alpha=" + String(profile.lora_alpha || "-") +
        " | dropout=" + String(Number(profile.lora_dropout || 0).toFixed(2)) +
        " | targets=" + String(targetModules) +
        " | DoRA " + (profile.use_dora ? "on" : "off") +
        " | rsLoRA " + (profile.use_rslora ? "on" : "off");

      setMetricValue(els.metricF1, Number(benchmark.token_f1), "", 3);
      els.metricF1Delta.innerHTML = benchmark.available ? ("delta " + formatSigned(Number(benchmark.token_f1_delta), 3, false)) : "No benchmark yet";
      setMetricValue(els.metricPpl, Number(benchmark.perplexity), "", 2);
      els.metricPplDelta.innerHTML = benchmark.available ? ("delta " + formatSigned(Number(benchmark.perplexity_delta), 2, true)) : "No benchmark yet";
      setMetricValue(els.metricGen, Number(benchmark.avg_gen_seconds), "s", 2);
      els.metricGenDelta.innerHTML = benchmark.available ? ("delta " + formatSigned(Number(benchmark.avg_gen_seconds_delta), 2, true) + "s") : "No benchmark yet";
      els.metricTrain.textContent = Number.isFinite(Number(benchmark.train_hours)) ? (Number(benchmark.train_hours).toFixed(1) + "h") : "-";
      els.metricSamples.textContent = benchmark.available ? ("eval " + String(benchmark.eval_samples || 0) + " samples") : "Awaiting adapter metadata";
      els.statusBox.textContent = JSON.stringify(status, null, 2);
      state.status = status;
      updateConversationSummary();
    }

    function appendInlineFormatted(target, text) {
      const value = String(text || "");
      const re = /`([^`]+)`/g;
      let lastIndex = 0;
      let match = re.exec(value);
      while (match) {
        if (match.index > lastIndex) {
          target.appendChild(document.createTextNode(value.slice(lastIndex, match.index)));
        }
        const code = document.createElement("code");
        code.textContent = match[1];
        target.appendChild(code);
        lastIndex = match.index + match[0].length;
        match = re.exec(value);
      }
      if (lastIndex < value.length) {
        target.appendChild(document.createTextNode(value.slice(lastIndex)));
      }
    }

    function appendTextWithBreaks(target, text) {
      const lines = String(text || "").split("\n");
      lines.forEach((line, index) => {
        appendInlineFormatted(target, line);
        if (index < lines.length - 1) target.appendChild(document.createElement("br"));
      });
    }

    function renderTextBlock(container, block) {
      const lines = String(block || "").split("\n").map((line) => line.trimRight());
      const nonEmpty = lines.filter((line) => line.trim());
      if (!nonEmpty.length) return;

      if (nonEmpty.every((line) => /^>\s?/.test(line))) {
        const quote = document.createElement("blockquote");
        appendTextWithBreaks(quote, nonEmpty.map((line) => line.replace(/^>\s?/, "")).join("\n"));
        container.appendChild(quote);
        return;
      }

      if (nonEmpty.every((line) => /^[-*]\s+/.test(line))) {
        const list = document.createElement("ul");
        nonEmpty.forEach((line) => {
          const item = document.createElement("li");
          appendInlineFormatted(item, line.replace(/^[-*]\s+/, ""));
          list.appendChild(item);
        });
        container.appendChild(list);
        return;
      }

      if (nonEmpty.every((line) => /^\d+\.\s+/.test(line))) {
        const list = document.createElement("ol");
        nonEmpty.forEach((line) => {
          const item = document.createElement("li");
          appendInlineFormatted(item, line.replace(/^\d+\.\s+/, ""));
          list.appendChild(item);
        });
        container.appendChild(list);
        return;
      }

      const paragraph = document.createElement("p");
      appendTextWithBreaks(paragraph, nonEmpty.join("\n"));
      container.appendChild(paragraph);
    }

    function renderMessageBody(text) {
      const body = document.createElement("div");
      body.className = "body";
      const parts = String(text || "").replace(/\r\n/g, "\n").split(/```/);
      parts.forEach((part, index) => {
        if (!part) return;
        if (index % 2 === 1) {
          const block = document.createElement("div");
          block.className = "code-block";
          const codeHead = document.createElement("div");
          codeHead.className = "code-head";
          let codeText = part;
          let lang = "code";
          const newlineIndex = part.indexOf("\n");
          if (newlineIndex >= 0) {
            const possibleLang = part.slice(0, newlineIndex).trim();
            if (/^[A-Za-z0-9_#+.-]{1,20}$/.test(possibleLang)) {
              lang = possibleLang;
              codeText = part.slice(newlineIndex + 1);
            }
          }
          codeHead.textContent = lang;
          const pre = document.createElement("pre");
          const code = document.createElement("code");
          code.textContent = codeText.replace(/^\n/, "");
          pre.appendChild(code);
          block.appendChild(codeHead);
          block.appendChild(pre);
          body.appendChild(block);
          return;
        }
        part.split(/\n{2,}/).forEach((chunk) => renderTextBlock(body, chunk));
      });
      if (!body.childNodes.length) {
        const paragraph = document.createElement("p");
        paragraph.textContent = String(text || "");
        body.appendChild(paragraph);
      }
      return body;
    }

    function addMessage(kind, text, meta, options) {
      if (els.welcomeCard && els.welcomeCard.parentNode) {
        els.welcomeCard.parentNode.removeChild(els.welcomeCard);
        els.welcomeCard = null;
      }
      const opts = options || {};
      const node = document.createElement("article");
      node.className = "msg " + kind;

      const top = document.createElement("div");
      top.className = "msg-top";
      const who = document.createElement("div");
      who.className = "who";
      who.textContent = kind === "user" ? "You" : "Assistant";
      const time = document.createElement("div");
      time.className = "msg-time";
      time.textContent = formatClock(opts.at);
      top.appendChild(who);
      top.appendChild(time);
      node.appendChild(top);

      node.appendChild(renderMessageBody(text));

      if (meta) {
        const metaRow = document.createElement("div");
        metaRow.className = "meta";
        ["total_ms", "output_tokens", "tokens_per_sec", "preset_used"].forEach((key) => {
          if (!meta[key]) return;
          const item = document.createElement("span");
          if (key === "total_ms") item.textContent = "total " + meta[key] + " ms";
          if (key === "output_tokens") item.textContent = "tokens " + meta[key];
          if (key === "tokens_per_sec") item.textContent = "speed " + meta[key] + "/s";
          if (key === "preset_used") item.textContent = "preset " + meta[key];
          metaRow.appendChild(item);
        });
        if (meta.pending) {
          const item = document.createElement("span");
          item.textContent = meta.pending;
          metaRow.appendChild(item);
        }
        if (metaRow.childNodes.length) node.appendChild(metaRow);
      }

      if (!(meta && meta.pending)) {
        const actions = document.createElement("div");
        actions.className = "msg-actions";

        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "mini-btn";
        copyBtn.textContent = "Copy";
        copyBtn.addEventListener("click", async () => {
          try {
            await copyText(text, kind === "bot" ? "Reply copied." : "Message copied.");
          } catch (err) {
            showToast("Copy failed: " + err.message, "err");
          }
        });
        actions.appendChild(copyBtn);

        const quoteBtn = document.createElement("button");
        quoteBtn.type = "button";
        quoteBtn.className = "mini-btn";
        quoteBtn.textContent = "Quote";
        quoteBtn.addEventListener("click", () => quoteMessage(text));
        actions.appendChild(quoteBtn);

        node.appendChild(actions);
      }

      els.thread.appendChild(node);
      els.thread.scrollTop = els.thread.scrollHeight;
      return node;
    }

    function rememberMessage(kind, text, meta) {
      const safeMeta = meta || {};
      state.messages.push({ kind: kind, text: text, meta: safeMeta, at: new Date().toISOString() });
      if (kind === "bot" && !safeMeta.local_only) state.lastBotText = text;
      saveConversation();
      updateConversationSummary();
    }

    function addLocalNotice(text) {
      const meta = { local_only: true };
      addMessage("bot", text, meta, { at: new Date().toISOString() });
      rememberMessage("bot", text, meta);
    }

    function setPendingMessage() {
      if (state.pendingNode) return;
      const node = addMessage("bot", "Working on it...", { pending: "running local generation" }, { at: new Date().toISOString() });
      node.classList.add("pending");
      state.pendingNode = node;
    }

    function clearPendingMessage() {
      if (state.pendingNode && state.pendingNode.parentNode) state.pendingNode.parentNode.removeChild(state.pendingNode);
      state.pendingNode = null;
    }

    async function jget(path) {
      const response = await fetch(path);
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || ("HTTP " + response.status));
      return data;
    }

    async function jpost(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || ("HTTP " + response.status));
      return data;
    }

    async function refreshStatus() {
      try {
        const data = await jget("/api/status");
        summarizeStatus(data.status);
      } catch (err) {
        els.statusBox.textContent = "Status error: " + err.message;
        els.runtimeBadge.textContent = "status error";
      }
    }

    function autoResizePrompt() {
      els.prompt.style.height = "auto";
      els.prompt.style.height = Math.min(220, els.prompt.scrollHeight) + "px";
    }

    function parseSlashCommand(text) {
      const trimmed = String(text || "").trim();
      if (!trimmed.startsWith("/")) return null;
      const firstSpace = trimmed.indexOf(" ");
      const name = (firstSpace === -1 ? trimmed.slice(1) : trimmed.slice(1, firstSpace)).trim().toLowerCase();
      const arg = firstSpace === -1 ? "" : trimmed.slice(firstSpace + 1).trim();
      if (!name) return null;
      return { name: name, arg: arg };
    }

    function summarizeLocalStatus() {
      if (!state.status) return "Runtime status has not loaded yet.";
      const profile = state.status.profile || {};
      const benchmark = profile.benchmark || {};
      const parts = [
        "Runtime " + (state.status.loaded ? "loaded" : "not ready"),
        "device " + String(state.status.device || "unknown"),
        "adapter " + (state.status.adapter_loaded ? "ready" : "missing"),
        "preset " + state.settings.preset,
        "context " + String(buildHistoryPayload().length) + " turns"
      ];
      if (benchmark.available && Number.isFinite(Number(benchmark.token_f1))) {
        parts.push("F1 " + Number(benchmark.token_f1).toFixed(3));
      }
      return parts.join(" | ");
    }

    async function handleSlashCommand(command) {
      if (!command) return false;
      if (command.name === "help") {
        addLocalNotice(
          "Slash commands stay local:\n" +
          "- `/help` show command help\n" +
          "- `/preset <name>` switch presets, for example `/preset coding`\n" +
          "- `/steer <text>` update the session steering note\n" +
          "- `/new` start a new session\n" +
          "- `/clear` clear the current session on the server and in the UI\n" +
          "- `/export` download the current transcript\n" +
          "- `/status` summarize the current runtime and session state"
        );
        return true;
      }
      if (command.name === "preset") {
        const presetName = command.arg.toLowerCase();
        if (!PRESETS[presetName]) {
          addLocalNotice("Unknown preset. Available presets: `direct`, `balanced`, `reasoning`, `creative`, `coding`.");
          return true;
        }
        applyPreset(presetName, true);
        addLocalNotice("Preset changed to `" + presetName + "`.");
        return true;
      }
      if (command.name === "steer") {
        state.settings.systemHint = command.arg;
        els.systemHint.value = command.arg;
        saveSettings();
        updateSendNote();
        addLocalNotice(command.arg ? "Session steering updated." : "Session steering cleared.");
        return true;
      }
      if (command.name === "new") {
        newSession();
        return true;
      }
      if (command.name === "clear") {
        await clearSession(true);
        return true;
      }
      if (command.name === "export") {
        exportTranscript();
        return true;
      }
      if (command.name === "status") {
        addLocalNotice(summarizeLocalStatus());
        return true;
      }
      addLocalNotice("Unknown slash command `/" + command.name + "`. Use `/help`.");
      return true;
    }

    async function sendMessage() {
      const text = els.prompt.value.trim();
      if (!text || state.sending) return;
      const command = parseSlashCommand(text);
      if (command) {
        els.prompt.value = "";
        autoResizePrompt();
        saveDraft();
        await handleSlashCommand(command);
        return;
      }
      const history = buildHistoryPayload();
      addMessage("user", text, {}, { at: new Date().toISOString() });
      rememberMessage("user", text);
      els.prompt.value = "";
      autoResizePrompt();
      saveDraft();
      setSending(true);
      setPendingMessage();
      try {
        const data = await jpost("/api/chat", {
          session_id: state.sessionId,
          history: history,
          message: text,
          preset: state.settings.preset,
          system_hint: state.settings.systemHint,
          max_new_tokens: Number(state.settings.maxNew),
          temperature: Number(state.settings.temp),
          top_p: Number(state.settings.topP)
        });
        clearPendingMessage();
        addMessage("bot", data.response, data.timing || {}, { at: new Date().toISOString() });
        rememberMessage("bot", data.response, data.timing || {});
      } catch (err) {
        clearPendingMessage();
        const textOut = "Error: " + err.message;
        addMessage("bot", textOut, { local_only: true }, { at: new Date().toISOString() });
        rememberMessage("bot", textOut, { local_only: true });
      } finally {
        setSending(false);
      }
    }

    async function clearSession(remote) {
      if (remote !== false) {
        try {
          await jpost("/api/clear", { session_id: state.sessionId });
        } catch (err) {
          showToast("Clear failed: " + err.message, "err");
          return;
        }
      }
      state.messages = [];
      state.lastBotText = "";
      saveConversation();
      renderConversation();
      showToast("Session cleared.", "ok");
    }

    function newSession() {
      state.sessionId = makeSessionId();
      localStorage.setItem(SESSION_KEY, state.sessionId);
      els.sessionBadge.textContent = "session " + state.sessionId.slice(0, 8);
      state.messages = [];
      state.lastBotText = "";
      saveConversation();
      renderConversation();
      restoreDraft();
      showToast("Started a new blank session.", "ok");
    }

    async function copyLastReply() {
      if (!state.lastBotText) {
        showToast("No assistant reply to copy yet.", "err");
        return;
      }
      try {
        await copyText(state.lastBotText, "Last reply copied.");
      } catch (err) {
        showToast("Copy failed: " + err.message, "err");
      }
    }

    function exportTranscript() {
      const payload = {
        exported_at: new Date().toISOString(),
        session_id: state.sessionId,
        settings: state.settings,
        profile: state.status ? (state.status.profile || {}) : {},
        messages: state.messages
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "supermix-qwen-v26-" + state.sessionId.slice(0, 8) + ".json";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showToast("Chat exported.", "ok");
    }

    function bindInputs() {
      els.maxNew.addEventListener("input", () => { state.settings.maxNew = Math.max(24, Number(els.maxNew.value) || 112); updateSendNote(); saveSettings(); });
      els.temp.addEventListener("input", () => { const value = Number(els.temp.value); state.settings.temp = Number.isFinite(value) ? value : 0.20; updateSendNote(); saveSettings(); });
      els.topP.addEventListener("input", () => { const value = Number(els.topP.value); state.settings.topP = Number.isFinite(value) ? value : 0.92; updateSendNote(); saveSettings(); });
      els.systemHint.addEventListener("input", () => { state.settings.systemHint = els.systemHint.value; updateSendNote(); saveSettings(); });
      document.querySelectorAll("[data-preset]").forEach((btn) => btn.addEventListener("click", () => applyPreset(btn.dataset.preset, true)));
      els.prompt.addEventListener("input", () => { autoResizePrompt(); saveDraft(); });
      els.prompt.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } });
      document.getElementById("sendBtn").addEventListener("click", sendMessage);
      document.getElementById("refreshBtn").addEventListener("click", refreshStatus);
      document.getElementById("clearBtn").addEventListener("click", () => clearSession(true));
      document.getElementById("newSessionBtn").addEventListener("click", newSession);
      document.getElementById("copyLastBtn").addEventListener("click", copyLastReply);
      document.getElementById("exportBtn").addEventListener("click", exportTranscript);
    }

    function initStarterChips() {
      STARTERS.forEach((text) => {
        const button = document.createElement("button");
        button.className = "chip";
        button.textContent = text;
        button.addEventListener("click", () => { els.prompt.value = text; autoResizePrompt(); saveDraft(); els.prompt.focus(); });
        els.starterChips.appendChild(button);
      });
    }

    function initFollowupChips() {
      FOLLOW_UPS.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "followup-btn";
        button.textContent = item.label;
        button.addEventListener("click", () => {
          if (!state.lastBotText) {
            showToast("Ask the model something first.", "err");
            return;
          }
          setComposerPrompt(item.prompt, false);
        });
        els.followupRow.appendChild(button);
      });
    }

    ensureSessionId();
    loadSettings();
    restoreConversation();
    restoreDraft();
    bindInputs();
    initStarterChips();
    initFollowupChips();
    autoResizePrompt();
    refreshStatus();
    setInterval(refreshStatus, 15000);
  </script>
</body>
</html>
"""


ARTIFACT_TAG_RE = re.compile(
    r"\[[^\]\n]*(?:variant|worked solution|set\d+|reflective|counterexample|debug|planning|mentor|teaching)[^\]\n]*\]",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
LEAD_NOISE_PHRASES = (
    "let me reason through this carefully.",
    "let me work through this step by step.",
    "let me work through this carefully.",
    "walk me through the solution:",
    "solve this step by step:",
)
LEGACY_NESTED_ADAPTER_PREFIX = "base_model.model.base_model.model.model."
NESTED_PREFIX_FRAGMENT = "base_model.model.base_model.model."


def clean_generated_response(text: str) -> str:
    out = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not out:
        return ""
    out = re.sub(r"^\s*assistant\s*:\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(
        r"^\s*(?:a\s+\w+\s+angle|build\s+a\s+\w+\s+angle\s+for\s+problem-solving|shift\s+to\s+a\s+\w+\s+angle\s+for\s+problem-solving)\s*:?\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = ARTIFACT_TAG_RE.sub(" ", out)
    out = re.sub(r"^\s*(\[[^\]\n]{1,90}\]\s*)+", "", out).strip()
    lowered = out.lower()
    for phrase in LEAD_NOISE_PHRASES:
        if lowered.startswith(phrase):
            out = out[len(phrase) :].lstrip(" :-\n")
            lowered = out.lower()
    out = re.sub(r"[ \t]+", " ", out).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _build_bullets_from_text(text: str, n: int) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part.strip()]
    if not parts:
        return ""
    return "\n".join(f"- {re.sub(r'^[\\-*\\d\\.\\)\\s]+', '', part).strip()}" for part in parts[: max(1, min(int(n), len(parts)))])


def _normalize_bullet_output(text: str, n: int) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    cleaned: List[str] = []
    for line in lines:
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if line and len(line.split()) >= 3:
            cleaned.append(line)
    if not cleaned:
        return _build_bullets_from_text(text, n=n)
    return "\n".join(f"- {cleaned[i]}" for i in range(max(1, min(int(n), len(cleaned)))))


def enforce_response_contract(user_text: str, response_text: str) -> str:
    user_low = str(user_text or "").lower()
    out = str(response_text or "").strip()
    if not out:
        return out

    math_match = re.search(r"what is\s+(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", user_low)
    if "just the answer" in user_low and math_match:
        a = float(math_match.group(1))
        op = math_match.group(2)
        b = float(math_match.group(3))
        if op == "+":
            val = a + b
        elif op == "-":
            val = a - b
        elif op in {"*", "x"}:
            val = a * b
        else:
            if abs(b) < 1e-12:
                return "undefined"
            val = a / b
        return str(int(round(val))) if abs(val - round(val)) < 1e-9 else f"{val:.6g}"

    if "difference between precision and recall" in user_low:
        return (
            "Precision is the fraction of predicted positives that are actually positive. "
            "Recall is the fraction of actual positives that the model correctly finds."
        )

    if "overfitting" in user_low and "bullet" in user_low:
        return (
            "- Overfitting means the model memorizes training details instead of learning general patterns.\n"
            "- It usually performs well on training data but worse on unseen data.\n"
            "- You can reduce it with regularization, simpler models, and better validation."
        )

    if "bullet" in user_low:
        match = re.search(r"(\d+)\s+(?:short\s+)?bullet", user_low)
        out = _normalize_bullet_output(out, n=int(match.group(1)) if match else 3)

    if "just the answer" in user_low:
        nums = NUMBER_RE.findall(out)
        if nums:
            return nums[-1]
    return out


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except Exception:
        return int(fallback)


def clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(fallback)
    return max(minimum, min(maximum, number))


def normalize_history_payload(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    cleaned: List[Dict[str, str]] = []
    for item in value[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    return cleaned[-12:]


def slug_to_label(value: str) -> str:
    text = re.sub(r"^qwen_supermix_enhanced_", "", str(value or "").strip())
    text = re.sub(r"[_\-]+", " ", text).strip()
    words = []
    for word in text.split():
        words.append(word.upper() if re.fullmatch(r"v\d+", word.lower()) else word.capitalize())
    return " ".join(words) or "Latest Adapter"


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to load JSON from %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def summarize_benchmark(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"available": False}
    base = data.get("base") if isinstance(data.get("base"), dict) else {}
    tuned = data.get("tuned") if isinstance(data.get("tuned"), dict) else {}
    train_stats = data.get("train_stats") if isinstance(data.get("train_stats"), dict) else {}
    if not tuned:
        return {"available": False}
    return {
        "available": True,
        "eval_samples": int(float(tuned.get("eval_samples") or base.get("eval_samples") or 0.0)),
        "token_f1": float(tuned.get("token_f1") or 0.0),
        "token_f1_delta": float(tuned.get("token_f1") or 0.0) - float(base.get("token_f1") or 0.0),
        "perplexity": float(tuned.get("perplexity") or 0.0),
        "perplexity_delta": float(tuned.get("perplexity") or 0.0) - float(base.get("perplexity") or 0.0),
        "avg_gen_seconds": float(tuned.get("avg_gen_seconds") or 0.0),
        "avg_gen_seconds_delta": float(tuned.get("avg_gen_seconds") or 0.0) - float(base.get("avg_gen_seconds") or 0.0),
        "train_hours": float(train_stats.get("train_seconds") or 0.0) / 3600.0,
    }


def build_artifact_profile(adapter_dir: Path) -> Dict[str, Any]:
    artifact_dir = adapter_dir.parent
    adapter_cfg = load_json_if_exists(adapter_dir / "adapter_config.json") or {}
    benchmark = summarize_benchmark(load_json_if_exists(artifact_dir / "benchmark_results.json"))
    target_modules = adapter_cfg.get("target_modules")
    if not isinstance(target_modules, list):
        target_modules = []
    return {
        "artifact_name": artifact_dir.name,
        "label": slug_to_label(artifact_dir.name),
        "artifact_dir": str(artifact_dir),
        "adapter_dir": str(adapter_dir),
        "base_model": str(adapter_cfg.get("base_model_name_or_path") or ""),
        "lora_rank": int(adapter_cfg.get("r") or 0),
        "lora_alpha": int(adapter_cfg.get("lora_alpha") or 0),
        "lora_dropout": float(adapter_cfg.get("lora_dropout") or 0.0),
        "use_dora": bool(adapter_cfg.get("use_dora")),
        "use_rslora": bool(adapter_cfg.get("use_rslora")),
        "target_modules": [str(item) for item in target_modules if isinstance(item, str)],
        "benchmark": benchmark,
    }


def score_adapter_dir(adapter_dir: Path) -> Tuple[int, float]:
    parent = adapter_dir.parent
    benchmark_file = parent / "benchmark_results.json"
    if benchmark_file.exists():
        return (2, benchmark_file.stat().st_mtime)
    checkpoint_meta = parent / "checkpoint_meta.json"
    if checkpoint_meta.exists():
        return (1, checkpoint_meta.stat().st_mtime)
    weight_file = adapter_dir / "adapter_model.safetensors"
    if weight_file.exists():
        return (0, weight_file.stat().st_mtime)
    return (0, adapter_dir.stat().st_mtime)


def resolve_gui_default_adapter_dir(project_root: Path) -> Optional[Path]:
    pointer_path = project_root / ".gui_default_adapter.txt"
    if not pointer_path.exists():
        return None
    try:
        raw = pointer_path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    except OSError as exc:
        logging.warning("Failed to read GUI default adapter pointer %s: %s", pointer_path, exc)
        return None
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    cfg = candidate / "adapter_config.json"
    weights = candidate / "adapter_model.safetensors"
    if cfg.exists() and weights.exists():
        return candidate.resolve()
    logging.warning("Ignoring invalid GUI default adapter pointer %s -> %s", pointer_path, candidate)
    return None


def find_latest_adapter_dir(project_root: Path) -> Path:
    pinned = resolve_gui_default_adapter_dir(project_root)
    if pinned is not None:
        return pinned
    candidates: Dict[str, Path] = {}
    for pattern in ("artifacts/qwen_supermix_enhanced_*/adapter", "artifacts/*/adapter", "artifacts/**/adapter"):
        for candidate in project_root.glob(pattern):
            if (candidate / "adapter_config.json").exists() and (candidate / "adapter_model.safetensors").exists():
                candidates[str(candidate.resolve())] = candidate.resolve()
    if not candidates:
        raise FileNotFoundError("Could not find any Qwen adapter directory under artifacts.")
    return max(candidates.values(), key=score_adapter_dir)


def resolve_adapter_dir(project_root: Path, explicit_adapter_dir: str) -> Path:
    raw = str(explicit_adapter_dir or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        if not (candidate / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter directory is missing adapter_config.json: {candidate}")
        return candidate
    return find_latest_adapter_dir(project_root)


def looks_like_model_repo_id(value: str) -> bool:
    return bool(MODEL_REPO_ID_RE.fullmatch(str(value or "").strip()))


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        logging.debug("Skipping inaccessible path during exists() probe: %s (%s)", path, exc)
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError as exc:
        logging.debug("Skipping inaccessible path during is_dir() probe: %s (%s)", path, exc)
        return False


def safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError as exc:
        logging.debug("Falling back to unresolved path for %s (%s)", path, exc)
        return path


def sorted_child_dirs(path: Path) -> List[Path]:
    try:
        children = [child for child in path.iterdir() if safe_is_dir(child)]
    except OSError as exc:
        logging.debug("Skipping inaccessible directory listing: %s (%s)", path, exc)
        return []

    def mtime(child: Path) -> float:
        try:
            return child.stat().st_mtime
        except OSError as exc:
            logging.debug("Skipping mtime probe for %s (%s)", child, exc)
            return -1.0

    return sorted(children, key=mtime, reverse=True)


def is_local_model_dir(path: Path) -> bool:
    return safe_is_dir(path) and safe_exists(path / "config.json") and any(safe_exists(path / name) for name in MODEL_WEIGHT_FILES)


def iter_hf_cache_roots() -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()

    def add_path(path: Optional[Path]) -> None:
        if path is None:
            return
        candidate = path.expanduser()
        resolved = safe_resolve(candidate)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    for env_name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            add_path(Path(raw))
    hf_home = str(os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        add_path(Path(hf_home) / "hub")
    xdg_cache = str(os.environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg_cache:
        add_path(Path(xdg_cache) / "huggingface" / "hub")
    add_path(Path.home() / ".cache" / "huggingface" / "hub")
    local_appdata = str(os.environ.get("LOCALAPPDATA") or "").strip()
    if local_appdata:
        add_path(Path(local_appdata) / "huggingface" / "hub")
    return roots


def find_cached_model_snapshot(repo_id: str, extra_roots: Optional[List[Path]] = None) -> Optional[Path]:
    folder_name = f"models--{repo_id.replace('/', '--')}"
    roots = list(extra_roots or []) + iter_hf_cache_roots()
    seen: set[str] = set()

    def iter_snapshot_candidates(repo_cache_dir: Path) -> List[Path]:
        candidates: List[Path] = []
        refs_dir = repo_cache_dir / "refs"
        for ref_name in ("main", "master"):
            ref_file = refs_dir / ref_name
            if safe_exists(ref_file):
                try:
                    snapshot_name = ref_file.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError as exc:
                    logging.debug("Skipping inaccessible HF ref file %s (%s)", ref_file, exc)
                    snapshot_name = ""
                if snapshot_name:
                    candidates.append(repo_cache_dir / "snapshots" / snapshot_name)
        snapshots_dir = repo_cache_dir / "snapshots"
        if safe_exists(snapshots_dir):
            candidates.extend(sorted_child_dirs(snapshots_dir))
        return candidates

    for root in roots:
        for repo_cache_dir in (root, root / folder_name):
            key = str(repo_cache_dir)
            if key in seen:
                continue
            seen.add(key)
            if is_local_model_dir(repo_cache_dir):
                return safe_resolve(repo_cache_dir)
            if not safe_exists(repo_cache_dir):
                continue
            for snapshot_dir in iter_snapshot_candidates(repo_cache_dir):
                if is_local_model_dir(snapshot_dir):
                    return safe_resolve(snapshot_dir)
    return None


def resolve_local_base_model_path(value: str) -> str:
    raw = str(value or "").strip()
    override_raw = str(os.environ.get(BASE_MODEL_OVERRIDE_ENV) or "").strip()
    repo_id = raw if looks_like_model_repo_id(raw) else DEFAULT_BASE_MODEL_REPO
    bundled_base_model = find_bundled_base_model_dir()

    if override_raw:
        override_path = Path(override_raw).expanduser()
        resolved_override = find_cached_model_snapshot(repo_id, extra_roots=[override_path])
        if resolved_override is not None:
            return str(resolved_override)
        raise FileNotFoundError(
            f"{BASE_MODEL_OVERRIDE_ENV} is set to '{override_path}', but no usable local base model was found there for '{repo_id}'."
        )

    if raw:
        raw_path = Path(raw).expanduser()
        if safe_exists(raw_path):
            if not is_local_model_dir(raw_path):
                raise FileNotFoundError(f"Base model directory exists but does not look usable for offline loading: {raw_path}")
            return str(safe_resolve(raw_path))
        if looks_like_model_repo_id(raw):
            if raw == DEFAULT_BASE_MODEL_REPO and bundled_base_model is not None:
                return str(bundled_base_model)
            resolved_snapshot = find_cached_model_snapshot(raw)
            if resolved_snapshot is not None:
                return str(resolved_snapshot)
            default_snapshot = Path(DEFAULT_BASE_MODEL)
            if raw == DEFAULT_BASE_MODEL_REPO and safe_exists(default_snapshot) and is_local_model_dir(default_snapshot):
                return str(safe_resolve(default_snapshot))
            raise FileNotFoundError(
                f"Could not find a local Hugging Face cache snapshot for '{raw}'. "
                f"Set {BASE_MODEL_OVERRIDE_ENV} to a local model directory or pre-download the base model."
            )
        raise FileNotFoundError(f"Base model path does not exist: {raw_path}")

    if bundled_base_model is not None:
        return str(bundled_base_model)

    default_snapshot = Path(DEFAULT_BASE_MODEL)
    if safe_exists(default_snapshot) and is_local_model_dir(default_snapshot):
        return str(safe_resolve(default_snapshot))

    resolved_snapshot = find_cached_model_snapshot(DEFAULT_BASE_MODEL_REPO)
    if resolved_snapshot is not None:
        return str(resolved_snapshot)

    raise FileNotFoundError(
        f"Could not resolve a local base model for '{DEFAULT_BASE_MODEL_REPO}'. "
        f"Set {BASE_MODEL_OVERRIDE_ENV} to a local model directory or pre-download the base model."
    )


def resolve_base_model_path(explicit_base_model: str, adapter_dir: Path) -> str:
    raw = str(explicit_base_model or "").strip()
    if raw:
        return resolve_local_base_model_path(raw)
    adapter_cfg = load_json_if_exists(adapter_dir / "adapter_config.json") or {}
    base_model = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
    return resolve_local_base_model_path(base_model)


def find_bundled_base_model_dir() -> Optional[Path]:
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    for folder_name in ("bundled_base_model", "base_model"):
        candidate = Path(meipass) / folder_name
        if is_local_model_dir(candidate):
            return safe_resolve(candidate)
    return None


class Engine:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device, adapter_loaded: bool, profile: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adapter_loaded = adapter_loaded
        self.profile = dict(profile)
        self.lock = threading.RLock()
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "device": str(self.device),
            "adapter_loaded": bool(self.adapter_loaded),
            "sessions": len(self.sessions),
            "profile": self.profile,
            "defaults": {"preset": "balanced", "generation": PRESET_GENERATION["balanced"]},
        }

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)

    def _build_prompt(self, history: List[Dict[str, str]], user_text: str, preset: str, system_hint: str) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        preset_hint = PRESET_HINTS.get(preset)
        if preset_hint:
            messages.append({"role": "system", "content": preset_hint})
        if system_hint:
            messages.append({"role": "system", "content": f"Session steering: {system_hint}"})
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = []
        for message in messages:
            lines.append(f"{str(message.get('role', 'user')).upper()}: {str(message.get('content', ''))}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def chat(
        self,
        session_id: str,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        preset: str,
        system_hint: str,
        history_override: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")

        preset_name = preset if preset in PRESET_GENERATION else "balanced"
        defaults = PRESET_GENERATION[preset_name]
        system_hint = str(system_hint or "").strip()[:320]

        with self.lock:
            stored_history = list(self.sessions.get(session_id, []))[-12:]
        history = list(history_override)[-12:] if history_override is not None else stored_history

        prompt = self._build_prompt(history, user_text, preset_name, system_hint)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)

        final_max_new_tokens = clamp_int(max_new_tokens, 24, 512, int(defaults["max_new_tokens"]))
        final_temperature = clamp_float(temperature, 0.0, 1.4, float(defaults["temperature"]))
        final_top_p = clamp_float(top_p, 0.1, 1.0, float(defaults["top_p"]))
        do_sample = final_temperature >= 0.16

        gen_kwargs = {
            "max_new_tokens": final_max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.08,
            "no_repeat_ngram_size": 4,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = final_temperature
            gen_kwargs["top_p"] = final_top_p
            gen_kwargs["top_k"] = 40

        wall_start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        total_ms = (time.perf_counter() - wall_start) * 1000.0

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        response = clean_generated_response(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        response = enforce_response_contract(user_text=user_text, response_text=response) or "(no output)"

        with self.lock:
            new_history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response},
            ]
            self.sessions[session_id] = new_history[-20:]

        output_tokens = int(new_tokens.shape[0])
        seconds = max(1e-6, total_ms / 1000.0)
        return {
            "ok": True,
            "session_id": session_id,
            "response": response,
            "preset_used": preset_name,
            "output_tokens": output_tokens,
            "timing": {
                "total_ms": round(total_ms, 1),
                "tokens_per_sec": round(output_tokens / seconds, 2),
                "preset_used": preset_name,
                "output_tokens": output_tokens,
            },
        }


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict):
            return state
        raise TypeError(f"Unexpected adapter_model.bin payload type: {type(state)}")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _is_legacy_nested_adapter_state(state_dict: Dict[str, torch.Tensor]) -> bool:
    return bool(state_dict) and next(iter(state_dict.keys())).startswith(LEGACY_NESTED_ADAPTER_PREFIX)


def _canonicalize_adapter_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    remapped: Dict[str, torch.Tensor] = {}
    remapped_count = 0
    for key, value in state_dict.items():
        new_key = key
        while NESTED_PREFIX_FRAGMENT in new_key:
            new_key = new_key.replace(NESTED_PREFIX_FRAGMENT, "base_model.model.")
        if new_key != key:
            remapped_count += 1
        remapped[new_key] = value
    return remapped, remapped_count


def _set_model_use_cache(model: Any, enabled: bool) -> None:
    use_cache = bool(enabled)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = use_cache
    base_model = getattr(model, "base_model", None)
    base_cfg = getattr(base_model, "config", None)
    if base_cfg is not None:
        base_cfg.use_cache = use_cache


def _load_adapter_with_compat(model: Any, adapter_dir: Path, device: torch.device) -> Any:
    state_dict = _load_adapter_state_dict(adapter_dir)
    if _is_legacy_nested_adapter_state(state_dict):
        logging.info("[adapter] legacy nested format detected: %s", adapter_dir)
    remapped_state, remapped_count = _canonicalize_adapter_state_dict(state_dict)
    peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
    peft_cfg.inference_mode = True
    model = get_peft_model(model, peft_cfg)
    incompat = set_peft_model_state_dict(model, remapped_state, adapter_name="default")
    logging.info(
        "[adapter] loaded remapped=%s missing=%s unexpected=%s",
        remapped_count,
        len(getattr(incompat, "missing_keys", []) or []),
        len(getattr(incompat, "unexpected_keys", []) or []),
    )
    return model.to(device)


def resolve_device(preferred: str) -> torch.device:
    pref = str(preferred).strip().lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_engine(base_model: str, adapter_dir: Path, device: torch.device) -> Engine:
    base_model = resolve_local_base_model_path(base_model)
    profile = build_artifact_profile(adapter_dir)
    logging.info("Loading tokenizer from %s", base_model)
    tokenizer = Qwen2Tokenizer.from_pretrained(base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading base model weights from %s", base_model)
    model = Qwen2ForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).to(device)
    _set_model_use_cache(model, enabled=True)

    adapter_loaded = False
    if adapter_dir.exists():
        logging.info("Loading adapter weights from %s", adapter_dir)
        model = _load_adapter_with_compat(model=model, adapter_dir=adapter_dir, device=device)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload().to(device)
            logging.info("[adapter] merged into base model for faster inference")
        adapter_loaded = True
    else:
        logging.warning("Adapter path does not exist: %s", adapter_dir)

    _set_model_use_cache(model, enabled=True)
    model.eval()
    logging.info("Model ready for inference")
    return Engine(model=model, tokenizer=tokenizer, device=device, adapter_loaded=adapter_loaded, profile=profile)


def build_app(engine: Engine) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return HTML

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "status": engine.status()})

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip() or str(uuid.uuid4())
        message = str(payload.get("message") or "").strip()
        preset = str(payload.get("preset") or "balanced").strip().lower()
        history_override = normalize_history_payload(payload.get("history")) if "history" in payload else None
        try:
            return jsonify(
                engine.chat(
                    session_id=session_id,
                    user_text=message,
                    max_new_tokens=int(payload.get("max_new_tokens") or PRESET_GENERATION["balanced"]["max_new_tokens"]),
                    temperature=float(payload.get("temperature") or PRESET_GENERATION["balanced"]["temperature"]),
                    top_p=float(payload.get("top_p") or PRESET_GENERATION["balanced"]["top_p"]),
                    preset=preset,
                    system_hint=str(payload.get("system_hint") or ""),
                    history_override=history_override,
                )
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/clear")
    def api_clear():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        engine.clear(session_id)
        return jsonify({"ok": True, "session_id": session_id})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Supermix Qwen v26 web chat.")
    ap.add_argument("--base_model", default="", help="Base model path or cached Hugging Face model id.")
    ap.add_argument("--adapter_dir", default="", help="Adapter directory. Defaults to the latest benchmarked adapter.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = resolve_adapter_dir(project_root, args.adapter_dir)
    base_model = resolve_base_model_path(args.base_model, adapter_dir)
    device = resolve_device("cuda" if args.device == "auto" else args.device)

    logging.info("[load] device=%s", device)
    logging.info("[load] base_model=%s", base_model)
    logging.info("[load] adapter_dir=%s", adapter_dir)

    engine = load_engine(base_model=base_model, adapter_dir=adapter_dir, device=device)
    logging.info("[ready] adapter_loaded=%s", engine.adapter_loaded)

    app = build_app(engine)
    logging.info("[ready] web ui: http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
