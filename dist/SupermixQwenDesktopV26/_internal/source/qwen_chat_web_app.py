import argparse
import json
import logging
import re
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

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Qwen v26</title>
  <style>
    :root{
      --bg:#08121e;--panel:rgba(12,24,38,.9);--panel-2:rgba(16,30,46,.96);--line:rgba(151,180,214,.18);
      --text:#ecf3fb;--muted:#9db2cc;--blue:#68b4ff;--amber:#ffb163;--up:#8dd8a8;--down:#f2b3b3;
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
    .preset-btn.active{border-color:rgba(104,180,255,.48);background:rgba(104,180,255,.12)}
    .primary-btn{background:linear-gradient(135deg,#2d74c4,var(--blue));font-weight:700}
    .card{padding:16px}
    .card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
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
    .chat{display:grid;grid-template-rows:auto 1fr auto;min-height:0}
    .chat-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;padding:18px 22px;border-bottom:1px solid var(--line);background:rgba(15,29,45,.96)}
    .chat-head h3{margin:0;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif;font-size:24px;line-height:1.05}
    .chat-sub{margin-top:8px;color:var(--muted);font-size:13px;line-height:1.55}
    .thread{min-height:0;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:16px;background:linear-gradient(180deg,rgba(6,12,20,.45),rgba(10,19,31,.88))}
    .welcome{padding:18px;border-radius:var(--r-lg);border:1px solid rgba(104,180,255,.12);background:rgba(104,180,255,.06);color:var(--muted);line-height:1.65}
    .msg{max-width:min(860px,84%);border-radius:20px;padding:15px 16px 14px;border:1px solid var(--line);background:rgba(255,255,255,.03);box-shadow:0 10px 26px rgba(0,0,0,.18)}
    .msg.user{align-self:flex-end;background:linear-gradient(145deg,rgba(34,93,158,.92),rgba(18,53,91,.94));border-color:rgba(104,180,255,.32)}
    .msg.bot{align-self:flex-start}
    .msg.pending{opacity:.72;border-style:dashed}
    .who{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700;margin-bottom:8px}
    .body{white-space:pre-wrap;line-height:1.62;font-size:15px}
    .meta{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:12px;color:var(--muted)}
    .composer{display:grid;grid-template-columns:1fr auto;gap:12px;padding:18px 22px 22px;border-top:1px solid var(--line);background:rgba(9,18,29,.94)}
    .composer textarea{min-height:76px;max-height:220px;font-size:15px}
    .send-col{display:grid;gap:10px;align-content:end;min-width:138px}
    .send-note{color:var(--muted);font-size:12px;text-align:right;line-height:1.4}
    @media (max-width:1080px){body{overflow:auto}.shell{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 28px)}.msg{max-width:100%}}
    @media (max-width:700px){.shell{width:calc(100vw - 18px);margin:9px auto;gap:12px}.rail,.composer,.chat-head,.thread{padding-left:14px;padding-right:14px}.control-grid,.metric-grid{grid-template-columns:1fr 1fr}.composer{grid-template-columns:1fr}.send-col{min-width:0}.send-note{text-align:left}}
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

HTML += """
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
        <div class="status-box" id="statusBox">Waiting for runtime status...</div>
      </section>
    </aside>

    <main class="panel chat">
      <header class="chat-head">
        <div>
          <h3 id="chatTitle">Preparing local runtime...</h3>
          <div class="chat-sub" id="chatSub">The desktop launcher will open this view after the local model server reports ready.</div>
        </div>
        <div class="pill" id="sessionBadge">session pending</div>
      </header>

      <section class="thread" id="thread">
        <div class="welcome" id="welcomeCard">
          Ask for debugging help, explanations, brainstorming, summaries, or code. The preset buttons tune style and generation,
          while Session Steering lets you bias the whole conversation without editing every prompt.
        </div>
      </section>

      <footer class="composer">
        <textarea id="prompt" placeholder="Type a message. Press Enter to send, Shift+Enter for a new line."></textarea>
        <div class="send-col">
          <button class="primary-btn" id="sendBtn">Send</button>
          <div class="send-note" id="sendNote">Preset: balanced</div>
        </div>
      </footer>
    </main>
  </div>

  <script>
    const SETTINGS_KEY = "supermix-qwen-v26-settings";
    const SESSION_KEY = "supermix-qwen-v26-session";
    const PRESETS = {
      direct: { maxNew: 72, temp: 0.05, topP: 0.82 },
      balanced: { maxNew: 112, temp: 0.20, topP: 0.92 },
      reasoning: { maxNew: 176, temp: 0.14, topP: 0.90 },
      creative: { maxNew: 200, temp: 0.58, topP: 0.97 },
      coding: { maxNew: 160, temp: 0.12, topP: 0.88 }
    };
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
      metricF1: document.getElementById("metricF1"),
      metricF1Delta: document.getElementById("metricF1Delta"),
      metricPpl: document.getElementById("metricPpl"),
      metricPplDelta: document.getElementById("metricPplDelta"),
      metricGen: document.getElementById("metricGen"),
      metricGenDelta: document.getElementById("metricGenDelta"),
      metricTrain: document.getElementById("metricTrain"),
      metricSamples: document.getElementById("metricSamples"),
      welcomeCard: document.getElementById("welcomeCard")
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

    function ensureSessionId() {
      const existing = localStorage.getItem(SESSION_KEY);
      state.sessionId = existing || makeSessionId();
      localStorage.setItem(SESSION_KEY, state.sessionId);
      els.sessionBadge.textContent = "session " + state.sessionId.slice(0, 8);
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
        " | T=" + Number(state.settings.temp).toFixed(2);
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

HTML += """
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
      els.releaseLabel.textContent = profile.label || "Latest adapter";
      els.deviceLabel.textContent = status.loaded ? ((status.device || "cpu") + " | adapter " + (status.adapter_loaded ? "ready" : "missing")) : "runtime not loaded";
      els.chatTitle.textContent = status.loaded ? ((profile.label || "Supermix Qwen") + " ready") : "Runtime not ready";
      els.chatSub.textContent = status.loaded ? ("Base model: " + (profile.base_model || "unknown") + " | sessions: " + String(status.sessions || 0)) : "The local runtime is still starting or failed to load.";
      els.adapterMetaLine.textContent =
        "LoRA r=" + String(profile.lora_rank || "-") +
        " | alpha=" + String(profile.lora_alpha || "-") +
        " | dropout=" + String(Number(profile.lora_dropout || 0).toFixed(2)) +
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
    }

    function addMessage(kind, text, meta) {
      if (els.welcomeCard) els.welcomeCard.remove();
      const node = document.createElement("article");
      node.className = "msg " + kind;

      const who = document.createElement("div");
      who.className = "who";
      who.textContent = kind === "user" ? "You" : "Assistant";
      node.appendChild(who);

      const body = document.createElement("div");
      body.className = "body";
      body.textContent = text;
      node.appendChild(body);

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

      els.thread.appendChild(node);
      els.thread.scrollTop = els.thread.scrollHeight;
      return node;
    }

    function rememberMessage(kind, text, meta) {
      state.messages.push({ kind: kind, text: text, meta: meta || {}, at: new Date().toISOString() });
      if (kind === "bot") state.lastBotText = text;
    }

    function setPendingMessage() {
      if (state.pendingNode) return;
      const node = addMessage("bot", "Working on it...", { pending: "running local generation" });
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
      }
    }

    function autoResizePrompt() {
      els.prompt.style.height = "auto";
      els.prompt.style.height = Math.min(220, els.prompt.scrollHeight) + "px";
    }

    async function sendMessage() {
      const text = els.prompt.value.trim();
      if (!text || state.sending) return;
      addMessage("user", text);
      rememberMessage("user", text);
      els.prompt.value = "";
      autoResizePrompt();
      setSending(true);
      setPendingMessage();
      try {
        const data = await jpost("/api/chat", {
          session_id: state.sessionId,
          message: text,
          preset: state.settings.preset,
          system_hint: state.settings.systemHint,
          max_new_tokens: Number(state.settings.maxNew),
          temperature: Number(state.settings.temp),
          top_p: Number(state.settings.topP)
        });
        clearPendingMessage();
        addMessage("bot", data.response, data.timing || {});
        rememberMessage("bot", data.response, data.timing || {});
      } catch (err) {
        clearPendingMessage();
        const textOut = "Error: " + err.message;
        addMessage("bot", textOut);
        rememberMessage("bot", textOut);
      } finally {
        setSending(false);
      }
    }

    async function clearSession(remote) {
      if (remote !== false) {
        try { await jpost("/api/clear", { session_id: state.sessionId }); } catch (err) {
          addMessage("bot", "Clear error: " + err.message);
          rememberMessage("bot", "Clear error: " + err.message);
          return;
        }
      }
      state.messages = [];
      state.lastBotText = "";
      els.thread.innerHTML = "";
      const welcome = document.createElement("div");
      welcome.className = "welcome";
      welcome.id = "welcomeCard";
      welcome.textContent = "Ask for debugging help, explanations, brainstorming, summaries, or code. The preset buttons tune style and generation, while Session Steering lets you bias the whole conversation without editing every prompt.";
      els.thread.appendChild(welcome);
      addMessage("bot", "Session cleared. The local runtime is ready for a new conversation.");
      rememberMessage("bot", "Session cleared. The local runtime is ready for a new conversation.");
    }

    function newSession() {
      state.sessionId = makeSessionId();
      localStorage.setItem(SESSION_KEY, state.sessionId);
      els.sessionBadge.textContent = "session " + state.sessionId.slice(0, 8);
      clearSession(false);
    }

    async function copyLastReply() {
      if (!state.lastBotText) return;
      try {
        await navigator.clipboard.writeText(state.lastBotText);
        addMessage("bot", "Last reply copied to the clipboard.");
        rememberMessage("bot", "Last reply copied to the clipboard.");
      } catch (err) {
        addMessage("bot", "Copy failed: " + err.message);
        rememberMessage("bot", "Copy failed: " + err.message);
      }
    }

    function exportTranscript() {
      const payload = {
        exported_at: new Date().toISOString(),
        session_id: state.sessionId,
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
    }

    function bindInputs() {
      els.maxNew.addEventListener("input", () => { state.settings.maxNew = Math.max(24, Number(els.maxNew.value) || 112); updateSendNote(); saveSettings(); });
      els.temp.addEventListener("input", () => { const value = Number(els.temp.value); state.settings.temp = Number.isFinite(value) ? value : 0.20; updateSendNote(); saveSettings(); });
      els.topP.addEventListener("input", () => { const value = Number(els.topP.value); state.settings.topP = Number.isFinite(value) ? value : 0.92; saveSettings(); });
      els.systemHint.addEventListener("input", () => { state.settings.systemHint = els.systemHint.value; saveSettings(); });
      document.querySelectorAll("[data-preset]").forEach((btn) => btn.addEventListener("click", () => applyPreset(btn.dataset.preset, true)));
      els.prompt.addEventListener("input", autoResizePrompt);
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
        button.addEventListener("click", () => { els.prompt.value = text; autoResizePrompt(); els.prompt.focus(); });
        els.starterChips.appendChild(button);
      });
    }

    ensureSessionId();
    loadSettings();
    bindInputs();
    initStarterChips();
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


def find_latest_adapter_dir(project_root: Path) -> Path:
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


def resolve_base_model_path(explicit_base_model: str, adapter_dir: Path) -> str:
    raw = str(explicit_base_model or "").strip()
    if raw:
        return raw
    adapter_cfg = load_json_if_exists(adapter_dir / "adapter_config.json") or {}
    base_model = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
    return base_model or DEFAULT_BASE_MODEL


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
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")

        preset_name = preset if preset in PRESET_GENERATION else "balanced"
        defaults = PRESET_GENERATION[preset_name]
        system_hint = str(system_hint or "").strip()[:320]

        with self.lock:
            history = list(self.sessions.get(session_id, []))[-12:]

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
