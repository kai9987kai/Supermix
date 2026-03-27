from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from multimodel_catalog import DEFAULT_MODELS_DIR, discover_model_records, models_to_json
from multimodel_runtime import UnifiedModelManager
from qwen_chat_desktop_app import BASE_MODEL_OVERRIDE_ENV


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Studio</title>
  <style>
    :root{
      --bg:#071019;--panel:rgba(10,20,33,.9);--panel-2:rgba(13,25,40,.96);--line:rgba(135,164,203,.15);
      --text:#eef5ff;--muted:#9eb3cf;--blue:#70b8ff;--teal:#63d9c8;--amber:#ffb366;--rose:#ff8d9a;
      --green:#8be1a7;--shadow:0 30px 80px rgba(0,0,0,.34);--r-xl:26px;--r-lg:18px;--r-md:14px;--r-sm:12px;
    }
    *{box-sizing:border-box} html,body{height:100%}
    body{
      margin:0;color:var(--text);overflow:hidden;
      font-family:"Aptos","Bahnschrift","Segoe UI Variable Text","Segoe UI",sans-serif;
      background:
        radial-gradient(circle at 12% 18%, rgba(112,184,255,.18), transparent 24%),
        radial-gradient(circle at 84% 86%, rgba(99,217,200,.14), transparent 26%),
        radial-gradient(circle at 70% 16%, rgba(255,179,102,.12), transparent 22%),
        linear-gradient(160deg,#040b14 0%,#081321 48%,#071019 100%);
    }
    .shell{display:grid;grid-template-columns:400px 1fr;gap:18px;width:min(1560px,calc(100vw - 24px));height:calc(100vh - 24px);margin:12px auto}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-xl);box-shadow:var(--shadow);backdrop-filter:blur(18px);overflow:hidden}
    .side{padding:18px;display:grid;grid-template-rows:auto auto auto auto 1fr;gap:14px;overflow:auto}
    .hero,.card,.thread-card{border-radius:var(--r-lg);border:1px solid var(--line);background:var(--panel-2)}
    .hero{padding:18px;background:linear-gradient(145deg,rgba(18,41,66,.96),rgba(10,20,33,.98))}
    .eyebrow{display:inline-flex;align-items:center;gap:9px;color:var(--blue);font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;margin-bottom:12px}
    .eyebrow::before{content:"";width:10px;height:10px;border-radius:999px;background:var(--blue);box-shadow:0 0 16px rgba(112,184,255,.75)}
    h1{margin:0;font-size:31px;line-height:1.02;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    p{margin:10px 0 0;color:var(--muted);line-height:1.58;font-size:14px}
    .pill-row,.cap-row,.chip-row,.action-row{display:flex;flex-wrap:wrap;gap:8px}
    .pill,.cap,.chip,.ghost,.primary,select,input,textarea{
      border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--text);font:inherit
    }
    .pill,.cap,.chip,.ghost,.primary{display:inline-flex;align-items:center;gap:8px;padding:9px 12px}
    .cap{font-size:12px}
    .cap.chat{border-color:rgba(112,184,255,.28);background:rgba(112,184,255,.08)}
    .cap.image{border-color:rgba(99,217,200,.28);background:rgba(99,217,200,.08)}
    .pill small{color:var(--muted)}
    .card{padding:16px}
    .card h2{margin:0 0 12px;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
    .field{display:grid;gap:6px;margin-bottom:12px}
    .field label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    select,input,textarea{width:100%;padding:12px 13px;border-radius:var(--r-sm);background:rgba(3,10,16,.82)}
    textarea{resize:vertical;min-height:82px;max-height:220px;line-height:1.52}
    select:focus,input:focus,textarea:focus{outline:none;border-color:rgba(112,184,255,.48);box-shadow:0 0 0 3px rgba(112,184,255,.11)}
    .primary{cursor:pointer;background:linear-gradient(135deg,#2f74c5,var(--blue));font-weight:700}
    .ghost{cursor:pointer}
    .stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
    .stat{padding:12px;border-radius:var(--r-md);background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.04)}
    .stat .k{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    .stat .v{margin-top:8px;font-size:24px;line-height:1;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    .stat .d{margin-top:8px;font-size:12px;color:var(--muted);line-height:1.5}
    .model-note,.status-box{padding:12px 13px;border-radius:var(--r-md);background:rgba(4,10,16,.82);border:1px solid rgba(255,255,255,.05);color:#c6d7ec}
    .status-box{white-space:pre-wrap;min-height:116px;max-height:220px;overflow:auto;font-family:Consolas,"Cascadia Code",monospace;font-size:12px;line-height:1.5}
    .chat{display:grid;grid-template-rows:auto 1fr auto;min-height:0}
    .chat-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:18px 22px;border-bottom:1px solid var(--line);background:rgba(12,24,38,.94)}
    .chat-head h3{margin:0;font-size:24px;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    .chat-sub{margin-top:8px;color:var(--muted);font-size:13px;line-height:1.58}
    .thread{padding:20px 22px;overflow:auto;display:flex;flex-direction:column;gap:16px;background:
      radial-gradient(circle at top right, rgba(112,184,255,.08), transparent 24%),
      linear-gradient(180deg, rgba(7,12,20,.48), rgba(10,18,29,.92))
    }
    .welcome{padding:18px;border-radius:18px;border:1px solid rgba(112,184,255,.15);background:rgba(112,184,255,.06);color:var(--muted);line-height:1.62}
    .msg{max-width:min(900px,88%);padding:14px 16px;border-radius:22px;border:1px solid var(--line);background:rgba(255,255,255,.03);box-shadow:0 10px 28px rgba(0,0,0,.16)}
    .msg.user{align-self:flex-end;background:linear-gradient(145deg,rgba(30,82,136,.92),rgba(17,49,84,.95));border-color:rgba(112,184,255,.28)}
    .msg.assistant{align-self:flex-start}
    .msg.pending{opacity:.72;border-style:dashed}
    .msg-top{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}
    .who{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    .msg-meta{display:flex;flex-wrap:wrap;gap:8px}
    .meta-pill{padding:6px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);font-size:12px;color:#d1e1f5}
    .body{white-space:pre-wrap;line-height:1.62;font-size:15px}
    .route{margin-top:10px;color:var(--muted);font-size:12px;line-height:1.5}
    .image-card{display:grid;gap:12px}
    .image-card img{display:block;max-width:min(520px,100%);width:100%;border-radius:18px;border:1px solid rgba(255,255,255,.08);background:#050b13}
    .image-caption{color:#d1e1f5;font-size:13px;line-height:1.58}
    .msg-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
    .mini-btn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--text);cursor:pointer;font:inherit}
    .trace-box{margin-top:12px;padding:12px 13px;border-radius:14px;border:1px solid rgba(255,255,255,.06);background:rgba(6,12,20,.72);color:#c6d7ec;font-size:12px;line-height:1.55;white-space:pre-wrap}
    .composer{padding:18px 22px;border-top:1px solid var(--line);background:rgba(9,18,29,.94);display:grid;grid-template-columns:1fr auto;gap:14px;align-items:end}
    .composer-main{display:grid;gap:10px}
    .mode-row{display:grid;grid-template-columns:180px 180px 1fr;gap:10px}
    .note{font-size:12px;color:var(--muted);line-height:1.45}
    .send-col{display:grid;gap:10px;min-width:150px}
    .toast-rack{position:fixed;right:16px;bottom:16px;display:grid;gap:10px;z-index:20}
    .toast{padding:12px 14px;border-radius:16px;border:1px solid rgba(255,255,255,.08);background:rgba(7,15,24,.94);box-shadow:0 18px 42px rgba(0,0,0,.28);max-width:380px}
    .toast.err{border-color:rgba(255,141,154,.28)} .toast.ok{border-color:rgba(139,225,167,.22)}
    .thread::-webkit-scrollbar,.side::-webkit-scrollbar,.status-box::-webkit-scrollbar{width:8px}
    .thread::-webkit-scrollbar-thumb,.side::-webkit-scrollbar-thumb,.status-box::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:999px}
    @media (max-width:1120px){body{overflow:auto}.shell{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 24px)}}
    @media (max-width:760px){.shell{width:calc(100vw - 16px);margin:8px auto;gap:12px}.thread,.composer,.chat-head,.side{padding-left:14px;padding-right:14px}.stats{grid-template-columns:1fr 1fr}.mode-row{grid-template-columns:1fr}.composer{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel side">
      <section class="hero">
        <div class="eyebrow">Bundled Local Model Studio</div>
        <h1>Supermix Studio</h1>
        <p>One desktop interface for every local zip model, with per-prompt Auto routing and image-capable models rendered directly in the chat thread.</p>
        <div class="pill-row" style="margin-top:14px">
          <div class="pill"><strong id="catalogCount">0</strong><small>models detected</small></div>
          <div class="pill"><strong id="activeBadge">Auto</strong><small>active route</small></div>
        </div>
      </section>

      <section class="card">
        <h2>Model Selector</h2>
        <div class="field">
          <label>Current Model</label>
          <select id="modelSelect"></select>
        </div>
        <div class="field">
          <label>Action Mode</label>
          <select id="actionMode">
            <option value="auto">Auto</option>
            <option value="text">Text</option>
            <option value="image">Image</option>
          </select>
        </div>
        <div class="action-row">
          <button class="primary" id="loadModelBtn">Use Selection</button>
          <button class="ghost" id="refreshBtn">Refresh</button>
          <button class="ghost" id="clearBtn">Clear Chat</button>
        </div>
        <p class="model-note" id="modelNote">Scanning local models...</p>
        <div class="cap-row" id="capabilities"></div>
      </section>

      <section class="card">
        <h2>Runtime Snapshot</h2>
        <div class="stats">
          <div class="stat"><div class="k">Benchmark</div><div class="v" id="statBenchmark">-</div><div class="d" id="statBenchmarkDetail">Saved score</div></div>
          <div class="stat"><div class="k">Family</div><div class="v" id="statFamily">-</div><div class="d" id="statFamilyDetail">Model line</div></div>
          <div class="stat"><div class="k">Package</div><div class="v" id="statPackage">-</div><div class="d" id="statPackageDetail">Zip size</div></div>
          <div class="stat"><div class="k">Active</div><div class="v" id="statActive">-</div><div class="d" id="statActiveDetail">Loaded backend</div></div>
        </div>
      </section>

      <section class="card">
        <h2>Prompt Tuning</h2>
        <div class="field">
          <label>Agent Mode</label>
          <select id="agentMode">
            <option value="off">Off</option>
            <option value="collective">Collective Panel</option>
          </select>
        </div>
        <div class="field">
          <label>Memory Learning</label>
          <select id="memoryMode">
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </div>
        <div class="field">
          <label>Web Search Tool</label>
          <select id="webSearchMode">
            <option value="off">Off</option>
            <option value="on">On</option>
          </select>
        </div>
        <div class="field">
          <label>Style</label>
          <select id="styleMode">
            <option value="auto">Auto</option>
            <option value="balanced">Balanced</option>
            <option value="concise">Concise</option>
            <option value="creative">Creative</option>
            <option value="analyst">Analyst</option>
            <option value="coding">Coding</option>
          </select>
        </div>
        <div class="field">
          <label>System Hint</label>
          <textarea id="systemHint" placeholder="Optional session steering for text models. Example: prefer direct answers and concrete steps."></textarea>
        </div>
        <div class="field">
          <label>Image Style</label>
          <select id="imageStyle">
            <option value="auto">Auto</option>
            <option value="photo">Photo</option>
            <option value="cinematic">Cinematic</option>
            <option value="illustration">Illustration</option>
            <option value="anime">Anime</option>
          </select>
        </div>
        <div class="note">Collective Panel mode consults every chat-capable local model before the final reply. Memory Learning persists session facts and strong prior exchanges, and Web Search gives the models a callable lookup tool for current information.</div>
      </section>

      <section class="card">
        <h2>Exports</h2>
        <div class="field">
          <label>Save Path</label>
          <input id="savePath" placeholder="Optional folder or full .png path">
        </div>
        <div class="action-row">
          <button class="ghost" id="saveChatImageBtn">Save Chat Image</button>
          <button class="ghost" id="saveLastImageBtn">Save Last Image</button>
        </div>
        <div class="note">Use a folder path to keep the generated filename, or a full `.png` path when you want a specific exported file target.</div>
      </section>

      <section class="card">
        <h2>Status</h2>
        <div class="status-box" id="statusBox">Waiting for runtime status...</div>
      </section>

      <section class="card">
        <h2>Memory Snapshot</h2>
        <div class="status-box" id="memoryBox">Waiting for session memory...</div>
      </section>
    </aside>

    <main class="panel chat">
      <header class="chat-head">
        <div>
          <h3 id="chatTitle">Preparing multimodel runtime...</h3>
          <div class="chat-sub" id="chatSub">The dropdown lets you pin one model, while Auto chooses the most appropriate local model for each prompt.</div>
        </div>
        <div class="pill" id="sessionBadge">session pending</div>
      </header>

      <section class="thread" id="thread">
        <div class="welcome" id="welcomeCard">
          Use <strong>Auto</strong> for per-prompt routing, or pin a single model from the dropdown.
          Image-capable models can return generated images directly in this thread.
        </div>
      </section>

      <footer class="composer">
        <div class="composer-main">
          <div class="mode-row">
            <input id="imageWidth" type="number" min="64" max="1024" step="64" value="512" placeholder="Width">
            <input id="imageHeight" type="number" min="64" max="1024" step="64" value="512" placeholder="Height">
            <input id="imageSteps" type="number" min="1" max="4" step="1" value="2" placeholder="Steps">
          </div>
          <textarea id="prompt" placeholder="Type a message, coding question, reasoning task, or image prompt. Press Enter to send, Shift+Enter for a new line."></textarea>
          <div class="chip-row" id="starterChips"></div>
        </div>
        <div class="send-col">
          <button class="primary" id="sendBtn">Send</button>
          <div class="note" id="routeNote">Route: Auto</div>
        </div>
      </footer>
    </main>
  </div>
  <div class="toast-rack" id="toastRack"></div>

  <script>
    const el = (id) => document.getElementById(id);
    const thread = el('thread');
    const sessionKey = 'supermix-studio-session-id';
    let sessionId = localStorage.getItem(sessionKey);
    if(!sessionId){
      sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
      localStorage.setItem(sessionKey, sessionId);
    }
    el('sessionBadge').textContent = 'session ' + sessionId.slice(0, 8);

    const STARTERS = [
      'Debug this stack trace and tell me the most likely root cause.',
      'Compare two ways to solve this problem and tell me the tradeoffs.',
      'Generate a cinematic poster of a storm-battered lighthouse at night.',
      'Rewrite this draft so it sounds more direct and professional.',
      'Give me a compact step-by-step plan to finish this task.'
    ];

    let catalog = [];
    let selectedModelKey = 'auto';
    let transcript = [];
    let lastGeneratedImagePath = '';

    function escapeHtml(value){
      return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function showToast(kind, message){
      const item = document.createElement('div');
      item.className = 'toast ' + (kind || 'ok');
      item.textContent = message;
      el('toastRack').appendChild(item);
      setTimeout(() => item.remove(), 3200);
    }

    async function jget(path){
      const response = await fetch(path);
      const data = await response.json();
      if(!response.ok || data.ok === false){
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      return data;
    }

    async function jpost(path, payload){
      const response = await fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if(!response.ok || data.ok === false){
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      return data;
    }

    function bytesToText(value){
      const num = Number(value || 0);
      if(!num) return '-';
      if(num >= 1024*1024*1024) return (num / (1024*1024*1024)).toFixed(2) + ' GB';
      if(num >= 1024*1024) return (num / (1024*1024)).toFixed(1) + ' MB';
      return num + ' B';
    }

    function buildTranscriptEntry(kind, payload){
      const role = kind === 'user' ? 'user' : 'assistant';
      return {
        role,
        kind: payload.kind || 'text',
        model_label: payload.model_label || (role === 'user' ? 'You' : 'Assistant'),
        response: payload.response || '',
        prompt_used: payload.prompt_used || '',
        output_path: payload.output_path || '',
        image_url: payload.image_url || '',
        route_reason: payload.route_reason || ''
      };
    }

    function scoreText(record){
      if(record.common_overall_exact != null) return record.common_overall_exact.toFixed(3);
      if(record.recipe_eval_accuracy != null) return record.recipe_eval_accuracy.toFixed(3) + ' *';
      return '-';
    }

    function findRecord(key){
      return catalog.find(item => item.key === key) || null;
    }

    function renderCapabilities(record){
      const box = el('capabilities');
      box.innerHTML = '';
      (record?.capabilities || []).forEach(cap => {
        const chip = document.createElement('div');
        chip.className = 'cap ' + cap;
        chip.textContent = cap === 'image' ? 'Image' : 'Chat';
        box.appendChild(chip);
      });
    }

    function updateModelPanel(record){
      if(!record) return;
      el('modelNote').textContent = record.note || record.benchmark_hint || 'No extra note.';
      renderCapabilities(record);
      el('statBenchmark').textContent = scoreText(record);
      el('statBenchmarkDetail').textContent = record.score_source === 'recipe_eval_only' ? 'Recipe holdout only' : (record.common_row_key || 'No benchmark row');
      el('statFamily').textContent = record.family || '-';
      el('statFamilyDetail').textContent = record.kind || '-';
      el('statPackage').textContent = bytesToText(record.zip_size_bytes);
      el('statPackageDetail').textContent = record.zip_name || '';
    }

    function renderCatalog(){
      const select = el('modelSelect');
      select.innerHTML = '';
      catalog.forEach(record => {
        const option = document.createElement('option');
        option.value = record.key;
        option.textContent = record.key === 'auto'
          ? 'Auto'
          : `${record.label} (${record.family}, ${scoreText(record)})`;
        select.appendChild(option);
      });
      select.value = selectedModelKey;
      el('catalogCount').textContent = String(Math.max(0, catalog.length - 1));
      updateModelPanel(findRecord(selectedModelKey) || findRecord('auto'));
    }

    function addMessage(kind, payload){
      const card = document.createElement('div');
      card.className = 'msg ' + kind;
      const metaBadges = [];
      if(payload.model_label) metaBadges.push(`<span class="meta-pill">${escapeHtml(payload.model_label)}</span>`);
      if(payload.kind === 'image') metaBadges.push('<span class="meta-pill">image</span>');
      card.innerHTML = `
        <div class="msg-top">
          <div class="who">${kind === 'user' ? 'You' : 'Assistant'}</div>
          <div class="msg-meta">${metaBadges.join('')}</div>
        </div>
      `;
      if(payload.kind === 'image'){
        const block = document.createElement('div');
        block.className = 'image-card';
        const caption = payload.prompt_used || payload.response || '';
        block.innerHTML = `
          <img src="${escapeHtml(payload.image_url || '')}" alt="generated image">
          <div class="image-caption">${escapeHtml(caption)}</div>
        `;
        card.appendChild(block);
        if(payload.output_path){
          const actions = document.createElement('div');
          actions.className = 'msg-actions';
          actions.innerHTML = `<button class="mini-btn save-image-btn" data-output-path="${escapeHtml(payload.output_path || '')}">Save Image To Path</button>`;
          card.appendChild(actions);
          lastGeneratedImagePath = payload.output_path || lastGeneratedImagePath;
        }
      } else {
        const body = document.createElement('div');
        body.className = 'body';
        body.textContent = payload.response || '';
        card.appendChild(body);
      }
      if(payload.route_reason){
        const route = document.createElement('div');
        route.className = 'route';
        route.textContent = payload.route_reason;
        card.appendChild(route);
      }
      if(payload.agent_trace){
        const blocks = [];
        if((payload.agent_trace.memory_notes || []).length){
          blocks.push('Memory\\n' + payload.agent_trace.memory_notes.map(item => '- ' + item).join('\\n'));
        }
        if((payload.agent_trace.consulted_models || []).length){
          blocks.push('Consulted Models\\n' + payload.agent_trace.consulted_models.join(', '));
        }
        if((payload.agent_trace.tool_events || []).length){
          const searchBits = payload.agent_trace.tool_events.map(event => {
            const domains = (event.results || []).slice(0, 3).map(item => item.domain || item.url || item.title).filter(Boolean).join(', ');
            return `- ${event.query}${domains ? ' -> ' + domains : ''}`;
          });
          blocks.push('Web Search\\n' + searchBits.join('\\n'));
        }
        if(blocks.length){
          const trace = document.createElement('div');
          trace.className = 'trace-box';
          trace.textContent = blocks.join('\\n\\n');
          card.appendChild(trace);
        }
      }
      thread.appendChild(card);
      thread.scrollTop = thread.scrollHeight;
      transcript.push(buildTranscriptEntry(kind, payload));
    }

    async function refresh(){
      try{
        const [statusResp, catalogResp, memoryResp] = await Promise.all([
          jget('/api/status'),
          jget('/api/catalog'),
          jget('/api/memory?session_id=' + encodeURIComponent(sessionId))
        ]);
        catalog = catalogResp.models || [];
        selectedModelKey = statusResp.status.selected_model_key || selectedModelKey || 'auto';
        renderCatalog();
        const activeRecord = findRecord(statusResp.status.active_model_key || selectedModelKey) || findRecord(selectedModelKey) || findRecord('auto');
        updateModelPanel(activeRecord);
        el('statusBox').textContent = JSON.stringify(statusResp.status, null, 2);
        el('memoryBox').textContent = JSON.stringify(memoryResp.memory || {}, null, 2);
        el('activeBadge').textContent = statusResp.status.active_model_label || (selectedModelKey === 'auto' ? 'Auto' : selectedModelKey);
        el('statActive').textContent = statusResp.status.active_model_label || '-';
        el('statActiveDetail').textContent = statusResp.status.device || 'runtime idle';
        el('chatTitle').textContent = statusResp.status.active_model_label
          ? `Supermix Studio - ${statusResp.status.active_model_label}`
          : 'Supermix Studio';
        el('chatSub').textContent = statusResp.status.last_route_reason || 'Auto chooses a model per prompt, while manual selection pins one model.';
        el('routeNote').textContent = 'Route: ' + (selectedModelKey === 'auto' ? 'Auto' : (findRecord(selectedModelKey)?.label || selectedModelKey));
      }catch(error){
        el('statusBox').textContent = 'Status error: ' + error.message;
      }
    }

    async function selectModel(){
      const modelKey = el('modelSelect').value;
      try{
        const data = await jpost('/api/select_model', {model_key: modelKey});
        selectedModelKey = data.status.selected_model_key || modelKey;
        renderCatalog();
        showToast('ok', (modelKey === 'auto' ? 'Auto routing enabled.' : 'Selection updated.'));
        refresh();
      }catch(error){
        showToast('err', error.message);
      }
    }

    async function clearChat(){
      try{
        await jpost('/api/clear', {session_id: sessionId});
        thread.innerHTML = '';
        transcript = [];
        lastGeneratedImagePath = '';
        const welcome = document.createElement('div');
        welcome.className = 'welcome';
        welcome.textContent = 'Session cleared.';
        thread.appendChild(welcome);
      }catch(error){
        showToast('err', error.message);
      }
    }

    function currentSavePath(){
      return el('savePath').value.trim();
    }

    async function exportChatImage(){
      if(!transcript.length){
        showToast('err', 'There is no conversation to export yet.');
        return;
      }
      try{
        const data = await jpost('/api/export_chat_image', {
          session_id: sessionId,
          destination_path: currentSavePath(),
          transcript
        });
        showToast('ok', 'Chat image saved to ' + data.saved_path);
      }catch(error){
        showToast('err', error.message);
      }
    }

    async function saveGeneratedImage(sourcePath){
      if(!sourcePath){
        showToast('err', 'Missing generated image path.');
        return;
      }
      try{
        const data = await jpost('/api/save_generated_image', {
          source_path: sourcePath,
          destination_path: currentSavePath()
        });
        showToast('ok', 'Image saved to ' + data.saved_path);
      }catch(error){
        showToast('err', error.message);
      }
    }

    async function saveLastImage(){
      if(!lastGeneratedImagePath){
        showToast('err', 'There is no generated image in this session yet.');
        return;
      }
      await saveGeneratedImage(lastGeneratedImagePath);
    }

    async function sendPrompt(){
      const prompt = el('prompt').value.trim();
      if(!prompt) return;
      addMessage('user', {response: prompt, kind: 'text'});
      el('prompt').value = '';
      const payload = {
        session_id: sessionId,
        message: prompt,
        model_key: el('modelSelect').value,
        action_mode: el('actionMode').value,
        settings: {
          agent_mode: el('agentMode').value,
          memory_enabled: el('memoryMode').value === 'on',
          web_search_enabled: el('webSearchMode').value === 'on',
          style_mode: el('styleMode').value,
          system_hint: el('systemHint').value,
          image_style: el('imageStyle').value,
          image_width: Number(el('imageWidth').value || 512),
          image_height: Number(el('imageHeight').value || 512),
          image_steps: Number(el('imageSteps').value || 2)
        }
      };
      el('sendBtn').disabled = true;
      el('sendBtn').textContent = 'Working...';
      try{
        const data = await jpost('/api/chat', payload);
        addMessage('assistant', data);
        selectedModelKey = data.selected_model_key || selectedModelKey;
        refresh();
      }catch(error){
        addMessage('assistant', {response: 'Error: ' + error.message, kind: 'text'});
      }finally{
        el('sendBtn').disabled = false;
        el('sendBtn').textContent = 'Send';
      }
    }

    function buildStarterChips(){
      const box = el('starterChips');
      STARTERS.forEach(text => {
        const chip = document.createElement('button');
        chip.className = 'chip';
        chip.textContent = text;
        chip.onclick = () => {
          el('prompt').value = text;
          el('prompt').focus();
        };
        box.appendChild(chip);
      });
    }

    el('loadModelBtn').onclick = selectModel;
    el('refreshBtn').onclick = refresh;
    el('clearBtn').onclick = clearChat;
    el('sendBtn').onclick = sendPrompt;
    el('saveChatImageBtn').onclick = exportChatImage;
    el('saveLastImageBtn').onclick = saveLastImage;
    thread.addEventListener('click', (event) => {
      const button = event.target.closest('.save-image-btn');
      if(!button) return;
      saveGeneratedImage(button.dataset.outputPath || '');
    });
    el('prompt').addEventListener('keydown', (event) => {
      if(event.key === 'Enter' && !event.shiftKey){
        event.preventDefault();
        sendPrompt();
      }
    });

    buildStarterChips();
    refresh();
  </script>
</body>
</html>
"""


def build_app(manager: UnifiedModelManager) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/catalog")
    def api_catalog():
        return jsonify({"ok": True, "models": models_to_json(manager.records)})

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "status": manager.status()})

    @app.get("/api/memory")
    def api_memory():
        session_id = str(request.args.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id is required"}), 400
        return jsonify({"ok": True, "memory": manager.session_memory_snapshot(session_id)})

    @app.post("/api/select_model")
    def api_select_model():
        payload = request.get_json(force=True, silent=True) or {}
        model_key = str(payload.get("model_key") or "auto").strip() or "auto"
        status = manager.select_model(model_key=model_key, eager=False)
        return jsonify({"ok": True, "status": status})

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip() or str(uuid.uuid4())
        prompt = str(payload.get("message") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "error": "message is required"}), 400
        try:
            result = manager.handle_prompt(
                session_id=session_id,
                prompt=prompt,
                model_key=str(payload.get("model_key") or "auto").strip() or "auto",
                action_mode=str(payload.get("action_mode") or "auto").strip().lower(),
                settings=dict(payload.get("settings") or {}),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/clear")
    def api_clear():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id is required"}), 400
        manager.clear(session_id)
        return jsonify({"ok": True, "cleared": True})

    @app.post("/api/save_generated_image")
    def api_save_generated_image():
        payload = request.get_json(force=True, silent=True) or {}
        source_path = str(payload.get("source_path") or "").strip()
        if not source_path:
            return jsonify({"ok": False, "error": "source_path is required"}), 400
        try:
            result = manager.save_generated_image(
                source_path=source_path,
                destination_hint=str(payload.get("destination_path") or "").strip(),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/export_chat_image")
    def api_export_chat_image():
        payload = request.get_json(force=True, silent=True) or {}
        transcript = payload.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            return jsonify({"ok": False, "error": "transcript is required"}), 400
        try:
            result = manager.export_chat_image(
                session_id=str(payload.get("session_id") or "").strip() or str(uuid.uuid4()),
                transcript=transcript,
                destination_hint=str(payload.get("destination_path") or "").strip(),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/generated/<path:model_key>/<path:filename>")
    def generated_file(model_key: str, filename: str):
        target_dir = manager.generated_dir / model_key
        return send_from_directory(target_dir, filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified local Supermix multimodel web app.")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--common_summary", default="")
    parser.add_argument("--qwen_base_model_dir", default="")
    parser.add_argument("--state_dir", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--device_preference", default="cuda,npu,xpu,cpu,dml,mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.qwen_base_model_dir or "").strip():
        os.environ[BASE_MODEL_OVERRIDE_ENV] = str(Path(args.qwen_base_model_dir).resolve())
    models_dir = Path(args.models_dir).resolve()
    common_summary = Path(args.common_summary).resolve() if str(args.common_summary or "").strip() else None
    state_dir = Path(args.state_dir).resolve() if str(args.state_dir or "").strip() else (Path.cwd().resolve() / "run_state_multimodel")
    extraction_root = state_dir / "extracted_models"
    generated_dir = state_dir / "generated_images"

    records = tuple(
        discover_model_records(
            models_dir=models_dir,
            common_summary_path=common_summary if common_summary is not None else Path(),
        )
    ) if common_summary is not None else tuple(discover_model_records(models_dir=models_dir))
    manager = UnifiedModelManager(
        records=records,
        extraction_root=extraction_root,
        generated_dir=generated_dir,
        device_preference=str(args.device_preference),
    )
    app = build_app(manager)
    print(f"Supermix Studio: http://{args.host}:{args.port}")
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
