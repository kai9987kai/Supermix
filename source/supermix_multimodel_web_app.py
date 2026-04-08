from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from multimodel_catalog import DEFAULT_COMMON_SUMMARY, DEFAULT_MODELS_DIR, discover_model_records, models_to_json
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
    .shell{display:grid;grid-template-columns:clamp(320px,23vw,380px) minmax(0,1fr);gap:18px;width:min(1560px,calc(100vw - 24px));height:calc(100vh - 24px);margin:12px auto}
    .shell.focus-chat{grid-template-columns:minmax(0,1fr)}
    .shell.focus-chat .side{display:none}
    .shell.focus-chat .msg{max-width:min(1180px,94%)}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-xl);box-shadow:var(--shadow);backdrop-filter:blur(18px);overflow:hidden}
    .side{padding:18px;display:grid;grid-template-rows:auto auto auto auto 1fr;gap:14px;overflow:auto}
    .hero,.card,.thread-card{border-radius:var(--r-lg);border:1px solid var(--line);background:var(--panel-2)}
    .hero{padding:18px;background:linear-gradient(145deg,rgba(18,41,66,.96),rgba(10,20,33,.98))}
    .eyebrow{display:inline-flex;align-items:center;gap:9px;color:var(--blue);font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;margin-bottom:12px}
    .eyebrow::before{content:"";width:10px;height:10px;border-radius:999px;background:var(--blue);box-shadow:0 0 16px rgba(112,184,255,.75)}
    h1{margin:0;font-size:31px;line-height:1.02;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    p{margin:10px 0 0;color:var(--muted);line-height:1.58;font-size:14px}
    .pill-row,.cap-row,.chip-row,.action-row,.focus-row,.live-row{display:flex;flex-wrap:wrap;gap:8px}
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
    .chat{display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;min-height:0;min-width:0}
    .chat-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:18px 22px;border-bottom:1px solid var(--line);background:rgba(12,24,38,.94)}
    .chat-head h3{margin:0;font-size:24px;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    .chat-sub{margin-top:8px;color:var(--muted);font-size:13px;line-height:1.58}
    .live-strip{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:12px 22px;border-bottom:1px solid rgba(255,255,255,.05);background:rgba(8,16,27,.92)}
    .live-strip .note{margin-top:4px}
    .thread{padding:20px 22px;overflow:auto;display:flex;flex-direction:column;gap:16px;min-width:0;scrollbar-gutter:stable;background:
      radial-gradient(circle at top right, rgba(112,184,255,.08), transparent 24%),
      linear-gradient(180deg, rgba(7,12,20,.48), rgba(10,18,29,.92))
    }
    .thread.compact{padding:14px 18px;gap:10px}
    .thread.compact .msg{padding:11px 13px;border-radius:18px}
    .thread.compact .body{font-size:14px;line-height:1.54}
    .thread.compact .msg-top{margin-bottom:8px}
    .welcome{padding:18px;border-radius:18px;border:1px solid rgba(112,184,255,.15);background:rgba(112,184,255,.06);color:var(--muted);line-height:1.62}
    .msg{max-width:min(980px,92%);padding:14px 16px;border-radius:22px;border:1px solid var(--line);background:rgba(255,255,255,.03);box-shadow:0 10px 28px rgba(0,0,0,.16)}
    .msg.user{align-self:flex-end;background:linear-gradient(145deg,rgba(30,82,136,.92),rgba(17,49,84,.95));border-color:rgba(112,184,255,.28)}
    .msg.assistant{align-self:flex-start}
    .msg.pending{opacity:.72;border-style:dashed}
    .msg-top{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}
    .who{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    .msg-meta{display:flex;flex-wrap:wrap;gap:8px}
    .meta-pill{padding:6px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.04);font-size:12px;color:#d1e1f5}
    .meta-pill.accent{border-color:rgba(99,217,200,.24);background:rgba(99,217,200,.09)}
    .body{white-space:pre-wrap;line-height:1.62;font-size:15px}
    .route{margin-top:10px;color:var(--muted);font-size:12px;line-height:1.5}
    .image-card{display:grid;gap:12px}
    .image-card img{display:block;max-width:min(520px,100%);width:100%;border-radius:18px;border:1px solid rgba(255,255,255,.08);background:#050b13}
    .image-caption{color:#d1e1f5;font-size:13px;line-height:1.58}
    .upload-box{display:grid;gap:10px;padding:12px;border-radius:16px;border:1px solid rgba(255,255,255,.06);background:rgba(4,10,16,.72)}
    .upload-row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
    .upload-preview{display:none;gap:10px;align-items:flex-start}
    .upload-preview img{display:block;max-width:140px;width:140px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:#050b13}
    .upload-meta{display:grid;gap:6px;font-size:12px;color:#c6d7ec;line-height:1.5}
    .msg-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
    .mini-btn{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--text);cursor:pointer;font:inherit}
    .trace-box{margin-top:12px;padding:12px 13px;border-radius:14px;border:1px solid rgba(255,255,255,.06);background:rgba(6,12,20,.72);color:#c6d7ec;font-size:12px;line-height:1.55;white-space:pre-wrap}
    .composer{padding:16px 22px;border-top:1px solid var(--line);background:rgba(9,18,29,.94);display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:stretch;min-height:0}
    .composer-main{display:grid;grid-template-rows:auto minmax(0,1fr);gap:10px;min-width:0;min-height:0}
    .compose-toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center}
    .deck-tabs{display:flex;flex-wrap:wrap;gap:8px}
    .deck-tab{cursor:pointer}
    .deck-tab.active{border-color:rgba(112,184,255,.36);background:rgba(112,184,255,.10)}
    .compose-scroll{display:grid;gap:12px;max-height:min(42vh,430px);overflow:auto;padding-right:4px;min-height:0;scrollbar-gutter:stable}
    .compose-panel{display:grid;gap:10px}
    .compose-panel[hidden]{display:none}
    .workbench-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:12px;align-items:start}
    .response-deck{display:flex;flex-wrap:wrap;gap:8px}
    .mode-row{display:grid;grid-template-columns:180px 180px 1fr;gap:10px}
    .subgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .thread-tools{display:grid;gap:10px}
    .thread-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
    .thread-kpi{padding:10px 11px;border-radius:12px;border:1px solid rgba(255,255,255,.05);background:rgba(255,255,255,.03)}
    .thread-kpi .k{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    .thread-kpi .v{margin-top:6px;font-size:18px;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
    .dispatch-box,.draft-list,.compare-slot,.context-list,.bookmark-list{padding:12px 13px;border-radius:var(--r-md);border:1px solid rgba(255,255,255,.06);background:rgba(4,10,16,.76)}
    .dispatch-box{color:#d7e6fb;font-size:12px;line-height:1.55}
    .dispatch-box strong{display:block;margin-bottom:8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
    .draft-list{display:grid;gap:8px;max-height:240px;overflow:auto}
    .draft-item{padding:11px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.05);background:rgba(255,255,255,.03)}
    .draft-item-top{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
    .draft-item-title{font-size:13px;font-weight:700;color:#eef5ff}
    .draft-item-sub{margin-top:6px;font-size:12px;color:var(--muted);line-height:1.5;white-space:pre-wrap}
    .draft-item-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .compare-slot{display:grid;gap:6px;color:#d7e6fb}
    .compare-slot.empty{color:var(--muted);opacity:.78}
    .compare-slot .slot-head{display:flex;justify-content:space-between;gap:10px;align-items:center}
    .compare-slot .slot-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
    .compare-slot .slot-meta{font-size:12px;color:#d7e6fb}
    .compare-slot .slot-body{font-size:12px;line-height:1.55;white-space:pre-wrap}
    .context-list,.bookmark-list{display:grid;gap:8px;max-height:220px;overflow:auto}
    .context-item,.bookmark-item{padding:11px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.05);background:rgba(255,255,255,.03)}
    .context-item-title,.bookmark-item-title{font-size:12px;font-weight:700;color:#eef5ff}
    .context-item-sub,.bookmark-item-sub{margin-top:6px;font-size:12px;color:var(--muted);line-height:1.5;white-space:pre-wrap}
    .context-item-actions,.bookmark-item-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .msg.bookmarked{box-shadow:0 0 0 1px rgba(255,179,102,.38),0 10px 28px rgba(0,0,0,.16)}
    .focus-chip{cursor:pointer}
    .focus-chip.active{border-color:rgba(112,184,255,.36);background:rgba(112,184,255,.10)}
    .msg.dim{opacity:.28;transform:scale(.992)}
    .composer-meta{display:flex;justify-content:space-between;gap:12px;align-items:center}
    .composer-stats{display:flex;flex-wrap:wrap;gap:8px}
    .composer-stat{padding:6px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.03);font-size:12px;color:#d1e1f5}
    .note{font-size:12px;color:var(--muted);line-height:1.45}
    .send-col{display:grid;grid-template-rows:auto auto 1fr;gap:10px;min-width:170px;align-items:stretch}
    .send-col .primary{justify-content:center;min-height:54px}
    .send-support{display:grid;gap:8px;align-content:start}
    .toast-rack{position:fixed;right:16px;bottom:16px;display:grid;gap:10px;z-index:20}
    .toast{padding:12px 14px;border-radius:16px;border:1px solid rgba(255,255,255,.08);background:rgba(7,15,24,.94);box-shadow:0 18px 42px rgba(0,0,0,.28);max-width:380px}
    .toast.err{border-color:rgba(255,141,154,.28)} .toast.ok{border-color:rgba(139,225,167,.22)}
    .thread::-webkit-scrollbar,.side::-webkit-scrollbar,.status-box::-webkit-scrollbar,.compose-scroll::-webkit-scrollbar,.draft-list::-webkit-scrollbar,.context-list::-webkit-scrollbar,.bookmark-list::-webkit-scrollbar{width:8px}
    .thread::-webkit-scrollbar-thumb,.side::-webkit-scrollbar-thumb,.status-box::-webkit-scrollbar-thumb,.compose-scroll::-webkit-scrollbar-thumb,.draft-list::-webkit-scrollbar-thumb,.context-list::-webkit-scrollbar-thumb,.bookmark-list::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:999px}
    @media (max-height:900px){.compose-scroll{max-height:min(34vh,300px)}}
    @media (max-width:1120px){body{overflow:auto}.shell{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 24px)}.workbench-grid{grid-template-columns:1fr}.compose-toolbar{display:grid}}
    @media (max-width:760px){.shell{width:calc(100vw - 16px);margin:8px auto;gap:12px}.thread,.composer,.chat-head,.side,.live-strip{padding-left:14px;padding-right:14px}.stats,.thread-kpis,.subgrid{grid-template-columns:1fr 1fr}.mode-row,.workbench-grid{grid-template-columns:1fr}.composer{grid-template-columns:1fr}.send-col{min-width:0}.live-strip{display:grid}.compose-toolbar{display:grid}}
  </style>
</head>
<body>
  <div class="shell" id="appShell">
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
            <option value="vision">Vision</option>
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
        <h2>Discovery</h2>
        <div class="field">
          <label>Find Model</label>
          <input id="modelSearch" placeholder="Search family, label, capability, or note">
        </div>
        <div class="field">
          <label>Capability Filter</label>
          <select id="capabilityFilter">
            <option value="all">All Models</option>
            <option value="chat">Chat</option>
            <option value="vision">Vision</option>
            <option value="image">Image</option>
          </select>
        </div>
        <div class="chip-row" id="quickPickChips"></div>
        <div class="note" id="discoveryNote">Use search and capability filters to narrow the installed model catalog.</div>
      </section>

      <section class="card">
        <h2>Model Store</h2>
        <div class="action-row">
          <button class="ghost" id="refreshStoreBtn">Refresh Store</button>
        </div>
        <div class="note" id="modelStoreNote">Browse every published Supermix artifact from Hugging Face and install it into the local model directory.</div>
        <div class="draft-list" id="modelStoreList">Loading remote store...</div>
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

      <section class="card" id="threeDViewerCard" style="display:none">
        <h2>3D Model Viewer</h2>
        <div class="stats">
          <div class="stat"><div class="k">Parameters</div><div class="v" id="threeDStatParams">-</div><div class="d">Small specialist size</div></div>
          <div class="stat"><div class="k">Train Acc</div><div class="v" id="threeDStatTrain">-</div><div class="d">Last training run</div></div>
          <div class="stat"><div class="k">Val Acc</div><div class="v" id="threeDStatVal">-</div><div class="d">Holdout accuracy</div></div>
          <div class="stat"><div class="k">Concepts</div><div class="v" id="threeDStatConcepts">-</div><div class="d">OpenSCAD targets</div></div>
        </div>
        <p class="model-note" id="threeDViewerNote">Select the 3D model to inspect its packaged artifact.</p>
        <div class="action-row" style="margin-top:12px">
          <a class="ghost" id="threeDZipLink" href="#" download>Download Model ZIP</a>
          <a class="ghost" id="threeDSummaryLink" href="#" download>Download Summary JSON</a>
        </div>
        <div class="status-box" id="threeDViewerSummary" style="margin-top:12px">Waiting for 3D model details...</div>
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
        <h2>Session Brief</h2>
        <div class="field">
          <label>Objective</label>
          <input id="sessionObjective" placeholder="What is this chat trying to achieve?">
        </div>
        <div class="field">
          <label>Constraints</label>
          <textarea id="sessionConstraints" placeholder="Deadlines, limits, must-use tools, style constraints, or facts to preserve."></textarea>
        </div>
        <div class="field">
          <label>Done Looks Like</label>
          <input id="sessionDone" placeholder="What would count as a good final answer?">
        </div>
        <div class="action-row">
          <button class="ghost" id="applyBriefBtn">Apply To Hint</button>
          <button class="ghost" id="clearBriefBtn">Clear Brief</button>
        </div>
        <div class="note" id="sessionBriefNote">Keep a compact working brief here. It can be folded into the next prompt without rewriting it every time.</div>
      </section>

      <section class="card">
        <h2>Draft Shelf</h2>
        <div class="field">
          <label>Draft Label</label>
          <input id="draftLabel" placeholder="Optional label for the current prompt draft">
        </div>
        <div class="action-row">
          <button class="ghost" id="saveDraftBtn">Save Draft</button>
          <button class="ghost" id="insertLatestDraftBtn">Insert Latest</button>
        </div>
        <div class="draft-list" id="savedDrafts">No saved drafts yet.</div>
      </section>

      <section class="card">
        <h2>Context Bank</h2>
        <div class="field">
          <label>Manual Context Note</label>
          <textarea id="contextNoteInput" placeholder="Save a fact, requirement, or constraint you want to keep on hand."></textarea>
        </div>
        <div class="action-row">
          <button class="ghost" id="addContextNoteBtn">Add Note</button>
          <button class="ghost" id="captureLastReplyBtn">Capture Last Reply</button>
          <button class="ghost" id="clearContextBankBtn">Clear Context</button>
        </div>
        <div class="context-list" id="contextBankList">No saved context yet.</div>
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
        <h2>Thread Tools</h2>
        <div class="thread-tools">
          <div class="thread-kpis">
            <div class="thread-kpi"><div class="k">Messages</div><div class="v" id="threadMessageCount">0</div></div>
            <div class="thread-kpi"><div class="k">Assistant</div><div class="v" id="threadAssistantCount">0</div></div>
            <div class="thread-kpi"><div class="k">Images</div><div class="v" id="threadImageCount">0</div></div>
          </div>
          <div class="field" style="margin-bottom:0">
            <label>Find In Thread</label>
            <input id="threadFilter" placeholder="Filter replies, prompts, or route notes">
          </div>
          <div class="subgrid">
            <button class="ghost" id="copyLastBtn">Copy Last Reply</button>
            <button class="ghost" id="jumpBottomBtn">Jump To Latest</button>
            <button class="ghost" id="copyThreadBtn">Copy Full Thread</button>
            <button class="ghost" id="downloadThreadBtn">Download JSON</button>
          </div>
          <div class="note">Filter dims non-matching messages. Copy and download actions use the live session transcript you see in the thread.</div>
        </div>
      </section>

      <section class="card">
        <h2>Thread Navigator</h2>
        <div class="bookmark-list" id="threadBookmarks">No bookmarks yet.</div>
        <div class="note" id="bookmarkNote">Bookmark a message in the thread to jump back to it later.</div>
      </section>

      <section class="card">
        <h2>Compare Bench</h2>
        <div class="compare-slot empty" id="compareSlotA">Pin an assistant reply into slot A to compare models, tone, or route decisions.</div>
        <div class="compare-slot empty" id="compareSlotB" style="margin-top:10px">Pin a second assistant reply into slot B for a direct side-by-side view.</div>
        <div class="status-box" id="compareSummary" style="margin-top:12px">Choose two assistant replies to compare structure, route notes, and length.</div>
        <div class="action-row" style="margin-top:12px">
          <button class="ghost" id="swapCompareBtn">Swap A/B</button>
          <button class="ghost" id="clearCompareBtn">Clear Compare</button>
        </div>
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

      <section class="live-strip">
        <div>
          <div class="live-row" id="liveStateChips"></div>
          <div class="note" id="liveStateNote">Checking runtime state...</div>
        </div>
        <div class="composer-stats" id="sessionMetrics"></div>
      </section>

      <section class="thread" id="thread">
        <div class="welcome" id="welcomeCard">
          Use <strong>Auto</strong> for per-prompt routing, or pin a single model from the dropdown.
          Image-capable models can return generated images directly in this thread, and vision-capable models can analyze an uploaded image in the same chat.
        </div>
      </section>

      <footer class="composer">
        <div class="composer-main">
          <div class="compose-toolbar">
            <div class="deck-tabs">
              <button class="mini-btn deck-tab" id="composeQuickBtn" data-compose-tab="quick">Quick</button>
              <button class="mini-btn deck-tab" id="composeMediaBtn" data-compose-tab="media">Media</button>
              <button class="mini-btn deck-tab" id="composeWorkbenchBtn" data-compose-tab="workbench">Workbench</button>
            </div>
            <div class="action-row">
              <button class="ghost" id="toggleSidebarBtn">Focus Layout</button>
              <button class="ghost" id="toggleThreadDensityBtn">Compact Thread</button>
            </div>
          </div>
          <div class="compose-scroll" id="composeScroll">
            <section class="compose-panel" data-compose-panel="quick">
              <textarea id="prompt" placeholder="Type a message, coding question, reasoning task, or image prompt. Press Enter to send, Shift+Enter for a new line."></textarea>
              <div class="chip-row" id="starterChips"></div>
              <div class="focus-row" id="focusChips"></div>
            </section>
            <section class="compose-panel" data-compose-panel="media" hidden>
              <div class="mode-row">
                <input id="imageWidth" type="number" min="64" max="1024" step="64" value="512" placeholder="Width">
                <input id="imageHeight" type="number" min="64" max="1024" step="64" value="512" placeholder="Height">
                <input id="imageSteps" type="number" min="1" max="4" step="1" value="2" placeholder="Steps">
              </div>
              <div class="upload-box" id="uploadBox" style="display:none">
                <div class="upload-row">
                  <input id="imageUpload" type="file" accept="image/*">
                  <button class="ghost" id="uploadBtn">Upload Image</button>
                  <button class="ghost" id="clearUploadBtn">Clear Upload</button>
                </div>
                <div class="note" id="uploadStatus">Select a vision-capable model or Auto to attach an image.</div>
                <div class="upload-preview" id="uploadPreview"></div>
              </div>
            </section>
            <section class="compose-panel" data-compose-panel="workbench" hidden>
              <div class="workbench-grid">
                <div class="dispatch-box" id="dispatchPreview">
                  <strong>Dispatch Preview</strong>
                  Route planning is loading.
                </div>
                <div class="dispatch-box">
                  <strong>Response Deck</strong>
                  <div class="response-deck" id="responseDeck"></div>
                  <div class="note" id="responseDeckNote">Add a response shape or output contract without rewriting the whole prompt.</div>
                </div>
              </div>
              <div class="composer-meta">
                <div class="composer-stats" id="promptStats"></div>
                <div class="note" id="shortcutNote">Enter sends, Shift+Enter adds a new line.</div>
              </div>
            </section>
          </div>
        </div>
        <div class="send-col">
          <button class="primary" id="sendBtn">Send</button>
          <div class="send-support">
            <div class="note" id="routeNote">Route: Auto</div>
            <div class="note" id="composerDockNote">Quick keeps the prompt visible. Media holds image controls. Workbench holds route preview and response shaping.</div>
          </div>
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
    const RESPONSE_DECK = [
      {key:'brief', label:'Brief', text:'Keep the answer compact and high-signal. Prefer a short paragraph or a tight bullet list.'},
      {key:'checklist', label:'Checklist', text:'Return the answer as a practical checklist with only the steps that matter.'},
      {key:'compare', label:'Compare', text:'Structure the answer as a comparison table or clear tradeoff breakdown before recommending one option.'},
      {key:'json', label:'JSON', text:'Return the result as strict JSON with stable field names and no prose outside the JSON block.'},
      {key:'deep', label:'Deep Dive', text:'Take extra time, show the reasoning path explicitly, and include the strongest caveats or failure modes.'},
    ];
    const FOCUS_PACKS = [
      {key:'grounded', label:'Grounded', style:'analyst', hint:'Use only claims you can support from the prompt, chat context, or available tools. Flag uncertainty explicitly.'},
      {key:'think', label:'Think Longer', style:'balanced', hint:'Spend extra time comparing options, checking for contradictions, and tightening the final answer before responding.'},
      {key:'coding', label:'Coding', style:'coding', hint:'Prefer concrete debugging steps, direct fixes, and implementation detail over generic advice.'},
      {key:'compare', label:'Compare', style:'analyst', hint:'Structure the answer as a compact comparison with tradeoffs, risks, and the strongest recommendation.'},
      {key:'creative', label:'Creative', style:'creative', hint:'Keep the answer imaginative but coherent, with vivid detail and a clear final shape.'}
    ];
    const uiStateKey = 'supermix-studio-ui-state-v4';
    const briefMarker = '[Session Brief]';

    let catalog = [];
    let modelStoreRows = [];
    let modelStoreJobs = [];
    let modelStorePollHandle = 0;
    let selectedModelKey = 'auto';
    let transcript = [];
    let lastGeneratedImagePath = '';
    let currentUploadedImagePath = '';
    let currentUploadedImageUrl = '';
    let currentUploadedImageName = '';
    let activeFocusKey = '';
    let composeTab = 'quick';
    let focusLayout = false;
    let compactThread = false;
    let sessionBrief = {objective:'', constraints:'', done:''};
    let savedDrafts = [];
    let contextBank = [];
    let threadBookmarks = [];
    let compareSlots = {a:null, b:null};
    let messageSerial = 0;

    function readUiState(){
      try{
        const raw = localStorage.getItem(uiStateKey);
        if(!raw) return {};
        return JSON.parse(raw) || {};
      }catch(_error){
        return {};
      }
    }

    const bootState = readUiState();
    sessionBrief = Object.assign({}, sessionBrief, bootState.sessionBrief || {});
    savedDrafts = Array.isArray(bootState.savedDrafts) ? bootState.savedDrafts.slice(0, 10) : [];
    contextBank = Array.isArray(bootState.contextBank) ? bootState.contextBank.slice(0, 12) : [];
    composeTab = typeof bootState.composeTab === 'string' ? bootState.composeTab : 'quick';
    focusLayout = Boolean(bootState.focusLayout);
    compactThread = Boolean(bootState.compactThread);

    function escapeHtml(value){
      return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function persistUiState(){
      localStorage.setItem(uiStateKey, JSON.stringify({
        sessionBrief,
        composeTab,
        focusLayout,
        compactThread,
        savedDrafts: savedDrafts.slice(0, 10),
        contextBank: contextBank.slice(0, 12),
      }));
    }

    function textWordCount(value){
      const text = String(value || '').trim();
      return text ? text.split(/\\s+/).length : 0;
    }

    function summarizeText(value, maxLength){
      const text = String(value || '').trim().replace(/\\s+/g, ' ');
      if(!text) return '-';
      if(text.length <= maxLength) return text;
      return text.slice(0, Math.max(0, maxLength - 3)).trimEnd() + '...';
    }

    function currentSessionBrief(){
      return {
        objective: (el('sessionObjective')?.value || '').trim(),
        constraints: (el('sessionConstraints')?.value || '').trim(),
        done: (el('sessionDone')?.value || '').trim(),
      };
    }

    function applySessionBriefInputs(){
      el('sessionObjective').value = sessionBrief.objective || '';
      el('sessionConstraints').value = sessionBrief.constraints || '';
      el('sessionDone').value = sessionBrief.done || '';
    }

    function buildSessionBriefText(source){
      const brief = source || currentSessionBrief();
      const lines = [];
      if(brief.objective) lines.push('Objective: ' + brief.objective);
      if(brief.constraints) lines.push('Constraints: ' + brief.constraints.replace(/\\s*\\n+\\s*/g, ' | '));
      if(brief.done) lines.push('Done: ' + brief.done);
      return lines.join('\\n');
    }

    function stripManagedBriefBlock(value){
      const text = String(value || '');
      const markerIndex = text.indexOf(briefMarker);
      return markerIndex === -1 ? text.trim() : text.slice(0, markerIndex).trim();
    }

    function composeSystemHint(){
      const base = stripManagedBriefBlock(el('systemHint').value || '');
      const brief = buildSessionBriefText();
      return [base, brief ? `${briefMarker}\\n${brief}` : ''].filter(Boolean).join('\\n\\n');
    }

    function syncSessionBrief(){
      sessionBrief = currentSessionBrief();
      persistUiState();
      renderSessionBrief();
      updatePromptStats();
      updateLiveState();
      updateDispatchPreview();
    }

    function clearSessionBrief(){
      sessionBrief = {objective:'', constraints:'', done:''};
      applySessionBriefInputs();
      persistUiState();
      renderSessionBrief();
      updatePromptStats();
      updateLiveState();
      updateDispatchPreview();
    }

    function renderSessionBrief(){
      const brief = currentSessionBrief();
      const parts = [];
      if(brief.objective) parts.push(`Objective: ${brief.objective}`);
      if(brief.constraints) parts.push(`Constraints: ${brief.constraints.replace(/\\s*\\n+\\s*/g, ' | ')}`);
      if(brief.done) parts.push(`Done: ${brief.done}`);
      el('sessionBriefNote').textContent = parts.length
        ? parts.join('\\n')
        : 'Keep a compact working brief here. It can be folded into the next prompt without rewriting it every time.';
    }

    function applyBriefToHint(){
      const composed = composeSystemHint();
      el('systemHint').value = composed;
      updatePromptStats();
      updateDispatchPreview();
      showToast('ok', composed ? 'Session brief folded into system hint.' : 'No brief to apply.');
    }

    function applyLayoutState(){
      el('appShell').classList.toggle('focus-chat', focusLayout);
      el('toggleSidebarBtn').textContent = focusLayout ? 'Show Studio Rail' : 'Focus Layout';
      el('composerDockNote').textContent = focusLayout
        ? 'Focus layout hides the left rail so the thread and composer get the full width.'
        : 'Quick keeps the prompt visible. Media holds image controls. Workbench holds route preview and response shaping.';
    }

    function applyThreadDensity(){
      thread.classList.toggle('compact', compactThread);
      el('toggleThreadDensityBtn').textContent = compactThread ? 'Comfortable Thread' : 'Compact Thread';
    }

    function setComposeTab(nextTab){
      const cooked = ['quick', 'media', 'workbench'].includes(nextTab) ? nextTab : 'quick';
      composeTab = cooked;
      document.querySelectorAll('[data-compose-tab]').forEach((button) => {
        button.classList.toggle('active', button.dataset.composeTab === cooked);
      });
      document.querySelectorAll('[data-compose-panel]').forEach((panel) => {
        panel.hidden = panel.dataset.composePanel !== cooked;
      });
      persistUiState();
    }

    function buildResponseDeck(){
      const box = el('responseDeck');
      box.innerHTML = '';
      RESPONSE_DECK.forEach((item) => {
        const button = document.createElement('button');
        button.className = 'chip';
        button.type = 'button';
        button.textContent = item.label;
        button.onclick = () => {
          const current = (el('prompt').value || '').trim();
          const nextPrompt = [current, item.text].filter(Boolean).join('\\n\\n');
          el('prompt').value = nextPrompt;
          updatePromptStats();
          el('prompt').focus();
          showToast('ok', `${item.label} response shape added.`);
        };
        box.appendChild(button);
      });
    }

    function inferDraftLabel(prompt){
      const compact = summarizeText(prompt, 42);
      return compact === '-' ? 'Untitled draft' : compact;
    }

    function renderSavedDrafts(){
      const box = el('savedDrafts');
      box.innerHTML = '';
      if(!savedDrafts.length){
        box.textContent = 'No saved drafts yet.';
        return;
      }
      savedDrafts.forEach((draft) => {
        const item = document.createElement('div');
        item.className = 'draft-item';
        item.innerHTML = `
          <div class="draft-item-top">
            <div>
              <div class="draft-item-title">${escapeHtml(draft.label || 'Untitled draft')}</div>
              <div class="note">${escapeHtml(draft.model_label || draft.model_key || 'current route')}</div>
            </div>
            <div class="note">${escapeHtml(draft.created_at || '')}</div>
          </div>
          <div class="draft-item-sub">${escapeHtml(summarizeText(draft.prompt, 180))}</div>
        `;
        const actions = document.createElement('div');
        actions.className = 'draft-item-actions';
        const insertBtn = document.createElement('button');
        insertBtn.className = 'mini-btn';
        insertBtn.textContent = 'Insert';
        insertBtn.onclick = () => {
          el('prompt').value = draft.prompt || '';
          if(draft.model_key && findRecord(draft.model_key)){
            selectedModelKey = draft.model_key;
            el('modelSelect').value = draft.model_key;
            updateModelPanel(findRecord(draft.model_key));
            refreshUploadPanel();
          }
          if(draft.style_mode) el('styleMode').value = draft.style_mode;
          updatePromptStats();
          updateLiveState();
          updateDispatchPreview();
          el('prompt').focus();
        };
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'mini-btn';
        deleteBtn.textContent = 'Delete';
        deleteBtn.onclick = () => {
          savedDrafts = savedDrafts.filter((itemDraft) => itemDraft.id !== draft.id);
          persistUiState();
          renderSavedDrafts();
          updateDispatchPreview();
        };
        actions.appendChild(insertBtn);
        actions.appendChild(deleteBtn);
        item.appendChild(actions);
        box.appendChild(item);
      });
    }

    function saveCurrentDraft(){
      const prompt = (el('prompt').value || '').trim();
      if(!prompt){
        showToast('err', 'Write a prompt before saving a draft.');
        return;
      }
      const record = findRecord(el('modelSelect').value) || findRecord(selectedModelKey) || findRecord('auto');
      const label = (el('draftLabel').value || '').trim() || inferDraftLabel(prompt);
      const draft = {
        id: String(Date.now()),
        label,
        prompt,
        model_key: record?.key || selectedModelKey,
        model_label: record?.label || record?.key || 'Auto',
        style_mode: el('styleMode').value,
        created_at: new Date().toLocaleString(),
      };
      savedDrafts = [draft, ...savedDrafts.filter((item) => item.prompt !== prompt)].slice(0, 10);
      persistUiState();
      renderSavedDrafts();
      updateDispatchPreview();
      showToast('ok', `Saved draft: ${label}`);
    }

    function renderContextBank(){
      const box = el('contextBankList');
      box.innerHTML = '';
      if(!contextBank.length){
        box.textContent = 'No saved context yet.';
        return;
      }
      contextBank.forEach((entry) => {
        const item = document.createElement('div');
        item.className = 'context-item';
        item.innerHTML = `
          <div class="context-item-title">${escapeHtml(entry.label || 'Context')}</div>
          <div class="context-item-sub">${escapeHtml(summarizeText(entry.text, 190))}</div>
        `;
        const actions = document.createElement('div');
        actions.className = 'context-item-actions';

        const insertBtn = document.createElement('button');
        insertBtn.className = 'mini-btn';
        insertBtn.textContent = 'Insert';
        insertBtn.onclick = () => {
          const current = (el('prompt').value || '').trim();
          el('prompt').value = [current, entry.text].filter(Boolean).join('\\n\\n');
          updatePromptStats();
          el('prompt').focus();
        };

        const hintBtn = document.createElement('button');
        hintBtn.className = 'mini-btn';
        hintBtn.textContent = 'To Hint';
        hintBtn.onclick = () => {
          const base = stripManagedBriefBlock(el('systemHint').value || '');
          el('systemHint').value = [base, entry.text].filter(Boolean).join('\\n\\n');
          updatePromptStats();
          updateDispatchPreview();
        };

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'mini-btn';
        deleteBtn.textContent = 'Delete';
        deleteBtn.onclick = () => {
          contextBank = contextBank.filter((row) => row.id !== entry.id);
          persistUiState();
          renderContextBank();
          updatePromptStats();
          updateLiveState();
        };

        actions.appendChild(insertBtn);
        actions.appendChild(hintBtn);
        actions.appendChild(deleteBtn);
        item.appendChild(actions);
        box.appendChild(item);
      });
    }

    function addContextEntry(text, label){
      const clean = String(text || '').trim();
      if(!clean) return false;
      const entry = {
        id: String(Date.now() + Math.random()),
        label: label || inferDraftLabel(clean),
        text: clean,
      };
      contextBank = [entry, ...contextBank.filter((item) => item.text !== clean)].slice(0, 12);
      persistUiState();
      renderContextBank();
      updatePromptStats();
      updateLiveState();
      updateDispatchPreview();
      return true;
    }

    function addManualContext(){
      const text = (el('contextNoteInput').value || '').trim();
      if(!text){
        showToast('err', 'Write a context note first.');
        return;
      }
      if(addContextEntry(text, 'Manual note')){
        el('contextNoteInput').value = '';
        showToast('ok', 'Context note saved.');
      }
    }

    function captureLastAssistantContext(){
      const last = [...transcript].reverse().find((item) => item.role === 'assistant' && item.response);
      if(!last){
        showToast('err', 'No assistant reply to capture yet.');
        return;
      }
      if(addContextEntry(last.response || '', last.model_label || 'Last reply')){
        showToast('ok', 'Last reply added to context bank.');
      }
    }

    function clearContextBank(){
      contextBank = [];
      persistUiState();
      renderContextBank();
      updatePromptStats();
      updateLiveState();
      updateDispatchPreview();
      showToast('ok', 'Context bank cleared.');
    }

    function renderThreadBookmarks(){
      const box = el('threadBookmarks');
      box.innerHTML = '';
      if(!threadBookmarks.length){
        box.textContent = 'No bookmarks yet.';
        el('bookmarkNote').textContent = 'Bookmark a message in the thread to jump back to it later.';
        return;
      }
      threadBookmarks.forEach((entry) => {
        const item = document.createElement('div');
        item.className = 'bookmark-item';
        item.innerHTML = `
          <div class="bookmark-item-title">${escapeHtml(entry.label || 'Bookmark')}</div>
          <div class="bookmark-item-sub">${escapeHtml(summarizeText(entry.snippet, 150))}</div>
        `;
        const actions = document.createElement('div');
        actions.className = 'bookmark-item-actions';

        const jumpBtn = document.createElement('button');
        jumpBtn.className = 'mini-btn';
        jumpBtn.textContent = 'Jump';
        jumpBtn.onclick = () => {
          const node = document.getElementById(entry.message_id);
          if(node){
            node.scrollIntoView({behavior:'smooth', block:'center'});
            node.classList.remove('dim');
          }
        };

        const removeBtn = document.createElement('button');
        removeBtn.className = 'mini-btn';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => {
          threadBookmarks = threadBookmarks.filter((itemRow) => itemRow.id !== entry.id);
          renderThreadBookmarks();
          const node = document.getElementById(entry.message_id);
          if(node){
            node.classList.remove('bookmarked');
          }
        };

        actions.appendChild(jumpBtn);
        actions.appendChild(removeBtn);
        item.appendChild(actions);
        box.appendChild(item);
      });
      el('bookmarkNote').textContent = `${threadBookmarks.length} bookmark${threadBookmarks.length === 1 ? '' : 's'} in this thread.`;
    }

    function addThreadBookmark(card){
      if(!card) return;
      const messageId = card.dataset.messageId || '';
      const snippet = card.dataset.messageText || '';
      if(!messageId || !snippet){
        showToast('err', 'Nothing bookmarkable on this message.');
        return;
      }
      const label = card.dataset.messageLabel || card.dataset.messageRole || 'Message';
      const entry = {
        id: String(Date.now() + Math.random()),
        message_id: messageId,
        label,
        snippet,
      };
      threadBookmarks = [entry, ...threadBookmarks.filter((item) => item.message_id !== messageId)].slice(0, 14);
      card.classList.add('bookmarked');
      renderThreadBookmarks();
      showToast('ok', 'Thread bookmark saved.');
    }

    function renderCompareSlot(nodeId, slotLabel, payload){
      const node = el(nodeId);
      if(!payload){
        node.className = 'compare-slot empty';
        node.textContent = `Pin an assistant reply into slot ${slotLabel} to compare models, tone, or route decisions.`;
        return;
      }
      node.className = 'compare-slot';
      node.innerHTML = `
        <div class="slot-head">
          <div class="slot-label">Slot ${slotLabel}</div>
          <div class="slot-meta">${escapeHtml(payload.model_label || 'Assistant')} | ${payload.word_count} words</div>
        </div>
        <div class="slot-body">${escapeHtml(summarizeText(payload.response, 200))}</div>
        <div class="note">${escapeHtml(payload.route_reason || 'No route note saved for this reply.')}</div>
      `;
    }

    function renderCompareBench(){
      renderCompareSlot('compareSlotA', 'A', compareSlots.a);
      renderCompareSlot('compareSlotB', 'B', compareSlots.b);
      const a = compareSlots.a;
      const b = compareSlots.b;
      if(!a || !b){
        el('compareSummary').textContent = 'Choose two assistant replies to compare structure, route notes, and length.';
        return;
      }
      const wordDelta = Math.abs((a.word_count || 0) - (b.word_count || 0));
      const longer = (a.word_count || 0) === (b.word_count || 0)
        ? 'same length'
        : ((a.word_count || 0) > (b.word_count || 0) ? `A longer by ${wordDelta} words` : `B longer by ${wordDelta} words`);
      const modelDelta = a.model_label === b.model_label ? 'same model family' : `${a.model_label} vs ${b.model_label}`;
      const routeDelta = a.route_reason && b.route_reason
        ? (a.route_reason === b.route_reason ? 'same route note' : 'different route notes')
        : 'one or both route notes missing';
      el('compareSummary').textContent = [
        `Models: ${modelDelta}`,
        `Length: ${longer}`,
        `Routing: ${routeDelta}`,
        `A preview: ${summarizeText(a.response, 140)}`,
        `B preview: ${summarizeText(b.response, 140)}`,
      ].join('\\n\\n');
    }

    function snapshotAssistantReply(payload){
      return {
        model_label: payload.model_label || 'Assistant',
        response: payload.response || '',
        route_reason: payload.route_reason || '',
        word_count: textWordCount(payload.response || ''),
      };
    }

    function updateDispatchPreview(){
      const record = findRecord(el('modelSelect').value || selectedModelKey) || findRecord(selectedModelKey) || findRecord('auto');
      const prompt = (el('prompt').value || '').trim();
      const lines = [];
      lines.push(`Route: ${record?.key === 'auto' ? 'Auto chooser' : (record?.label || record?.key || 'Auto')}`);
      lines.push(`Mode: ${el('actionMode').value} | Agent: ${el('agentMode').value} | Style: ${el('styleMode').value}`);
      lines.push(`Payload: ${textWordCount(prompt)} words${currentUploadedImagePath ? ' | image attached' : ''}${activeFocusKey ? ` | focus ${activeFocusKey}` : ''}`);
      const brief = buildSessionBriefText();
      if(brief) lines.push(`Brief: ${brief.replace(/\\n+/g, ' | ')}`);
      if(contextBank.length){
        const preview = contextBank.slice(0, 3).map((item) => item.label || inferDraftLabel(item.text)).join(', ');
        lines.push(`Context bank: ${contextBank.length} saved${preview ? ` | ${preview}` : ''}`);
      }
      const hint = stripManagedBriefBlock(el('systemHint').value || '');
      if(hint) lines.push(`Hint: ${summarizeText(hint, 170)}`);
      el('dispatchPreview').innerHTML = `<strong>Dispatch Preview</strong>${escapeHtml(lines.join('\\n')).replaceAll('\\n', '<br>')}`;
    }

    function showToast(kind, message){
      const item = document.createElement('div');
      item.className = 'toast ' + (kind || 'ok');
      item.textContent = message;
      el('toastRack').appendChild(item);
      setTimeout(() => item.remove(), 3200);
    }

    async function copyText(text, successMessage){
      try{
        await navigator.clipboard.writeText(text);
        showToast('ok', successMessage || 'Copied.');
      }catch(error){
        showToast('err', 'Clipboard error: ' + error.message);
      }
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
        kind: payload.kind || (payload.uploaded_image_url ? 'image' : 'text'),
        model_label: payload.model_label || (role === 'user' ? 'You' : 'Assistant'),
        response: payload.response || '',
        prompt_used: payload.prompt_used || payload.response || '',
        output_path: payload.output_path || payload.uploaded_image_path || '',
        image_url: payload.image_url || payload.uploaded_image_url || '',
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

    function filteredCatalogRecords(){
      const search = (el('modelSearch')?.value || '').trim().toLowerCase();
      const capability = (el('capabilityFilter')?.value || 'all').trim().toLowerCase();
      let rows = catalog.filter((record) => {
        if(capability !== 'all' && !(record.capabilities || []).includes(capability)) return false;
        if(!search) return true;
        const hay = [
          record.key,
          record.label,
          record.family,
          record.kind,
          record.note,
          record.benchmark_hint,
          (record.capabilities || []).join(' ')
        ].join(' ').toLowerCase();
        return hay.includes(search);
      });
      const selected = findRecord(selectedModelKey);
      if(selected && !rows.some((item) => item.key === selected.key)){
        rows = [selected, ...rows];
      }
      const autoRecord = findRecord('auto');
      if(autoRecord && !rows.some((item) => item.key === 'auto')){
        rows = [autoRecord, ...rows];
      }
      return rows;
    }

    function catalogBestRecord(predicate){
      const matches = catalog.filter((record) => record.key !== 'auto' && (!predicate || predicate(record)));
      if(!matches.length) return null;
      return matches.sort((a, b) => {
        const scoreA = Number(a.common_overall_exact ?? a.recipe_eval_accuracy ?? -1);
        const scoreB = Number(b.common_overall_exact ?? b.recipe_eval_accuracy ?? -1);
        return scoreB - scoreA;
      })[0];
    }

    function buildQuickPickRows(){
      const rows = [];
      const overall = catalogBestRecord(() => true);
      if(overall) rows.push({label:'Top Benchmark', key:overall.key});
      const image = catalogBestRecord((record) => (record.capabilities || []).includes('image'));
      if(image) rows.push({label:'Best Image', key:image.key});
      const vision = catalogBestRecord((record) => (record.capabilities || []).includes('vision'));
      if(vision) rows.push({label:'Best Vision', key:vision.key});
      const omni = catalogBestRecord((record) => String(record.family || '').toLowerCase().includes('omni'));
      if(omni) rows.push({label:'Latest Omni', key:omni.key});
      const specialist3d = catalog.find((record) => record.key === 'three_d_generation_micro_v1');
      if(specialist3d) rows.push({label:'3D Specialist', key:specialist3d.key});
      return rows;
    }

    function renderQuickPickChips(){
      const box = el('quickPickChips');
      box.innerHTML = '';
      buildQuickPickRows().forEach((item) => {
        const button = document.createElement('button');
        button.className = 'ghost';
        button.textContent = item.label;
        button.onclick = () => {
          selectedModelKey = item.key;
          el('modelSelect').value = item.key;
          updateModelPanel(findRecord(item.key));
          updateLiveState();
          showToast('ok', `Selected ${findRecord(item.key)?.label || item.key}.`);
        };
        box.appendChild(button);
      });
    }

    function updateDiscoveryNote(rows){
      const capability = (el('capabilityFilter')?.value || 'all').trim().toLowerCase();
      const search = (el('modelSearch')?.value || '').trim();
      const topVisible = catalogBestRecord((record) => rows.some((item) => item.key === record.key));
      const parts = [`${Math.max(0, rows.filter((row) => row.key !== 'auto').length)} visible`];
      parts.push(capability === 'all' ? 'all capabilities' : capability);
      if(search) parts.push(`search: ${search}`);
      if(topVisible) parts.push(`top visible: ${topVisible.label}`);
      el('discoveryNote').textContent = parts.join(' | ');
    }

    function latestStoreJob(fileName){
      return (modelStoreJobs || []).find((job) => job.file_name === fileName) || null;
    }

    function renderModelStore(){
      const box = el('modelStoreList');
      box.innerHTML = '';
      if(!modelStoreRows.length){
        box.textContent = 'No remote store entries loaded yet.';
        return;
      }
      const activeJobs = (modelStoreJobs || []).filter((job) => ['queued', 'downloading'].includes(job.status || '')).length;
      el('modelStoreNote').textContent = activeJobs
        ? `${activeJobs} install job${activeJobs === 1 ? '' : 's'} running. Installed files refresh the local catalog automatically.`
        : 'Browse every published Supermix artifact from Hugging Face and install it into the local model directory.';
      modelStoreRows.forEach((row) => {
        const item = document.createElement('div');
        item.className = 'draft-item';
        const job = latestStoreJob(row.file_name);
        const jobStatus = job ? String(job.status || '') : '';
        const progress = job && Number(job.total_bytes || 0) > 0
          ? `${Math.min(100, Math.round((Number(job.downloaded_bytes || 0) / Number(job.total_bytes || 1)) * 100))}%`
          : '';
        const statusText = row.installed
          ? (row.selectable ? 'Installed and selectable' : 'Downloaded locally')
          : (jobStatus ? `${jobStatus}${progress ? ` ${progress}` : ''}` : 'Remote only');
        item.innerHTML = `
          <div class="draft-item-top">
            <div>
              <div class="draft-item-title">${escapeHtml(row.label || row.file_name)}</div>
              <div class="note">${escapeHtml(row.file_name)}</div>
            </div>
            <div class="note">${escapeHtml(bytesToText(row.size_bytes || 0))}</div>
          </div>
          <div class="draft-item-sub">${escapeHtml([
            row.family || 'other',
            row.known ? ((row.capabilities || []).join(', ') || row.kind || 'known artifact') : 'download-only artifact',
            statusText
          ].filter(Boolean).join(' | '))}</div>
        `;
        if(row.note){
          const note = document.createElement('div');
          note.className = 'note';
          note.textContent = row.note;
          item.appendChild(note);
        }
        if(job && job.error){
          const err = document.createElement('div');
          err.className = 'note';
          err.textContent = 'Install error: ' + job.error;
          item.appendChild(err);
        }
        const actions = document.createElement('div');
        actions.className = 'draft-item-actions';
        const openBtn = document.createElement('a');
        openBtn.className = 'mini-btn';
        openBtn.href = row.download_url || '#';
        openBtn.target = '_blank';
        openBtn.rel = 'noreferrer';
        openBtn.textContent = 'Open Remote';
        actions.appendChild(openBtn);

        const installBtn = document.createElement('button');
        installBtn.className = 'mini-btn';
        installBtn.textContent = row.installed ? 'Installed' : (jobStatus === 'downloading' ? 'Downloading...' : 'Install');
        installBtn.disabled = Boolean(row.installed || ['queued', 'downloading'].includes(jobStatus));
        installBtn.onclick = () => installStoreModel(row.file_name);
        actions.appendChild(installBtn);
        item.appendChild(actions);
        box.appendChild(item);
      });
    }

    async function refreshModelStore(force=false){
      const suffix = force ? '?refresh=1' : '';
      try{
        const [storeResp, jobsResp] = await Promise.all([
          jget('/api/model_store' + suffix),
          jget('/api/model_store/jobs')
        ]);
        modelStoreRows = storeResp.models || [];
        modelStoreJobs = jobsResp.jobs || [];
        renderModelStore();
        if(modelStorePollHandle){
          clearTimeout(modelStorePollHandle);
          modelStorePollHandle = 0;
        }
        if((modelStoreJobs || []).some((job) => ['queued', 'downloading'].includes(job.status || ''))){
          modelStorePollHandle = setTimeout(() => refreshModelStore(false), 3000);
        }
      }catch(error){
        el('modelStoreNote').textContent = 'Model store error: ' + error.message;
        el('modelStoreList').textContent = 'Unable to reach the remote model store right now.';
      }
    }

    async function installStoreModel(fileName){
      try{
        const data = await jpost('/api/model_store/install', {file_name: fileName});
        showToast('ok', `Install started for ${fileName}.`);
        modelStoreJobs = [data.job || {}, ...(modelStoreJobs || []).filter((job) => job.job_id !== data.job?.job_id)];
        renderModelStore();
        refreshModelStore(false);
        refresh();
      }catch(error){
        showToast('err', error.message);
      }
    }

    function renderCapabilities(record){
      const box = el('capabilities');
      box.innerHTML = '';
      (record?.capabilities || []).forEach(cap => {
        const chip = document.createElement('div');
        chip.className = 'cap ' + cap;
        chip.textContent = cap === 'image' ? 'Image' : (cap === 'vision' ? 'Vision' : 'Chat');
        box.appendChild(chip);
      });
    }

    function shouldShowUpload(record){
      if(!record) return false;
      if(record.key === 'auto') return true;
      if((record.capabilities || []).includes('vision')) return true;
      return el('agentMode').value === 'collective';
    }

    function renderUploadPreview(){
      const box = el('uploadPreview');
      if(!currentUploadedImageUrl){
        box.style.display = 'none';
        box.innerHTML = '';
        return;
      }
      box.style.display = 'grid';
      box.innerHTML = `
        <img src="${escapeHtml(currentUploadedImageUrl)}" alt="uploaded image">
        <div class="upload-meta">
          <strong>${escapeHtml(currentUploadedImageName || 'uploaded image')}</strong>
          <div>${escapeHtml(currentUploadedImagePath)}</div>
        </div>
      `;
    }

    function refreshUploadPanel(){
      const record = findRecord(el('modelSelect').value) || findRecord(selectedModelKey) || findRecord('auto');
      const visible = shouldShowUpload(record);
      el('uploadBox').style.display = visible ? 'grid' : 'none';
      if(!visible){
        el('uploadStatus').textContent = 'Select a vision-capable model or Auto to attach an image.';
      } else if(currentUploadedImagePath){
        el('uploadStatus').textContent = 'Uploaded image is attached to the next prompt.';
      } else {
        el('uploadStatus').textContent = 'Upload an image if you want the vision models to analyze it.';
      }
      renderUploadPreview();
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
      refreshThreeDViewer(findRecord(el('modelSelect').value) || record);
    }

    function summarizeTranscript(){
      const messages = transcript.length;
      const assistant = transcript.filter(item => item.role === 'assistant').length;
      const images = transcript.filter(item => item.kind === 'image' || item.image_url).length;
      el('threadMessageCount').textContent = String(messages);
      el('threadAssistantCount').textContent = String(assistant);
      el('threadImageCount').textContent = String(images);
      const sessionBits = [
        `${messages} msgs`,
        `${assistant} replies`,
        images ? `${images} image outputs` : 'text-first session'
      ];
      el('sessionMetrics').innerHTML = sessionBits.map((bit) => `<div class="composer-stat">${escapeHtml(bit)}</div>`).join('');
    }

    function updatePromptStats(){
      const text = el('prompt').value || '';
      const words = text.trim() ? text.trim().split(/\\s+/).length : 0;
      const chars = text.length;
      const brief = buildSessionBriefText();
      const bits = [
        `${words} words`,
        `${chars} chars`,
        currentUploadedImagePath ? 'image attached' : 'no image',
        activeFocusKey ? `focus: ${activeFocusKey}` : 'focus: standard',
        brief ? 'brief active' : 'brief off',
        contextBank.length ? `context ${contextBank.length}` : 'context off'
      ];
      el('promptStats').innerHTML = bits.map((bit) => `<div class="composer-stat">${escapeHtml(bit)}</div>`).join('');
      updateDispatchPreview();
    }

    function updateLiveState(status){
      const chips = [];
      chips.push(selectedModelKey === 'auto' ? 'Auto route' : (findRecord(selectedModelKey)?.label || selectedModelKey));
      chips.push('action ' + el('actionMode').value);
      chips.push(el('agentMode').value === 'collective' ? 'collective panel' : 'single reply');
      chips.push(el('memoryMode').value === 'on' ? 'memory on' : 'memory off');
      chips.push(el('webSearchMode').value === 'on' ? 'web tool on' : 'web tool off');
      if(buildSessionBriefText()) chips.push('brief armed');
      if(contextBank.length) chips.push(`context ${contextBank.length}`);
      if(currentUploadedImagePath) chips.push('image attached');
      el('liveStateChips').innerHTML = chips.map((bit, idx) => `<div class="meta-pill ${idx === 0 ? 'accent' : ''}">${escapeHtml(bit)}</div>`).join('');
      el('liveStateNote').textContent = status?.last_route_reason || 'Use focus packs to bias how the assistant approaches the next prompt.';
      updateDispatchPreview();
    }

    function applyThreadFilter(){
      const query = (el('threadFilter').value || '').trim().toLowerCase();
      thread.querySelectorAll('.msg').forEach((node) => {
        const text = (node.dataset.search || '').toLowerCase();
        node.classList.toggle('dim', Boolean(query) && !text.includes(query));
      });
    }

    function transcriptAsText(){
      return transcript.map((item, index) => {
        const role = item.role === 'assistant' ? 'Assistant' : 'You';
        const route = item.route_reason ? `\\nRoute: ${item.route_reason}` : '';
        const content = item.kind === 'image'
          ? (item.prompt_used || item.response || '[image output]')
          : (item.response || '');
        return `${index + 1}. ${role} (${item.model_label || '-'})\\n${content}${route}`;
      }).join('\\n\\n');
    }

    function downloadTranscriptJson(){
      const blob = new Blob([JSON.stringify({session_id: sessionId, transcript}, null, 2)], {type:'application/json'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `supermix_thread_${sessionId.slice(0, 8)}.json`;
      document.body.appendChild(link);
      link.click();
      setTimeout(() => {
        URL.revokeObjectURL(link.href);
        link.remove();
      }, 0);
    }

    function jumpToLatest(){
      thread.scrollTop = thread.scrollHeight;
    }

    function formatMetric(value){
      if(value == null || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(3);
    }

    function setThreeDDownloadLink(id, href, label){
      const link = el(id);
      link.href = href || '#';
      link.textContent = label;
      link.style.pointerEvents = href ? 'auto' : 'none';
      link.style.opacity = href ? '1' : '.45';
    }

    async function refreshThreeDViewer(record){
      const card = el('threeDViewerCard');
      if(!record || record.key !== 'three_d_generation_micro_v1'){
        card.style.display = 'none';
        return;
      }
      card.style.display = 'block';
      el('threeDViewerSummary').textContent = 'Loading packaged 3D model details...';
      try{
        const data = await jget('/api/three_d_model_view');
        const model = data.model || {};
        el('threeDStatParams').textContent = model.parameter_count != null ? String(model.parameter_count) : '-';
        el('threeDStatTrain').textContent = formatMetric(model.train_accuracy);
        el('threeDStatVal').textContent = formatMetric(model.val_accuracy);
        el('threeDStatConcepts').textContent = model.concept_count != null ? String(model.concept_count) : '-';
        const counts = [
          model.zip_name || '',
          model.source_rows != null ? `${model.source_rows} source rows` : '',
          model.train_rows != null ? `${model.train_rows} train` : '',
          model.val_rows != null ? `${model.val_rows} val` : ''
        ].filter(Boolean).join(' | ');
        const concepts = (model.concept_labels || []).slice(0, 10).join(', ');
        el('threeDViewerNote').textContent = [counts, concepts ? `Concepts: ${concepts}` : ''].filter(Boolean).join('\\n');
        const samples = (model.sample_predictions || []).map((item) => {
          const confidence = item.confidence != null ? ` (${formatMetric(item.confidence)})` : '';
          return `${item.predicted_label || item.predicted_concept || 'prediction'}${confidence}\\nPrompt: ${item.prompt || ''}`;
        });
        el('threeDViewerSummary').textContent = samples.length
          ? samples.join('\\n\\n')
          : 'No sample predictions were packaged with this model.';
        setThreeDDownloadLink('threeDZipLink', model.download_zip_url || '', 'Download Model ZIP');
        setThreeDDownloadLink('threeDSummaryLink', model.download_summary_url || '', 'Download Summary JSON');
      }catch(error){
        el('threeDViewerSummary').textContent = '3D model viewer error: ' + error.message;
        setThreeDDownloadLink('threeDZipLink', '', 'Download Model ZIP');
        setThreeDDownloadLink('threeDSummaryLink', '', 'Download Summary JSON');
      }
    }

    function renderCatalog(){
      const select = el('modelSelect');
      select.innerHTML = '';
      const rows = filteredCatalogRecords();
      rows.forEach(record => {
        const option = document.createElement('option');
        option.value = record.key;
        option.textContent = record.key === 'auto'
          ? 'Auto'
          : `${record.label} (${record.family}, ${scoreText(record)})`;
        select.appendChild(option);
      });
      if(rows.length && !rows.some((item) => item.key === selectedModelKey)){
        selectedModelKey = rows[0].key;
      }
      select.value = selectedModelKey;
      el('catalogCount').textContent = String(Math.max(0, catalog.length - 1));
      updateDiscoveryNote(rows);
      updateModelPanel(findRecord(selectedModelKey) || findRecord('auto'));
      refreshUploadPanel();
    }

    function addMessage(kind, payload){
      const card = document.createElement('div');
      card.className = 'msg ' + kind;
      const messageId = 'msg-' + (++messageSerial);
      const messageText = payload.response || payload.prompt_used || '';
      const messageLabel = payload.model_label || (kind === 'user' ? 'You' : 'Assistant');
      card.id = messageId;
      card.dataset.messageId = messageId;
      card.dataset.messageRole = kind;
      card.dataset.messageLabel = messageLabel;
      card.dataset.messageText = messageText;
      const metaBadges = [];
      if(payload.model_label) metaBadges.push(`<span class="meta-pill">${escapeHtml(payload.model_label)}</span>`);
      if(payload.kind === 'image') metaBadges.push('<span class="meta-pill">image</span>');
      card.innerHTML = `
        <div class="msg-top">
          <div class="who">${kind === 'user' ? 'You' : 'Assistant'}</div>
          <div class="msg-meta">${metaBadges.join('')}</div>
        </div>
      `;
      const actions = document.createElement('div');
      actions.className = 'msg-actions';
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
          actions.innerHTML = `<button class="mini-btn save-image-btn" data-output-path="${escapeHtml(payload.output_path || '')}">Save Image To Path</button>`;
          lastGeneratedImagePath = payload.output_path || lastGeneratedImagePath;
        }
      } else {
        const body = document.createElement('div');
        body.className = 'body';
        body.textContent = payload.response || '';
        card.appendChild(body);
        if(payload.uploaded_image_url){
          const preview = document.createElement('div');
          preview.className = 'image-card';
          preview.innerHTML = `
            <img src="${escapeHtml(payload.uploaded_image_url || '')}" alt="uploaded image">
            <div class="image-caption">${escapeHtml(payload.uploaded_image_name || 'Attached image')}</div>
          `;
          card.appendChild(preview);
        }
      }
      if(payload.response){
        actions.innerHTML += `<button class="mini-btn copy-text-btn" data-copy-text="${escapeHtml(payload.response || '')}">Copy Text</button>`;
      }
      if(messageText && payload.kind !== 'image'){
        actions.innerHTML += `<button class="mini-btn pin-context-btn">Pin Context</button>`;
        actions.innerHTML += `<button class="mini-btn bookmark-msg-btn">Bookmark</button>`;
      }
      if(kind === 'assistant' && payload.response){
        actions.innerHTML += `<button class="mini-btn reuse-msg-btn" data-reuse-text="${escapeHtml(payload.response || '')}">Reuse In Prompt</button>`;
        actions.innerHTML += `<button class="mini-btn compare-a-btn">Pin A</button>`;
        actions.innerHTML += `<button class="mini-btn compare-b-btn">Pin B</button>`;
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
            if(event.name === 'open_cmd'){
              return `- open_cmd -> ${(event.results || [])[0]?.snippet || 'Command Prompt opened'}`;
            }
            const domains = (event.results || []).slice(0, 3).map(item => item.domain || item.url || item.title).filter(Boolean).join(', ');
            return `- ${event.query}${domains ? ' -> ' + domains : ''}`;
          });
          blocks.push('Tool Activity\\n' + searchBits.join('\\n'));
        }
        if(blocks.length){
          const trace = document.createElement('div');
          trace.className = 'trace-box';
          trace.textContent = blocks.join('\\n\\n');
          card.appendChild(trace);
        }
      }
      if(actions.children.length){
        card.appendChild(actions);
      }
      card.dataset.search = [
        payload.model_label || '',
        payload.response || '',
        payload.prompt_used || '',
        payload.route_reason || '',
        payload.uploaded_image_name || ''
      ].join(' ');
      if(kind === 'assistant' && payload.response){
        card.dataset.comparePayload = JSON.stringify(snapshotAssistantReply(payload));
      }
      thread.appendChild(card);
      jumpToLatest();
      transcript.push(buildTranscriptEntry(kind, payload));
      summarizeTranscript();
      applyThreadFilter();
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
        renderQuickPickChips();
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
        updateLiveState(statusResp.status);
        summarizeTranscript();
        updatePromptStats();
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
        threadBookmarks = [];
        compareSlots = {a:null, b:null};
        lastGeneratedImagePath = '';
        clearUploadedImage();
        const welcome = document.createElement('div');
        welcome.className = 'welcome';
        welcome.textContent = 'Session cleared.';
        thread.appendChild(welcome);
        renderThreadBookmarks();
        renderCompareBench();
        summarizeTranscript();
        updatePromptStats();
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
      let prompt = el('prompt').value.trim();
      if(!prompt && currentUploadedImagePath){
        prompt = 'What is in this image?';
      }
      if(!prompt) return;
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
          uploaded_image_path: currentUploadedImagePath,
          style_mode: el('styleMode').value,
          system_hint: composeSystemHint(),
          image_style: el('imageStyle').value,
          image_width: Number(el('imageWidth').value || 512),
          image_height: Number(el('imageHeight').value || 512),
          image_steps: Number(el('imageSteps').value || 2)
        }
      };
      addMessage('user', {
        response: prompt,
        kind: 'text',
        uploaded_image_url: currentUploadedImageUrl,
        uploaded_image_path: currentUploadedImagePath,
        uploaded_image_name: currentUploadedImageName
      });
      el('sendBtn').disabled = true;
      el('sendBtn').textContent = 'Working...';
      try{
        const data = await jpost('/api/chat', payload);
        addMessage('assistant', data);
        selectedModelKey = data.selected_model_key || selectedModelKey;
        if(activeFocusKey){
          el('systemHint').value = '';
          activeFocusKey = '';
          renderFocusChips();
        }
        clearUploadedImage();
        refresh();
      }catch(error){
        addMessage('assistant', {response: 'Error: ' + error.message, kind: 'text'});
      }finally{
        el('sendBtn').disabled = false;
        el('sendBtn').textContent = 'Send';
      }
    }

    async function uploadImage(){
      const file = el('imageUpload').files?.[0];
      if(!file){
        showToast('err', 'Choose an image file first.');
        return;
      }
      const form = new FormData();
      form.append('session_id', sessionId);
      form.append('file', file);
      try{
        const response = await fetch('/api/upload_image', {method: 'POST', body: form});
        const data = await response.json();
        if(!response.ok || data.ok === false){
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        currentUploadedImagePath = data.saved_path || '';
        currentUploadedImageUrl = data.image_url || '';
        currentUploadedImageName = data.filename || file.name;
        el('uploadStatus').textContent = 'Uploaded image is attached to the next prompt.';
        renderUploadPreview();
        updatePromptStats();
        showToast('ok', 'Image uploaded.');
      }catch(error){
        showToast('err', error.message);
      }
    }

    function clearUploadedImage(){
      currentUploadedImagePath = '';
      currentUploadedImageUrl = '';
      currentUploadedImageName = '';
      el('imageUpload').value = '';
      refreshUploadPanel();
      updatePromptStats();
    }

    function buildStarterChips(){
      const box = el('starterChips');
      box.innerHTML = '';
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

    function renderFocusChips(){
      const box = el('focusChips');
      box.innerHTML = '';
      FOCUS_PACKS.forEach((pack) => {
        const chip = document.createElement('button');
        chip.className = 'chip focus-chip' + (activeFocusKey === pack.key ? ' active' : '');
        chip.textContent = pack.label;
        chip.onclick = () => {
          if(activeFocusKey === pack.key){
            activeFocusKey = '';
            el('systemHint').value = '';
          } else {
            activeFocusKey = pack.key;
            el('systemHint').value = pack.hint;
            if(pack.style) el('styleMode').value = pack.style;
          }
          renderFocusChips();
          updatePromptStats();
          updateLiveState();
        };
        box.appendChild(chip);
      });
    }

    el('loadModelBtn').onclick = selectModel;
    el('refreshBtn').onclick = refresh;
    el('clearBtn').onclick = clearChat;
    el('sendBtn').onclick = sendPrompt;
    el('uploadBtn').onclick = uploadImage;
    el('clearUploadBtn').onclick = clearUploadedImage;
    el('saveChatImageBtn').onclick = exportChatImage;
    el('saveLastImageBtn').onclick = saveLastImage;
    el('addContextNoteBtn').onclick = addManualContext;
    el('captureLastReplyBtn').onclick = captureLastAssistantContext;
    el('clearContextBankBtn').onclick = clearContextBank;
    el('toggleSidebarBtn').onclick = () => {
      focusLayout = !focusLayout;
      persistUiState();
      applyLayoutState();
    };
    el('toggleThreadDensityBtn').onclick = () => {
      compactThread = !compactThread;
      persistUiState();
      applyThreadDensity();
    };
    el('refreshStoreBtn').onclick = () => refreshModelStore(true);
    document.querySelectorAll('[data-compose-tab]').forEach((button) => {
      button.onclick = () => setComposeTab(button.dataset.composeTab || 'quick');
    });
    el('modelSearch').addEventListener('input', renderCatalog);
    el('capabilityFilter').addEventListener('change', renderCatalog);
    el('modelSelect').addEventListener('change', () => {
      refreshUploadPanel();
      refreshThreeDViewer(findRecord(el('modelSelect').value));
      updateLiveState();
      updateDiscoveryNote(filteredCatalogRecords());
    });
    el('agentMode').addEventListener('change', () => { refreshUploadPanel(); updateLiveState(); });
    el('actionMode').addEventListener('change', () => { refreshUploadPanel(); updateLiveState(); updatePromptStats(); });
    el('memoryMode').addEventListener('change', updateLiveState);
    el('webSearchMode').addEventListener('change', updateLiveState);
    el('styleMode').addEventListener('change', updatePromptStats);
    el('systemHint').addEventListener('input', updatePromptStats);
    el('sessionObjective').addEventListener('input', syncSessionBrief);
    el('sessionConstraints').addEventListener('input', syncSessionBrief);
    el('sessionDone').addEventListener('input', syncSessionBrief);
    el('threadFilter').addEventListener('input', applyThreadFilter);
    thread.addEventListener('click', (event) => {
      const button = event.target.closest('.save-image-btn');
      if(button){
        saveGeneratedImage(button.dataset.outputPath || '');
        return;
      }
      const copyButton = event.target.closest('.copy-text-btn');
      if(copyButton){
        copyText(copyButton.dataset.copyText || '', 'Message copied.');
        return;
      }
      const reuseButton = event.target.closest('.reuse-msg-btn');
      if(reuseButton){
        el('prompt').value = reuseButton.dataset.reuseText || '';
        el('prompt').focus();
        updatePromptStats();
        return;
      }
      const pinContextButton = event.target.closest('.pin-context-btn');
      if(pinContextButton){
        const card = pinContextButton.closest('.msg');
        if(addContextEntry(card?.dataset.messageText || '', card?.dataset.messageLabel || 'Pinned message')){
          showToast('ok', 'Message added to context bank.');
        }else{
          showToast('err', 'Nothing to add from this message.');
        }
        return;
      }
      const bookmarkButton = event.target.closest('.bookmark-msg-btn');
      if(bookmarkButton){
        addThreadBookmark(bookmarkButton.closest('.msg'));
        return;
      }
      const compareAButton = event.target.closest('.compare-a-btn');
      if(compareAButton){
        const card = compareAButton.closest('.msg');
        compareSlots.a = JSON.parse(card?.dataset.comparePayload || 'null');
        renderCompareBench();
        showToast('ok', 'Reply pinned to slot A.');
        return;
      }
      const compareBButton = event.target.closest('.compare-b-btn');
      if(compareBButton){
        const card = compareBButton.closest('.msg');
        compareSlots.b = JSON.parse(card?.dataset.comparePayload || 'null');
        renderCompareBench();
        showToast('ok', 'Reply pinned to slot B.');
      }
    });
    el('prompt').addEventListener('keydown', (event) => {
      if(event.key === 'Enter' && !event.shiftKey){
        event.preventDefault();
        sendPrompt();
      }
    });
    el('prompt').addEventListener('input', updatePromptStats);
    el('applyBriefBtn').onclick = applyBriefToHint;
    el('clearBriefBtn').onclick = () => {
      clearSessionBrief();
      showToast('ok', 'Session brief cleared.');
    };
    el('saveDraftBtn').onclick = saveCurrentDraft;
    el('insertLatestDraftBtn').onclick = () => {
      if(!savedDrafts.length){
        showToast('err', 'No saved drafts yet.');
        return;
      }
      el('prompt').value = savedDrafts[0].prompt || '';
      updatePromptStats();
      el('prompt').focus();
    };
    el('copyLastBtn').onclick = () => {
      const last = [...transcript].reverse().find((item) => item.role === 'assistant' && item.response);
      if(!last){
        showToast('err', 'No assistant reply yet.');
        return;
      }
      copyText(last.response || '', 'Last reply copied.');
    };
    el('copyThreadBtn').onclick = () => {
      if(!transcript.length){
        showToast('err', 'No thread to copy yet.');
        return;
      }
      copyText(transcriptAsText(), 'Thread copied.');
    };
    el('downloadThreadBtn').onclick = () => {
      if(!transcript.length){
        showToast('err', 'No thread to download yet.');
        return;
      }
      downloadTranscriptJson();
      showToast('ok', 'Thread JSON downloaded.');
    };
    el('jumpBottomBtn').onclick = jumpToLatest;
    el('swapCompareBtn').onclick = () => {
      const temp = compareSlots.a;
      compareSlots.a = compareSlots.b;
      compareSlots.b = temp;
      renderCompareBench();
    };
    el('clearCompareBtn').onclick = () => {
      compareSlots = {a:null, b:null};
      renderCompareBench();
    };

    buildStarterChips();
    buildResponseDeck();
    applySessionBriefInputs();
    applyLayoutState();
    applyThreadDensity();
    setComposeTab(composeTab);
    renderSessionBrief();
    renderSavedDrafts();
    renderContextBank();
    renderThreadBookmarks();
    renderCompareBench();
    renderFocusChips();
    summarizeTranscript();
    updatePromptStats();
    updateLiveState();
    refreshModelStore(true);
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

    @app.get("/api/model_store")
    def api_model_store():
        force_refresh = str(request.args.get("refresh") or "").strip().lower() in {"1", "true", "yes"}
        payload = manager.model_store_catalog(force_refresh=force_refresh)
        return jsonify({"ok": True, **payload})

    @app.get("/api/model_store/jobs")
    def api_model_store_jobs():
        return jsonify({"ok": True, **manager.model_store_jobs()})

    @app.post("/api/model_store/install")
    def api_model_store_install():
        payload = request.get_json(force=True, silent=True) or {}
        file_name = str(payload.get("file_name") or "").strip()
        if not file_name:
            return jsonify({"ok": False, "error": "file_name is required"}), 400
        try:
            job = manager.install_model_store_artifact(file_name)
            return jsonify({"ok": True, "job": job})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "status": manager.status()})

    @app.get("/api/memory")
    def api_memory():
        session_id = str(request.args.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id is required"}), 400
        return jsonify({"ok": True, "memory": manager.session_memory_snapshot(session_id)})

    @app.get("/api/three_d_model_view")
    def api_three_d_model_view():
        try:
            payload = dict(manager.three_d_model_view())
            payload["download_zip_url"] = "/download/three_d_model_zip"
            payload["download_summary_url"] = "/download/three_d_model_summary" if payload.get("summary_path") else ""
            return jsonify({"ok": True, "model": payload})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @app.post("/api/upload_image")
    def api_upload_image():
        session_id = str(request.form.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id is required"}), 400
        upload = request.files.get("file")
        if upload is None or not getattr(upload, "filename", ""):
            return jsonify({"ok": False, "error": "file is required"}), 400
        try:
            result = manager.store_uploaded_image(
                session_id=session_id,
                filename=str(upload.filename),
                raw_bytes=upload.read(),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

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

    @app.get("/uploads/<path:session_key>/<path:filename>")
    def uploaded_file(session_key: str, filename: str):
        target_dir = manager.uploads_dir / session_key
        return send_from_directory(target_dir, filename)

    @app.get("/download/three_d_model_zip")
    def download_three_d_model_zip():
        try:
            payload = manager.three_d_model_view()
            zip_path = Path(str(payload.get("zip_path") or "")).resolve()
            return send_from_directory(str(zip_path.parent), zip_path.name, as_attachment=True)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @app.get("/download/three_d_model_summary")
    def download_three_d_model_summary():
        try:
            payload = manager.three_d_model_view()
            summary_path = Path(str(payload.get("summary_path") or "")).resolve()
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            return send_from_directory(str(summary_path.parent), summary_path.name, as_attachment=True)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

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
        models_dir=models_dir,
        common_summary_path=common_summary if common_summary is not None else DEFAULT_COMMON_SUMMARY,
    )
    app = build_app(manager)
    print(f"Supermix Studio: http://{args.host}:{args.port}")
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
