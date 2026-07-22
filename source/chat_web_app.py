import argparse
import json
import threading
import time
import uuid
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
import torch

import chat_app
from device_utils import configure_torch_runtime, resolve_device
from route_policy_shadow_registry import RouteShadowAssignmentRegistry


HTML = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Champion Chat Web</title>
<style>
body{margin:0;background:#0b1220;color:#e5edf7;font-family:Segoe UI,Arial,sans-serif}
.wrap{max-width:1100px;margin:20px auto;padding:14px;display:grid;grid-template-columns:340px 1fr;gap:14px}
.card{background:#121c30;border:1px solid #24324e;border-radius:14px;box-shadow:0 10px 24px rgba(0,0,0,.25)}
.side{padding:14px}.chat{display:grid;grid-template-rows:auto 1fr auto;min-height:78vh}
.row{margin-bottom:10px}.row label{display:block;font-size:.75rem;color:#9fb1d1;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em}
input,select,textarea{width:100%;background:#0b1220;color:#e5edf7;border:1px solid #2a3a58;border-radius:10px;padding:10px}
button{background:#1d4ed8;color:white;border:0;border-radius:10px;padding:10px 12px;font-weight:600;cursor:pointer}
button.alt{background:#263449}.btns{display:flex;gap:8px;flex-wrap:wrap}
.status{white-space:pre-wrap;background:#0b1220;border:1px solid #2a3a58;border-radius:10px;padding:10px;min-height:100px;color:#b7c6df;font-size:.85rem}
.head{padding:12px 14px;border-bottom:1px solid #24324e;display:flex;justify-content:space-between;gap:8px;align-items:center}
.head small{color:#9fb1d1}
.msgs{padding:12px;overflow:auto;display:flex;flex-direction:column;gap:10px}
.msg{border:1px solid #24324e;border-radius:12px;padding:10px;background:#0b1220;max-width:85%;white-space:pre-wrap;line-height:1.35}
.msg.user{align-self:flex-end;background:#10203d;border-color:#244d90}.msg.bot{align-self:flex-start}
.msg .who{font-size:.72rem;color:#9fb1d1;margin-bottom:4px;text-transform:uppercase}.tim{margin-top:6px;color:#9fb1d1;font-size:.75rem}
.comp{padding:12px;border-top:1px solid #24324e;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end}
textarea{min-height:68px;max-height:180px;resize:vertical}
@media (max-width: 900px){.wrap{grid-template-columns:1fr}.chat{min-height:70vh}}
</style></head><body>
<div class='wrap'>
  <div class='card side'>
    <h3 style='margin:0 0 6px'>Champion Chat Web</h3>
    <div style='color:#9fb1d1;font-size:.9rem;margin-bottom:12px'>Load model/meta files, then chat in the browser.</div>
    <div class='row'><label>Weights (.pth)</label><input id='weights'></div>
    <div class='row'><label>Metadata (.json)</label><input id='meta'></div>
    <div class='row'><label>Style</label><select id='style'><option>auto</option><option>balanced</option><option>creative</option><option>concise</option><option>analyst</option></select></div>
    <div class='row'><label>Response Temp</label><input id='rt' type='number' min='0' max='1' step='0.01' value='0.08'></div>
    <div class='row'><label>Show Top Candidates</label><input id='showTop' type='number' min='0' max='10' step='1' value='0'></div>
    <div class='btns'><button id='loadBtn'>Load Model</button><button class='alt' id='statusBtn'>Refresh</button><button class='alt' id='clearBtn'>Clear Session</button></div>
    <div class='status' id='statusBox'>Loading status...</div>
  </div>
  <div class='card chat'>
    <div class='head'><div><div style='font-weight:700'>Web Chat</div><small id='metaLine'>No model loaded</small></div><small id='session'></small></div>
    <div class='msgs' id='msgs'></div>
    <div class='comp'><textarea id='prompt' placeholder='Type message, Enter to send (Shift+Enter newline)'></textarea><button id='sendBtn'>Send</button></div>
  </div>
</div>
<script>
const el=(id)=>document.getElementById(id), msgs=el('msgs');
let sid=localStorage.getItem('champion-web-sid'); if(!sid){{sid=(crypto.randomUUID?crypto.randomUUID():String(Date.now())); localStorage.setItem('champion-web-sid',sid);}} el('session').textContent='session '+sid.slice(0,8);
function add(kind,text,timing,top){{const d=document.createElement('div'); d.className='msg '+kind; d.innerHTML=`<div class='who'>${{kind==='user'?'You':'Bot'}}</div>`; const b=document.createElement('div'); b.textContent=text; d.appendChild(b); if(timing){{const t=document.createElement('div'); t.className='tim'; t.textContent=`Timing: infer=${{timing.infer}} ms, rank=${{timing.rank_pick}} ms, total=${{timing.total}} ms`; d.appendChild(t);}} if(top&&top.length){{const x=document.createElement('div'); x.className='tim'; x.innerHTML='Top candidates:<br>'+top.map((c,i)=>`${{i+1}}. (${{c.score.toFixed(3)}}) ${{c.text.slice(0,160)}}`).join('<br>'); d.appendChild(x);}} msgs.appendChild(d); msgs.scrollTop=msgs.scrollHeight; }}
async function jget(path){{const r=await fetch(path); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${{r.status}}`); return d;}}
async function jpost(path,p){{const r=await fetch(path,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(p||{{}})}}); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${{r.status}}`); return d;}}
async function refresh(){{try{{const d=await jget('/api/status'); el('statusBox').textContent=JSON.stringify(d.status,null,2); el('metaLine').textContent=d.status.loaded?`${{d.status.model_size}} | ${{d.status.feature_mode}} | labels=${{d.status.available_labels}}`:'No model loaded'; if(!el('weights').value&&d.status.weights) el('weights').value=d.status.weights; if(!el('meta').value&&d.status.meta) el('meta').value=d.status.meta; }}catch(e){{el('statusBox').textContent='Status error: '+e.message;}}}}
async function loadModel(){{el('statusBox').textContent='Loading model...'; try{{const d=await jpost('/api/load',{{weights:el('weights').value.trim(),meta:el('meta').value.trim()}}); el('statusBox').textContent='Loaded.\n'+JSON.stringify(d,null,2); refresh();}}catch(e){{el('statusBox').textContent='Load error: '+e.message;}}}}
async function send(){{const text=el('prompt').value.trim(); if(!text) return; add('user',text); el('prompt').value=''; try{{const d=await jpost('/api/chat',{{session_id:sid,message:text,style_mode:el('style').value,response_temperature:Number(el('rt').value),show_top_responses:Number(el('showTop').value)}}); add('bot',d.response,d.timing_ms,d.top_candidates);}}catch(e){{add('bot','Error: '+e.message);}}}}
async function clearSess(){{try{{await jpost('/api/clear',{{session_id:sid}}); msgs.innerHTML=''; add('bot','Session cleared.');}}catch(e){{add('bot','Clear error: '+e.message);}}}}
el('loadBtn').onclick=loadModel; el('statusBtn').onclick=refresh; el('clearBtn').onclick=clearSess; el('sendBtn').onclick=send; el('prompt').addEventListener('keydown',e=>{{if(e.key==='Enter'&&!e.shiftKey){{e.preventDefault();send();}}}}); refresh();
</script></body></html>"""


HTML = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Champion Chat Web</title>
<style>
:root{--bg:#eef1f3;--panel:#fbfcfd;--ink:#172026;--muted:#63717d;--line:#cfd8df;--accent:#087f5b;--accent2:#1d4b6d;--user:#dcefff;--bot:#fff;--shadow:0 18px 50px rgba(23,32,38,.12)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:linear-gradient(145deg,#eef1f3 0%,#dfe7ea 52%,#f7f8f2 100%);color:var(--ink);font-family:Aptos,Segoe UI,sans-serif}
.wrap{width:min(1240px,100%);margin:0 auto;padding:22px;display:grid;grid-template-columns:330px minmax(0,1fr);gap:16px}.panel,.chat{background:rgba(251,252,253,.94);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:8px}
.panel{padding:16px;display:flex;flex-direction:column;gap:14px}.chat{height:calc(100vh - 44px);min-height:680px;display:grid;grid-template-rows:auto 1fr auto;overflow:hidden}
.brand{display:flex;align-items:center;gap:10px}.mark{width:34px;height:34px;border-radius:7px;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:inset 0 0 0 1px rgba(255,255,255,.28)}
h1{font-size:1.05rem;margin:0}.sub{color:var(--muted);font-size:.84rem;line-height:1.4}.row{display:grid;gap:6px}.row label{font-size:.72rem;font-weight:800;color:#40505c;text-transform:uppercase;letter-spacing:.08em}
input,select,textarea{width:100%;border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:7px;padding:10px 11px;font:inherit}input:focus,select:focus,textarea:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(8,127,91,.16)}
.split,.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{border:0;border-radius:7px;padding:10px 12px;font:inherit;font-weight:800;cursor:pointer;background:var(--accent);color:#fff}button:hover{filter:brightness(.94)}button:disabled{opacity:.55;cursor:not-allowed}
button.alt{background:#e7edf0;color:#1e2a31;border:1px solid var(--line)}.status{white-space:pre-wrap;background:#f6f8f9;border:1px solid var(--line);border-radius:7px;padding:11px;min-height:112px;color:#40505c;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}
.shadow-status{display:grid;gap:8px;padding:10px;border:1px solid var(--line);border-radius:7px;background:#f6f8f9;color:#40505c;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.shadow-head{display:flex;align-items:center;justify-content:space-between;gap:8px;font-family:Aptos,Segoe UI,sans-serif;font-weight:800;text-transform:uppercase;letter-spacing:.05em}.shadow-head button{padding:7px 9px;font-size:.72rem}
.head{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;align-items:center;background:#fff}.head-title{font-weight:900}.head small{color:var(--muted)}
.pillrow{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.pill{border:1px solid var(--line);background:#f6f8f9;border-radius:999px;padding:5px 9px;color:#40505c;font-size:.76rem}
.msgs{padding:16px;overflow:auto;display:flex;flex-direction:column;gap:12px;background:linear-gradient(180deg,#f8fafb,#eef3f5)}.msg{position:relative;border:1px solid var(--line);border-radius:8px;padding:12px 13px;max-width:min(78%,820px);white-space:pre-wrap;line-height:1.45;background:var(--bot);box-shadow:0 6px 18px rgba(23,32,38,.06)}
.msg.user{align-self:flex-end;background:var(--user);border-color:#a8d2f2}.msg.bot{align-self:flex-start}.msg.pending{color:var(--muted);font-style:italic}.who{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:6px;color:#4b5b65;font-size:.72rem;font-weight:900;text-transform:uppercase;letter-spacing:.07em}
.copy{padding:4px 7px;border-radius:6px;background:#eef3f5;color:#31414c;border:1px solid var(--line);font-size:.72rem}.tim{margin-top:8px;color:#63717d;font-size:.76rem}details{margin-top:8px;border-top:1px solid var(--line);padding-top:8px;color:#63717d;font-size:.76rem}summary{cursor:pointer;font-weight:800;color:#40505c}
.comp{padding:13px 16px;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:end;background:#fff}textarea{min-height:54px;max-height:190px;resize:none;line-height:1.42}.hint{grid-column:1/-1;color:var(--muted);font-size:.76rem}
.quick{display:flex;gap:6px;flex-wrap:wrap;margin-top:2px}.quick button{background:#f6f8f9;color:#31414c;border:1px solid var(--line);font-weight:700;padding:7px 9px;font-size:.78rem}
@media (max-width:900px){.wrap{grid-template-columns:1fr;padding:10px}.chat{height:72vh;min-height:560px}.msg{max-width:94%}.split,.btns{grid-template-columns:1fr}}
</style></head><body>
<div class='wrap'>
  <aside class='panel'>
    <div class='brand'><div class='mark'></div><div><h1>Champion Chat</h1><div class='sub'>Local model console</div></div></div>
    <div class='row'><label for='weights'>Weights (.pth)</label><input id='weights' value='' spellcheck='false'></div>
    <div class='row'><label for='meta'>Metadata (.json)</label><input id='meta' value='' spellcheck='false'></div>
    <div class='split'><div class='row'><label for='style'>Style</label><select id='style'><option>auto</option><option>balanced</option><option>creative</option><option>concise</option><option>analyst</option></select></div><div class='row'><label for='showTop'>Candidates</label><input id='showTop' type='number' min='0' max='10' step='1' value='0'></div></div>
    <div class='row'><label for='rt'>Response temperature</label><input id='rt' type='number' min='0' max='1' step='0.01' value='0.08'></div>
    <div class='split'><div class='row'><label for='reasoningCycles'>Reasoning cycles</label><input id='reasoningCycles' type='text' placeholder='default or auto'></div><div class='row'><label for='exitTol'>Exit tolerance</label><input id='exitTol' type='number' min='0' step='0.0001' value='0.001'></div></div>
    <div class='split'><div class='row'><label for='exitEntropy'>Exit entropy</label><input id='exitEntropy' type='number' min='0' step='0.01' value='0.2'></div><div class='row'><label for='stabilityTol'>Stability tolerance</label><input id='stabilityTol' type='number' min='0' step='0.001' value='0.005'></div></div>
    <div class='split'><div class='row'><label for='stabilityPatience'>Stability patience</label><input id='stabilityPatience' type='number' min='0' max='64' step='1' value='2'></div><div class='row'><label for='stabilityMargin'>Stability margin</label><input id='stabilityMargin' type='number' min='0' step='0.0001' value='0.0001'></div></div>
    <div class='row'><label for='stabilityRankDepth'>Decision rank depth</label><input id='stabilityRankDepth' type='number' min='0' max='10' step='1' value='3'><div class='sub'>Verify the ordered top ranks; 0 intentionally disables rank verification.</div></div>
    <div class='row'><label for='adaptiveCompute'>Adaptive compute</label><select id='adaptiveCompute'><option value='off'>off</option><option value='on'>on</option></select></div>
    <div class='row'><label for='autoCompute'>Progressive auto compute</label><select id='autoCompute'><option value='off'>off</option><option value='on'>on</option></select></div>
    <div class='btns'><button id='loadBtn'>Load</button><button class='alt' id='statusBtn'>Refresh</button><button class='alt' id='clearBtn'>Clear</button><button class='alt' id='newSessionBtn'>New ID</button></div>
    <div class='status' id='statusBox'>Loading status...</div>
    <div class='shadow-status'><div class='shadow-head'><span>Shadow registry - read only</span><button class='alt' id='shadowBtn' type='button'>Refresh</button></div><div id='shadowBox'>Not loaded. Execution, activation, and promotion are unavailable.</div></div>
  </aside>
  <main class='chat'>
    <header class='head'><div><div class='head-title'>Web Chat</div><small id='metaLine'>No model loaded</small></div><div class='pillrow'><span class='pill' id='session'></span><span class='pill' id='runtimePill'>idle</span></div></header>
    <section class='msgs' id='msgs' aria-live='polite'></section>
    <section class='comp'><textarea id='prompt' placeholder='Type a message. Enter sends, Shift+Enter adds a line.'></textarea><button id='sendBtn'>Send</button><button class='alt' id='sweepBtn' type='button'>Sweep</button><div class='hint'>Drafts and the local transcript are kept in this browser session.</div><div class='quick'><button type='button' data-fill='Summarize the latest benchmark result and explain the weakest benchmark.'>Benchmark readout</button><button type='button' data-fill='Give a concise debugging checklist for this model response.'>Debug checklist</button><button type='button' data-fill='Answer as a concise analyst and include uncertainty when needed.'>Analyst mode</button></div></section>
  </main>
</div>
<script>
const el=(id)=>document.getElementById(id);
const els={msgs:el('msgs'),prompt:el('prompt'),sendBtn:el('sendBtn'),sweepBtn:el('sweepBtn'),loadBtn:el('loadBtn'),statusBtn:el('statusBtn'),clearBtn:el('clearBtn'),newSessionBtn:el('newSessionBtn'),shadowBtn:el('shadowBtn'),shadowBox:el('shadowBox'),statusBox:el('statusBox'),metaLine:el('metaLine'),session:el('session'),runtimePill:el('runtimePill'),weights:el('weights'),meta:el('meta'),style:el('style'),rt:el('rt'),showTop:el('showTop'),reasoningCycles:el('reasoningCycles'),adaptiveCompute:el('adaptiveCompute'),autoCompute:el('autoCompute'),exitTol:el('exitTol'),exitEntropy:el('exitEntropy'),stabilityPatience:el('stabilityPatience'),stabilityTol:el('stabilityTol'),stabilityMargin:el('stabilityMargin'),stabilityRankDepth:el('stabilityRankDepth')};
let sid=localStorage.getItem('champion-web-sid');if(!sid){sid=crypto.randomUUID?crypto.randomUUID():String(Date.now());localStorage.setItem('champion-web-sid',sid);}
const draftKey='champion-web-draft-v2';const transcriptKey=()=>('champion-web-transcript-v2-'+sid);let transcript=[];let sending=false;
function setSessionLabel(){els.session.textContent='session '+sid.slice(0,8);}
function loadTranscript(){try{transcript=JSON.parse(localStorage.getItem(transcriptKey())||'[]');}catch(_){transcript=[];}}
function saveTranscript(){localStorage.setItem(transcriptKey(),JSON.stringify(transcript.slice(-80)));}
function autoSizePrompt(){els.prompt.style.height='auto';els.prompt.style.height=Math.min(els.prompt.scrollHeight,190)+'px';}
function setBusy(active,label){sending=active;els.sendBtn.disabled=active;els.sweepBtn.disabled=active;els.loadBtn.disabled=active;els.runtimePill.textContent=label||(active?'working':'idle');}
function fmtNum(value,digits=3){const n=Number(value);return Number.isFinite(n)?n.toFixed(digits):null;}
function timingText(t){if(!t)return'';let s=`${t.total??'?'} ms total - ${t.infer??'?'} ms infer - ${t.rank_pick??'?'} ms rank`;if(t.cycles_used!==undefined&&t.cycles_used!==null){s+=` - cycles ${t.cycles_used}`;}return s;}
function reasoningCyclesValue(){const raw=els.reasoningCycles.value.trim();if(!raw)return null;const low=raw.toLowerCase();if(['auto','adaptive','smart'].includes(low))return 'auto';const n=Number(raw);return Number.isFinite(n)?n:raw;}
function addOptionalFiniteNumber(payload,key,input){const raw=input.value.trim();if(raw==='')return;const value=Number(raw);if(Number.isFinite(value))payload[key]=value;}
function computeText(compute){if(!compute||!compute.applied)return'';const parts=[];if(compute.prediction_verifier_active===true){parts.push('verifier active');}else if(compute.adaptive_compute){parts.push('verifier inactive');}if(compute.reasoning_budget_mode==='auto'){parts.push('mode auto');}if(compute.requested_reasoning_cycles!==undefined&&compute.requested_reasoning_cycles!==null){parts.push(`requested ${compute.requested_reasoning_cycles}`);}if(compute.cycles_used!==undefined&&compute.cycles_used!==null){parts.push(`used ${compute.cycles_used}`);}if(compute.exit_reason){parts.push(`exit ${compute.exit_reason}`);}const streak=fmtNum(compute.prediction_streak);if(streak){parts.push(`stable ${streak}`);}const drift=fmtNum(compute.prediction_confidence_delta);if(drift){parts.push(`drift ${drift}`);}const observedMargin=fmtNum(compute.prediction_margin,6);if(observedMargin){parts.push(`top-1 margin ${observedMargin}`);}const decisionMargin=fmtNum(compute.prediction_decision_margin,6);if(decisionMargin){parts.push(`decision margin ${decisionMargin}`);}const marginFloor=fmtNum(compute.prediction_stability_margin,6);if(marginFloor){parts.push(`margin floor ${marginFloor}`);}if(compute.prediction_rank_depth!==undefined&&compute.prediction_rank_depth!==null){parts.push(`verified depth ${compute.prediction_rank_depth}`);}if(compute.prediction_stability_rank_depth!==undefined&&compute.prediction_stability_rank_depth!==null){parts.push(`requested depth ${compute.prediction_stability_rank_depth}`);}if(compute.prediction_class_count!==undefined&&compute.prediction_class_count!==null){parts.push(`verifier ${compute.prediction_class_count} classes`);}if(compute.prediction_class_selection_valid===false){parts.push('verifier scope invalid');}const ponder=fmtNum(compute.ponder_cost);if(ponder){parts.push(`ponder ${ponder}`);}const consistency=fmtNum(compute.consistency_loss);if(consistency){parts.push(`consistency ${consistency}`);}const entropy=fmtNum(compute.gating_entropy);if(entropy){parts.push(`gate entropy ${entropy}`);}const exitEntropy=fmtNum(compute.exit_entropy_threshold);if(exitEntropy){parts.push(`exit entropy ${exitEntropy}`);}if(compute.auto_reasoning_policy&&Array.isArray(compute.auto_reasoning_policy.reasons)){parts.push(`policy ${compute.auto_reasoning_policy.reasons.slice(0,3).join(',')}`);}if(compute.auto_compute_plan){const plan=compute.auto_compute_plan;parts.push(`budget ${plan.selected_reasoning_cycles} (${plan.reason})`);parts.push(`${plan.forward_evaluations}/${plan.legacy_forward_evaluations} forwards`);if(plan.reused_probe_output)parts.push('probe reused');const rows=Array.isArray(plan.rows)?plan.rows:[];const selectedIndex=Number(plan.selected_index);const selectedRow=Number.isInteger(selectedIndex)&&selectedIndex>=0&&selectedIndex<rows.length?rows[selectedIndex]:(rows.length?rows[rows.length-1]:null);const shadow=selectedRow?selectedRow.mutual_stability_shadow:null;const js=shadow?fmtNum(shadow.js_divergence,6):null;if(js)parts.push(`shadow JSD ${js}`);}return parts.length?`Compute: ${parts.join(' - ')}`:'';}
function add(kind,text,timing,top,persist=true,compute=null){const card=document.createElement('article');card.className='msg '+kind;const who=document.createElement('div');who.className='who';const label=document.createElement('span');label.textContent=kind==='user'?'You':'Champion';who.appendChild(label);if(kind==='bot'&&text){const copy=document.createElement('button');copy.className='copy';copy.type='button';copy.textContent='Copy';copy.onclick=async()=>{try{await navigator.clipboard.writeText(text);copy.textContent='Copied';setTimeout(()=>{copy.textContent='Copy';},1200);}catch(_){copy.textContent='Failed';}};who.appendChild(copy);}const body=document.createElement('div');body.textContent=text;card.appendChild(who);card.appendChild(body);const tt=timingText(timing);if(tt){const node=document.createElement('div');node.className='tim';node.textContent=tt;card.appendChild(node);}const ct=computeText(compute);if(ct){const node=document.createElement('div');node.className='tim';node.textContent=ct;card.appendChild(node);}if(Array.isArray(top)&&top.length){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent=`Top candidates (${top.length})`;details.appendChild(summary);top.forEach((candidate,index)=>{const row=document.createElement('div');const score=Number(candidate.score);const scoreText=Number.isFinite(score)?score.toFixed(3):'n/a';row.textContent=`${index+1}. (${scoreText}) ${String(candidate.text||'').slice(0,220)}`;details.appendChild(row);});card.appendChild(details);}els.msgs.appendChild(card);els.msgs.scrollTo({top:els.msgs.scrollHeight,behavior:'smooth'});if(persist){transcript.push({kind,text,timing,top,compute,ts:Date.now()});saveTranscript();}return card;}
async function jget(path){const r=await fetch(path);const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||`HTTP ${r.status}`);return d;}
async function jpost(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload||{})});const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||`HTTP ${r.status}`);return d;}
function renderStatus(status){const lines=[status.loaded?'Model loaded':'No model loaded','Device: '+(status.device||'unknown'),'Size: '+(status.model_size||'unknown'),'Features: '+(status.feature_mode||'unknown'),'Labels: '+(status.available_labels??'unknown'),'Sessions: '+(status.sessions??0),'Runtime compute: '+(status.runtime_compute_supported?'supported':'not supported'),'Default cycles: '+(status.reasoning_cycles??'default'),'Adaptive: '+(status.adaptive_compute?'on':'off'),'Auto budget: '+(status.auto_compute?'on':'off'),'Exit entropy: '+(status.adaptive_exit_entropy??'default'),'Stability: '+(status.prediction_stability_patience??'off')+' cycles / '+(status.prediction_stability_tol??'default')+' drift / '+(status.prediction_stability_margin??'default')+' margin / '+(status.prediction_stability_rank_depth??'default')+' ranks'];els.statusBox.textContent=lines.join('\\n');els.metaLine.textContent=status.loaded?`${status.model_size} - ${status.feature_mode} - ${status.available_labels} labels`:'Choose model files and load them';els.runtimePill.textContent=status.loaded?'ready':'idle';if(!els.weights.value&&status.weights)els.weights.value=status.weights;if(!els.meta.value&&status.meta)els.meta.value=status.meta;if(els.reasoningCycles&&!els.reasoningCycles.value&&status.reasoning_cycles){els.reasoningCycles.value=status.reasoning_cycles;}if(els.adaptiveCompute){els.adaptiveCompute.value=status.adaptive_compute?'on':'off';}if(els.autoCompute){els.autoCompute.value=status.auto_compute?'on':'off';}if(els.exitTol&&status.adaptive_exit_tol!==undefined){els.exitTol.value=status.adaptive_exit_tol;}if(els.exitEntropy&&status.adaptive_exit_entropy!==undefined){els.exitEntropy.value=status.adaptive_exit_entropy;}if(els.stabilityPatience&&status.prediction_stability_patience!==undefined){els.stabilityPatience.value=status.prediction_stability_patience;}if(els.stabilityTol&&status.prediction_stability_tol!==undefined){els.stabilityTol.value=status.prediction_stability_tol;}if(els.stabilityMargin&&status.prediction_stability_margin!==undefined){els.stabilityMargin.value=status.prediction_stability_margin;}if(els.stabilityRankDepth&&status.prediction_stability_rank_depth!==undefined){els.stabilityRankDepth.value=status.prediction_stability_rank_depth;}}
async function refresh(){try{const data=await jget('/api/status');renderStatus(data.status);}catch(err){els.statusBox.textContent='Status error: '+err.message;els.runtimePill.textContent='status error';}}
function renderShadow(snapshot){const campaigns=Array.isArray(snapshot&&snapshot.campaigns)?snapshot.campaigns:[];const committed=campaigns.reduce((n,row)=>n+(Number(row.commitment_count)||0),0);const matched=campaigns.reduce((n,row)=>n+(Number(row.matched_assignment_count)||0),0);const processed=campaigns.reduce((n,row)=>n+(Number(row.processed_reveal_count)||0),0);const mismatched=campaigns.reduce((n,row)=>n+(Number(row.mismatched_assignment_count)||0),0);const chain=snapshot&&snapshot.event_chain;if(!snapshot||snapshot.available!==true){els.shadowBox.textContent=`Not initialized at ${(snapshot&&snapshot.registry_location)||'the canonical memory path'}.\nRead only - execution, activation, and promotion unavailable.`;return;}els.shadowBox.textContent=`${snapshot.ok?'Verified':'Verification failed'} - ${campaigns.length} campaigns - ${matched}/${committed} assignments matched - ${processed} reveals processed - ${mismatched} mismatches\nChain ${chain&&chain.ok?'verified':'failed'} (${Number(chain&&chain.verified_events)||0} events). Local chain only; browser access is read-only.`;}
async function refreshShadow(){els.shadowBtn.disabled=true;els.shadowBox.textContent='Reading local registry...';try{const data=await jget('/api/route_shadow_registry/status');renderShadow(data.route_shadow_registry||{});}catch(err){els.shadowBox.textContent='Registry status error: '+err.message;}finally{els.shadowBtn.disabled=false;}}
async function loadModel(){setBusy(true,'loading');els.statusBox.textContent='Loading model...';try{const data=await jpost('/api/load',{weights:els.weights.value.trim(),meta:els.meta.value.trim()});renderStatus(data);}catch(err){els.statusBox.textContent='Load error: '+err.message;els.runtimePill.textContent='load failed';}finally{setBusy(false,els.runtimePill.textContent==='load failed'?'load failed':'ready');}}
async function send(){const text=els.prompt.value.trim();if(!text||sending)return;add('user',text);els.prompt.value='';localStorage.removeItem(draftKey);autoSizePrompt();setBusy(true,'generating');const pending=add('bot','Generating response...',null,null,false);pending.classList.add('pending');try{const cycles=reasoningCyclesValue();const payload={session_id:sid,message:text,style_mode:els.style.value,response_temperature:Number(els.rt.value),show_top_responses:Number(els.showTop.value),reasoning_cycles:cycles,adaptive_compute:els.adaptiveCompute.value==='on',auto_compute:els.autoCompute.value==='on',adaptive_exit_tol:Number(els.exitTol.value),adaptive_exit_entropy:Number(els.exitEntropy.value),prediction_stability_patience:Number(els.stabilityPatience.value),prediction_stability_tol:Number(els.stabilityTol.value)};addOptionalFiniteNumber(payload,'prediction_stability_margin',els.stabilityMargin);addOptionalFiniteNumber(payload,'prediction_stability_rank_depth',els.stabilityRankDepth);const data=await jpost('/api/chat',payload);pending.remove();add('bot',data.response,data.timing_ms,data.top_candidates,true,data.compute);els.runtimePill.textContent=data.auto_compute_plan?`auto ${data.auto_compute_plan.selected_reasoning_cycles}`:(data.compute&&data.compute.applied?'compute active':'ready');}catch(err){pending.remove();add('bot','Error: '+err.message);els.runtimePill.textContent='chat error';}finally{setBusy(false,els.runtimePill.textContent);}}
async function sweepCompute(){const text=els.prompt.value.trim();if(!text||sending)return;setBusy(true,'sweeping');const pending=add('bot','Running compute sweep...',null,null,false);pending.classList.add('pending');try{const payload={session_id:sid,message:text,cycles:[1,3,8],adaptive_compute:els.adaptiveCompute.value==='on',adaptive_exit_tol:Number(els.exitTol.value),adaptive_exit_entropy:Number(els.exitEntropy.value),prediction_stability_patience:Number(els.stabilityPatience.value),prediction_stability_tol:Number(els.stabilityTol.value)};addOptionalFiniteNumber(payload,'prediction_stability_margin',els.stabilityMargin);addOptionalFiniteNumber(payload,'prediction_stability_rank_depth',els.stabilityRankDepth);const data=await jpost('/api/compute_sweep',payload);pending.remove();const lines=['Compute sweep for draft prompt:'];data.rows.forEach((row)=>{const entropy=fmtNum(row.entropy);const conf=fmtNum(row.confidence);const reason=row.compute&&row.compute.exit_reason?` - exit ${row.compute.exit_reason}`:'';lines.push(`cycles ${row.requested_cycles}: ${row.latency_ms} ms - used ${row.cycles_used} - label ${row.predicted_label} - conf ${conf??'n/a'} - entropy ${entropy??'n/a'}${reason}`);});add('bot',lines.join('\\n'),null,null,true,data.rows[data.rows.length-1]?.compute||null);els.runtimePill.textContent='sweep done';}catch(err){pending.remove();add('bot','Sweep error: '+err.message);els.runtimePill.textContent='sweep error';}finally{setBusy(false,els.runtimePill.textContent);}}
async function clearSess(){try{await jpost('/api/clear',{session_id:sid});}catch(_){}transcript=[];localStorage.removeItem(transcriptKey());els.msgs.innerHTML='';add('bot','Session cleared.',null,null,false);}
function newSession(){sid=crypto.randomUUID?crypto.randomUUID():String(Date.now());localStorage.setItem('champion-web-sid',sid);transcript=[];setSessionLabel();els.msgs.innerHTML='';add('bot','New session started.',null,null,false);}
els.loadBtn.onclick=loadModel;els.statusBtn.onclick=refresh;els.clearBtn.onclick=clearSess;els.newSessionBtn.onclick=newSession;els.shadowBtn.onclick=refreshShadow;els.sendBtn.onclick=send;els.sweepBtn.onclick=sweepCompute;els.prompt.value=localStorage.getItem(draftKey)||'';els.prompt.addEventListener('input',()=>{localStorage.setItem(draftKey,els.prompt.value);autoSizePrompt();});els.prompt.addEventListener('keydown',(event)=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}});
document.querySelectorAll('[data-fill]').forEach((button)=>{button.addEventListener('click',()=>{els.prompt.value=button.dataset.fill||'';localStorage.setItem(draftKey,els.prompt.value);autoSizePrompt();els.prompt.focus();});});
setSessionLabel();loadTranscript();if(transcript.length){transcript.forEach((item)=>add(item.kind,item.text,item.timing,item.top,false,item.compute));}else{add('bot','Session ready. Load a model to begin.',null,null,false);}autoSizePrompt();refresh();refreshShadow();
</script></body></html>"""


_RUNTIME_COMPUTE_DEFAULT_KEYS = (
    "reasoning_cycles",
    "adaptive_compute",
    "adaptive_exit_tol",
    "adaptive_exit_entropy",
    "prediction_stability_patience",
    "prediction_stability_tol",
    "prediction_stability_margin",
    "prediction_stability_rank_depth",
    "auto_compute",
)


def _library_runtime_compute_defaults() -> Dict[str, Any]:
    return {
        "reasoning_cycles": None,
        "adaptive_compute": False,
        "adaptive_exit_tol": 1e-3,
        "adaptive_exit_entropy": chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        "prediction_stability_patience": chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE,
        "prediction_stability_tol": chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
        "prediction_stability_margin": chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN,
        "prediction_stability_rank_depth": chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
        "auto_compute": False,
    }


def _normalize_runtime_compute_defaults(values: Dict[str, Any]) -> Dict[str, Any]:
    raw_cycles = values.get("reasoning_cycles")
    reasoning_cycles: Any
    if chat_app._is_auto_reasoning_cycles(raw_cycles):
        reasoning_cycles = "auto"
    else:
        reasoning_cycles = chat_app._coerce_optional_positive_int(
            raw_cycles,
            chat_app.MAX_RUNTIME_REASONING_CYCLES,
        )
    return {
        "reasoning_cycles": reasoning_cycles,
        "adaptive_compute": chat_app._coerce_bool(values.get("adaptive_compute")),
        "adaptive_exit_tol": chat_app._coerce_nonnegative_float(
            values.get("adaptive_exit_tol"),
            1e-3,
        ),
        "adaptive_exit_entropy": chat_app._coerce_nonnegative_float(
            values.get("adaptive_exit_entropy"),
            chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        ),
        "prediction_stability_patience": chat_app._coerce_nonnegative_int(
            values.get("prediction_stability_patience"),
            chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE,
            chat_app.MAX_RUNTIME_REASONING_CYCLES,
        ),
        "prediction_stability_tol": chat_app._coerce_nonnegative_float(
            values.get("prediction_stability_tol"),
            chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
        ),
        "prediction_stability_margin": chat_app._coerce_prediction_stability_margin(
            values.get("prediction_stability_margin"),
        ),
        "prediction_stability_rank_depth": chat_app._coerce_prediction_stability_rank_depth(
            values.get("prediction_stability_rank_depth"),
        ),
        "auto_compute": chat_app._coerce_bool(values.get("auto_compute")),
    }


def _runtime_compute_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Return only compute options the CLI user actually supplied."""
    values = {
        "reasoning_cycles": getattr(args, "reasoning_cycles", None),
        "adaptive_compute": getattr(args, "adaptive_compute", None),
        "adaptive_exit_tol": getattr(args, "adaptive_exit_tol", None),
        "adaptive_exit_entropy": getattr(args, "adaptive_exit_entropy", None),
        "prediction_stability_patience": getattr(args, "prediction_stability_patience", None),
        "prediction_stability_tol": getattr(args, "prediction_stability_tol", None),
        "prediction_stability_margin": getattr(args, "prediction_stability_margin", None),
        "prediction_stability_rank_depth": getattr(args, "prediction_stability_rank_depth", None),
        "auto_compute": getattr(args, "auto_compute", None),
    }
    return {key: value for key, value in values.items() if value is not None}


class Engine:
    def __init__(self, device: Any, device_info: Dict[str, Any], defaults: Dict[str, Any]):
        self.device = device
        self.device_info = dict(device_info or {})
        self._constructor_defaults = dict(defaults or {})
        self.defaults = self._build_effective_defaults({})
        self.lock = threading.RLock()
        self.model = None
        self.weights_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.feature_mode = "legacy"
        self.model_size = "base"
        self.buckets: Dict[int, List[Dict[str, Any]]] = {}
        self.available_labels: List[int] = list(range(chat_app.MODEL_CLASSES))
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}
        self.recent: Dict[str, List[str]] = {}
        registry_path = self._constructor_defaults.get("route_shadow_registry_path")
        self.route_shadow_registry_path = Path(
            registry_path or Path("tmp") / "memory" / "route-policy-shadow-registry.sqlite3"
        ).expanduser().resolve()

    def _build_effective_defaults(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        runtime_defaults = _library_runtime_compute_defaults()
        metadata_defaults = meta.get("runtime_defaults")
        if isinstance(metadata_defaults, dict):
            runtime_defaults.update(
                {
                    key: metadata_defaults[key]
                    for key in _RUNTIME_COMPUTE_DEFAULT_KEYS
                    if key in metadata_defaults
                }
            )
        runtime_defaults.update(
            {
                key: self._constructor_defaults[key]
                for key in _RUNTIME_COMPUTE_DEFAULT_KEYS
                if key in self._constructor_defaults
            }
        )
        effective = dict(self._constructor_defaults)
        effective.update(_normalize_runtime_compute_defaults(runtime_defaults))
        return effective

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "loaded": self.model is not None,
                "weights": self.weights_path,
                "meta": self.meta_path,
                "feature_mode": self.feature_mode,
                "model_size": self.model_size,
                "available_labels": len(self.available_labels),
                "device": self.device_info.get("resolved", str(self.device)),
                "sessions": len(self.sessions),
                "runtime_compute_supported": chat_app.model_supports_runtime_compute(self.model) if self.model is not None else False,
                "reasoning_cycles": chat_app._format_reasoning_cycles_setting(self.defaults.get("reasoning_cycles")),
                "adaptive_compute": bool(self.defaults.get("adaptive_compute", False)),
                "adaptive_exit_tol": chat_app._coerce_nonnegative_float(self.defaults.get("adaptive_exit_tol", 1e-3), 1e-3),
                "adaptive_exit_entropy": chat_app._coerce_nonnegative_float(self.defaults.get("adaptive_exit_entropy", chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY), chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
                "prediction_stability_patience": chat_app._coerce_nonnegative_int(self.defaults.get("prediction_stability_patience", chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE), chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE, chat_app.MAX_RUNTIME_REASONING_CYCLES),
                "prediction_stability_tol": chat_app._coerce_nonnegative_float(self.defaults.get("prediction_stability_tol", chat_app.DEFAULT_PREDICTION_STABILITY_TOL), chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
                "prediction_stability_margin": chat_app._coerce_prediction_stability_margin(self.defaults.get("prediction_stability_margin", chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN)),
                "prediction_stability_rank_depth": chat_app._coerce_prediction_stability_rank_depth(self.defaults.get("prediction_stability_rank_depth", chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH)),
                "auto_compute": chat_app._coerce_bool(self.defaults.get("auto_compute", False)),
            }

    def route_shadow_registry_snapshot(self) -> Dict[str, Any]:
        """Return compatible shadow-registry status without exposing mutations."""

        registry_path = self.route_shadow_registry_path
        if not registry_path.is_file():
            return {
                "ok": True,
                "available": False,
                "status": "not_initialized",
                "registry_location": f"memory/{registry_path.name}",
                "read_only": True,
                "campaign_count": 0,
                "campaigns": [],
                "event_chain": None,
                "execution_enabled": False,
                "activation_available": False,
                "automatic_promotion_allowed": False,
            }
        snapshot = RouteShadowAssignmentRegistry(registry_path, read_only=True).snapshot()
        return {
            **snapshot,
            "available": True,
            "status": "verified" if snapshot.get("ok") else "verification_failed",
            "registry_location": f"memory/{registry_path.name}",
            "read_only": True,
        }

    def _parse_buckets(self, meta: Dict[str, Any]) -> None:
        buckets = chat_app._parse_metadata_buckets(meta.get("buckets", {}))
        self.buckets = buckets
        self.available_labels = sorted(buckets.keys()) or list(range(chat_app.MODEL_CLASSES))

    def load(self, weights: str, meta_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        weights = str(Path(weights))
        meta_path = str(Path(meta_path))
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        meta = chat_app.load_metadata(meta_path)
        effective_defaults = self._build_effective_defaults(meta)
        raw_feature_mode = str(meta.get("feature_mode", "legacy")).strip().lower()
        feature_mode = chat_app.resolve_feature_mode(raw_feature_mode, smarter_auto=True)

        sd = chat_app.safe_load_state_dict(weights)
        inferred = chat_app.detect_model_size_from_state_dict(sd)
        resolved_model_size, _ = chat_app.resolve_runtime_model_size(
            str(effective_defaults.get("model_size", "auto")),
            str(meta.get("model_size", "")),
            inferred,
        )

        expansion_dim = chat_app._resolve_expansion_dim(None, meta, "expansion_dim", chat_app._default_expansion_dim_for_model_size(resolved_model_size), inferred, chat_app.EXPANSION_DIM_MODEL_SIZES, chat_app.detect_large_head_expansion_dim, sd)
        extra_expansion_dim = chat_app._resolve_expansion_dim(None, meta, "extra_expansion_dim", chat_app._default_extra_expansion_dim_for_model_size(resolved_model_size, expansion_dim), inferred, chat_app.EXTRA_EXPANSION_DIM_MODEL_SIZES, chat_app.detect_xlarge_aux_expansion_dim, sd)
        third_expansion_dim = chat_app._resolve_expansion_dim(None, meta, "third_expansion_dim", max(3072, extra_expansion_dim + expansion_dim), inferred, chat_app.THIRD_EXPANSION_DIM_MODEL_SIZES, chat_app.detect_xxlarge_third_expansion_dim, sd)
        fourth_expansion_dim = chat_app._resolve_expansion_dim(None, meta, "fourth_expansion_dim", max(4096, third_expansion_dim + expansion_dim), inferred, chat_app.FOURTH_EXPANSION_DIM_MODEL_SIZES, chat_app.detect_xxxlarge_fourth_expansion_dim, sd)
        fifth_expansion_dim = chat_app._resolve_expansion_dim(None, meta, "fifth_expansion_dim", max(6144, fourth_expansion_dim + expansion_dim), inferred, chat_app.FIFTH_EXPANSION_DIM_MODEL_SIZES, chat_app.detect_ultralarge_fifth_expansion_dim, sd)
        sixth_expansion_dim = chat_app._resolve_expansion_dim(None, meta, "sixth_expansion_dim", max(8192, fifth_expansion_dim + expansion_dim), inferred, chat_app.SIXTH_EXPANSION_DIM_MODEL_SIZES, chat_app.detect_megalarge_sixth_expansion_dim, sd)
        adapter_dropout = float(meta.get("adapter_dropout", 0.1))

        model = chat_app.build_model(
            model_size=resolved_model_size,
            expansion_dim=expansion_dim,
            dropout=adapter_dropout,
            extra_expansion_dim=extra_expansion_dim,
            third_expansion_dim=third_expansion_dim,
            fourth_expansion_dim=fourth_expansion_dim,
            fifth_expansion_dim=fifth_expansion_dim,
            sixth_expansion_dim=sixth_expansion_dim,
        ).to(self.device).eval()
        missing, unexpected = chat_app.load_weights_for_model(model, sd, model_size=resolved_model_size)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

        with self.lock:
            self.model = model
            self.weights_path = weights
            self.meta_path = meta_path
            self.feature_mode = feature_mode
            self.model_size = resolved_model_size
            self.defaults = effective_defaults
            self._parse_buckets(meta)
            self.sessions.clear()
            self.recent.clear()

        return {"ok": True, "load_ms": round((time.perf_counter()-t0)*1000,1), **self.status()}

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)
            self.recent.pop(session_id, None)

    def _resolve_sweep_cycles(self, cycles: Any) -> List[int]:
        return chat_app.resolve_runtime_compute_cycles(cycles)

    def _auto_compute_cycles(self, preferred_cycles: Any = None) -> List[int]:
        return chat_app.runtime_auto_compute_cycles(preferred_cycles)

    def _run_compute_sweep_rows(
        self,
        model,
        x,
        labels: List[int],
        cycles: Any,
        adaptive: bool,
        exit_tol: Optional[float],
        exit_entropy_threshold: Any = None,
        prediction_stability_patience: Any = None,
        prediction_stability_tol: Any = None,
        prediction_stability_margin: Any = None,
        prediction_stability_rank_depth: Any = None,
    ) -> List[Dict[str, Any]]:
        return chat_app.evaluate_runtime_compute_budgets(
            model,
            x,
            labels,
            cycles=cycles,
            adaptive_compute=adaptive,
            exit_tol=exit_tol,
            exit_entropy_threshold=exit_entropy_threshold,
            prediction_stability_patience=prediction_stability_patience,
            prediction_stability_tol=prediction_stability_tol,
            prediction_stability_margin=prediction_stability_margin,
            prediction_stability_rank_depth=prediction_stability_rank_depth,
        )

    def _select_auto_compute_budget(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return chat_app.select_auto_runtime_compute_budget(rows)

    def compute_sweep(
        self,
        session_id: str,
        user_text: str,
        cycles: Any = None,
        adaptive_compute: Any = None,
        adaptive_exit_tol: Any = None,
        adaptive_exit_entropy: Any = None,
        prediction_stability_patience: Any = None,
        prediction_stability_tol: Any = None,
        prediction_stability_margin: Any = None,
        prediction_stability_rank_depth: Any = None,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")
        requested_cycles: List[int] = []
        raw_cycles = cycles if isinstance(cycles, list) and cycles else [1, 3, 8]
        for raw in raw_cycles:
            parsed = chat_app._coerce_optional_positive_int(raw, chat_app.MAX_RUNTIME_REASONING_CYCLES)
            if parsed is not None and parsed not in requested_cycles:
                requested_cycles.append(parsed)
        if not requested_cycles:
            requested_cycles = [1, 3, 8]

        with self.lock:
            if self.model is None:
                raise RuntimeError("No model loaded")
            model = self.model
            feature_mode = self.feature_mode
            labels = list(self.available_labels)
            history = list(self.sessions.get(session_id, []))

        context = chat_app.build_context(history, user_text=user_text, max_turns=int(self.defaults.get("max_turns", 2)))
        x = chat_app.text_to_model_input(context, feature_mode=feature_mode).to(self.device)
        idx = torch.tensor(labels, dtype=torch.long, device=self.device)
        rows: List[Dict[str, Any]] = []
        resolved_prediction_stability_margin = chat_app._coerce_prediction_stability_margin(
            prediction_stability_margin
            if prediction_stability_margin is not None
            else self.defaults.get("prediction_stability_margin", chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN)
        )
        resolved_prediction_stability_rank_depth = chat_app._coerce_prediction_stability_rank_depth(
            prediction_stability_rank_depth
            if prediction_stability_rank_depth is not None
            else self.defaults.get("prediction_stability_rank_depth", chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH)
        )

        with torch.no_grad():
            for cycle_count in requested_cycles:
                t0 = time.perf_counter()
                model_out, compute_diag = chat_app.forward_with_runtime_compute(
                    model,
                    x,
                    reasoning_cycles=cycle_count,
                    adaptive_compute=adaptive_compute if adaptive_compute is not None else self.defaults.get("adaptive_compute", False),
                    exit_tol=adaptive_exit_tol if adaptive_exit_tol is not None else self.defaults.get("adaptive_exit_tol", 1e-3),
                    exit_entropy_threshold=adaptive_exit_entropy if adaptive_exit_entropy is not None else self.defaults.get("adaptive_exit_entropy", chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
                    prediction_stability_patience=prediction_stability_patience if prediction_stability_patience is not None else self.defaults.get("prediction_stability_patience", chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE),
                    prediction_stability_tol=prediction_stability_tol if prediction_stability_tol is not None else self.defaults.get("prediction_stability_tol", chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
                    prediction_stability_margin=resolved_prediction_stability_margin,
                    prediction_class_indices=labels,
                    prediction_stability_rank_depth=resolved_prediction_stability_rank_depth,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                logits = model_out[0, 0]
                avail_logits = logits.index_select(0, idx)
                probs = torch.softmax(avail_logits, dim=0)
                top_pos = int(torch.argmax(probs).item())
                entropy = -torch.sum(probs * torch.log(probs + 1e-9))
                rows.append({
                    "requested_cycles": cycle_count,
                    "latency_ms": latency_ms,
                    "cycles_used": compute_diag.get("cycles_used"),
                    "predicted_label": int(labels[top_pos]),
                    "confidence": float(probs[top_pos].item()),
                    "entropy": float(entropy.item()),
                    "compute": compute_diag,
                })

        return {
            "ok": True,
            "session_id": session_id,
            "history_turns": len(history),
            "rows": rows,
        }

    def chat(
        self,
        session_id: str,
        user_text: str,
        style_mode: Optional[str] = None,
        response_temperature: Optional[float] = None,
        show_top_responses: int = 0,
        reasoning_cycles: Any = None,
        adaptive_compute: Any = None,
        adaptive_exit_tol: Any = None,
        adaptive_exit_entropy: Any = None,
        prediction_stability_patience: Any = None,
        prediction_stability_tol: Any = None,
        auto_compute: Optional[bool] = None,
        prediction_stability_margin: Any = None,
        prediction_stability_rank_depth: Any = None,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")
        with self.lock:
            if self.model is None:
                raise RuntimeError("No model loaded")
            model = self.model
            feature_mode = self.feature_mode
            buckets = self.buckets
            labels = list(self.available_labels)
            history = list(self.sessions.get(session_id, []))
            recent_msgs = list(self.recent.get(session_id, []))
        t0 = time.perf_counter()
        t_infer = 0.0
        t_rank = 0.0

        context = chat_app.build_context(history, user_text=user_text, max_turns=int(self.defaults.get("max_turns", 2)))
        tt = time.perf_counter()
        x = chat_app.text_to_model_input(context, feature_mode=feature_mode).to(self.device)
        resolved_adaptive_compute = (
            self.defaults.get("adaptive_compute", False)
            if adaptive_compute is None
            else adaptive_compute
        )
        resolved_exit_tol = (
            self.defaults.get("adaptive_exit_tol")
            if adaptive_exit_tol is None
            else adaptive_exit_tol
        )
        resolved_prediction_stability_margin = chat_app._coerce_prediction_stability_margin(
            prediction_stability_margin
            if prediction_stability_margin is not None
            else self.defaults.get("prediction_stability_margin", chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN)
        )
        resolved_prediction_stability_rank_depth = chat_app._coerce_prediction_stability_rank_depth(
            prediction_stability_rank_depth
            if prediction_stability_rank_depth is not None
            else self.defaults.get("prediction_stability_rank_depth", chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH)
        )
        effective_reasoning_cycles = (
            self.defaults.get("reasoning_cycles")
            if reasoning_cycles is None
            else reasoning_cycles
        )
        compute_plan: Optional[Dict[str, Any]] = None
        auto_enabled = (
            chat_app._coerce_bool(self.defaults.get("auto_compute", False), default=False)
            if auto_compute is None
            else chat_app._coerce_bool(auto_compute, default=False)
        )
        if auto_enabled and chat_app.model_supports_runtime_compute(model):
            model_out, compute_metrics, compute_plan = chat_app.progressive_auto_compute_forward(
                model,
                x,
                labels,
                cycles=self._auto_compute_cycles(effective_reasoning_cycles),
                adaptive_compute=resolved_adaptive_compute,
                exit_tol=chat_app._coerce_nonnegative_float(
                    resolved_exit_tol,
                    default=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
                ),
                exit_entropy_threshold=adaptive_exit_entropy if adaptive_exit_entropy is not None else self.defaults.get("adaptive_exit_entropy", chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
                prediction_stability_patience=prediction_stability_patience if prediction_stability_patience is not None else self.defaults.get("prediction_stability_patience", chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE),
                prediction_stability_tol=prediction_stability_tol if prediction_stability_tol is not None else self.defaults.get("prediction_stability_tol", chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
                prediction_stability_margin=resolved_prediction_stability_margin,
                prediction_stability_rank_depth=resolved_prediction_stability_rank_depth,
                auto_reasoning_context=context,
            )
            effective_reasoning_cycles = compute_plan.get("selected_reasoning_cycles")
        else:
            with torch.no_grad():
                model_out, compute_metrics = chat_app.forward_with_runtime_compute(
                    model,
                    x,
                    reasoning_cycles=effective_reasoning_cycles,
                    adaptive_compute=resolved_adaptive_compute,
                    exit_tol=resolved_exit_tol,
                    exit_entropy_threshold=adaptive_exit_entropy if adaptive_exit_entropy is not None else self.defaults.get("adaptive_exit_entropy", chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
                    prediction_stability_patience=prediction_stability_patience if prediction_stability_patience is not None else self.defaults.get("prediction_stability_patience", chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE),
                    prediction_stability_tol=prediction_stability_tol if prediction_stability_tol is not None else self.defaults.get("prediction_stability_tol", chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
                    prediction_stability_margin=resolved_prediction_stability_margin,
                    auto_reasoning_context=context,
                    prediction_class_indices=labels,
                    prediction_stability_rank_depth=resolved_prediction_stability_rank_depth,
                )
        logits = model_out[0, 0]
        t_infer += time.perf_counter() - tt

        idx = torch.tensor(labels, dtype=torch.long, device=logits.device)
        avail_logits = logits.index_select(0, idx)
        probs = torch.softmax(avail_logits, dim=0)
        pool_mode = str(self.defaults.get("pool_mode", "all"))
        if pool_mode == "all":
            top_pos = list(range(len(labels)))
        else:
            k = max(1, min(int(self.defaults.get("top_labels", 3)), len(labels)))
            top_pos = torch.topk(avail_logits, k=k).indices.tolist()

        pooled: List[Dict[str, Any]] = []
        for pos in top_pos:
            label = labels[int(pos)]
            bucket_score = float(probs[int(pos)].item())
            for row in buckets.get(label, []):
                m = dict(row)
                m["bucket_score"] = bucket_score
                m["_source"] = "model"
                pooled.append(m)
        if (not pooled) and buckets:
            label = chat_app.choose_bucket_from_logits(logits, labels, temperature=float(self.defaults.get("temperature", 0.0)))
            pooled = list(buckets.get(label, []))

        dedup: Dict[str, Dict[str, Any]] = {}
        for row in pooled:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            prev = dedup.get(text)
            if prev is None:
                d = dict(row)
                d["_sources_set"] = {row.get("_source", "unknown")}
                dedup[text] = d
                continue
            src = row.get("_source", "unknown")
            base = max(float(prev.get("bucket_score", 0.0)), float(row.get("bucket_score", 0.0)))
            if src not in prev["_sources_set"]:
                base *= 1.10
                prev["_sources_set"].add(src)
            prev["bucket_score"] = base
            prev["count"] = int(prev.get("count", 1)) + int(row.get("count", 1))
        for k in list(dedup):
            dedup[k].pop("_sources_set", None)
            dedup[k].pop("_source", None)
        pooled = list(dedup.values())

        resolved_style = chat_app.infer_style_mode(user_text, requested_mode=style_mode or str(self.defaults.get("style_mode", "auto")))
        top_candidates: List[Dict[str, Any]] = []
        show_n = max(0, int(show_top_responses))
        if show_n > 0 and pooled:
            tt = time.perf_counter()
            ranked, scores = chat_app.rank_response_candidates(pooled, query_text=user_text, recent_assistant_messages=recent_msgs, style_mode=resolved_style)
            t_rank += time.perf_counter() - tt
            shown = 0
            for ridx in ranked:
                txt = str(pooled[ridx].get("text", "")).strip()
                if not txt:
                    continue
                top_candidates.append({"score": float(scores[ridx].item()), "text": txt})
                shown += 1
                if shown >= show_n:
                    break

        tt = time.perf_counter()
        resp = chat_app.pick_response(
            pooled,
            query_text=user_text,
            recent_assistant_messages=recent_msgs,
            response_temperature=float(self.defaults.get("response_temperature", 0.08) if response_temperature is None else response_temperature),
            style_mode=resolved_style,
            creativity=max(0.0, min(1.0, float(self.defaults.get("creativity", 0.2)))),
        )
        t_rank += time.perf_counter() - tt
        resp = chat_app.cleanup_response_text(resp) or "I do not have a trained response for that yet."

        with self.lock:
            hist = self.sessions.setdefault(session_id, [])
            hist.append((user_text, resp))
            if len(hist) > 40:
                del hist[:-40]
            recent = self.recent.setdefault(session_id, [])
            recent.append(resp)
            if len(recent) > 24:
                del recent[:-24]

        timing_ms = {
            "infer": round(t_infer * 1000, 1),
            "rank_pick": round(t_rank * 1000, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        }
        if "cycles_used" in compute_metrics:
            timing_ms["cycles_used"] = compute_metrics["cycles_used"]

        return {
            "ok": True,
            "session_id": session_id,
            "response": resp,
            "style_mode": resolved_style,
            "timing_ms": timing_ms,
            "compute": compute_metrics,
            "auto_compute_plan": compute_plan,
            "top_candidates": top_candidates,
        }


def build_app(engine: Engine, default_weights: str, default_meta: str):
    app = Flask(__name__)

    @app.get('/')
    def index():
        html = HTML.replace(
            "<input id='weights' value='' spellcheck='false'>",
            f"<input id='weights' value='{escape(default_weights, quote=True)}' spellcheck='false'>",
        )
        html = html.replace(
            "<input id='meta' value='' spellcheck='false'>",
            f"<input id='meta' value='{escape(default_meta, quote=True)}' spellcheck='false'>",
        )
        return html

    @app.get('/api/status')
    def api_status():
        return jsonify({"ok": True, "status": engine.status()})

    @app.get('/api/route_shadow_registry/status')
    def api_route_shadow_registry_status():
        try:
            response = jsonify({"ok": True, "route_shadow_registry": engine.route_shadow_registry_snapshot()})
            response.headers["Cache-Control"] = "no-store"
            return response
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post('/api/load')
    def api_load():
        p = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(engine.load(str(p.get('weights') or '').strip(), str(p.get('meta') or '').strip()))
        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post('/api/chat')
    def api_chat():
        p = request.get_json(force=True, silent=True) or {}
        sid = str(p.get('session_id') or '').strip() or str(uuid.uuid4())
        msg = str(p.get('message') or '').strip()
        try:
            return jsonify(engine.chat(
                session_id=sid,
                user_text=msg,
                style_mode=p.get('style_mode'),
                response_temperature=p.get('response_temperature'),
                show_top_responses=int(p.get('show_top_responses') or 0),
                reasoning_cycles=p.get('reasoning_cycles'),
                adaptive_compute=p.get('adaptive_compute'),
                adaptive_exit_tol=p.get('adaptive_exit_tol'),
                adaptive_exit_entropy=p.get('adaptive_exit_entropy'),
                prediction_stability_patience=p.get('prediction_stability_patience'),
                prediction_stability_tol=p.get('prediction_stability_tol'),
                auto_compute=p.get('auto_compute'),
                prediction_stability_margin=p.get('prediction_stability_margin'),
                prediction_stability_rank_depth=p.get('prediction_stability_rank_depth'),
            ))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post('/api/compute_sweep')
    def api_compute_sweep():
        p = request.get_json(force=True, silent=True) or {}
        sid = str(p.get('session_id') or '').strip() or str(uuid.uuid4())
        msg = str(p.get('message') or '').strip()
        try:
            return jsonify(engine.compute_sweep(
                session_id=sid,
                user_text=msg,
                cycles=p.get('cycles'),
                adaptive_compute=p.get('adaptive_compute'),
                adaptive_exit_tol=p.get('adaptive_exit_tol'),
                adaptive_exit_entropy=p.get('adaptive_exit_entropy'),
                prediction_stability_patience=p.get('prediction_stability_patience'),
                prediction_stability_tol=p.get('prediction_stability_tol'),
                prediction_stability_margin=p.get('prediction_stability_margin'),
                prediction_stability_rank_depth=p.get('prediction_stability_rank_depth'),
            ))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post('/api/clear')
    def api_clear():
        p = request.get_json(force=True, silent=True) or {}
        sid = str(p.get('session_id') or '').strip()
        if not sid:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        engine.clear(sid)
        return jsonify({"ok": True, "cleared": True, "session_id": sid})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description='Web interface for Champion chat model (loads local weights + metadata).')
    ap.add_argument('--weights', default='champion_model_chat_supermix_v27_500k_ft.pth')
    ap.add_argument('--meta', default='chat_model_meta_supermix_v27_500k.json')
    ap.add_argument('--autoload', action='store_true')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--device_preference', default='cuda,npu,xpu,dml,mps,cpu')
    ap.add_argument('--torch_num_threads', type=int, default=0)
    ap.add_argument('--torch_interop_threads', type=int, default=0)
    ap.add_argument('--matmul_precision', choices=['highest', 'high', 'medium'], default='high')
    ap.add_argument('--disable_tf32', action='store_true')
    ap.add_argument('--model_size', choices=['auto', *chat_app.VALID_RUNTIME_MODEL_SIZES], default='auto')
    ap.add_argument('--max_turns', type=int, default=2)
    ap.add_argument('--top_labels', type=int, default=3)
    ap.add_argument('--pool_mode', choices=['all','topk'], default='all')
    ap.add_argument('--response_temperature', type=float, default=0.08)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--style_mode', choices=['auto','balanced','creative','concise','analyst'], default='auto')
    ap.add_argument('--creativity', type=float, default=0.2)
    ap.add_argument('--reasoning_cycles', type=str, default=None)
    adaptive_compute_group = ap.add_mutually_exclusive_group()
    adaptive_compute_group.add_argument(
        '--adaptive_compute',
        dest='adaptive_compute',
        action='store_true',
        help='enable adaptive compute, overriding checkpoint metadata',
    )
    adaptive_compute_group.add_argument(
        '--no_adaptive_compute',
        dest='adaptive_compute',
        action='store_false',
        help='disable adaptive compute, overriding checkpoint metadata',
    )
    ap.set_defaults(adaptive_compute=None)
    ap.add_argument('--adaptive_exit_tol', type=float, default=None)
    ap.add_argument('--adaptive_exit_entropy', type=float, default=None)
    ap.add_argument('--prediction_stability_patience', type=int, default=None)
    ap.add_argument('--prediction_stability_tol', type=float, default=None)
    ap.add_argument('--prediction_stability_margin', type=float, default=None)
    ap.add_argument('--prediction_stability_rank_depth', type=int, default=None)
    auto_compute_group = ap.add_mutually_exclusive_group()
    auto_compute_group.add_argument(
        '--auto_compute',
        dest='auto_compute',
        action='store_true',
        help='enable automatic compute-budget selection, overriding checkpoint metadata',
    )
    auto_compute_group.add_argument(
        '--no_auto_compute',
        dest='auto_compute',
        action='store_false',
        help='disable automatic compute-budget selection, overriding checkpoint metadata',
    )
    ap.set_defaults(auto_compute=None)
    args = ap.parse_args()

    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    engine_defaults = {
        'model_size': args.model_size,
        'max_turns': int(args.max_turns),
        'top_labels': int(args.top_labels),
        'pool_mode': str(args.pool_mode),
        'response_temperature': float(args.response_temperature),
        'temperature': float(args.temperature),
        'style_mode': str(args.style_mode),
        'creativity': float(args.creativity),
    }
    engine_defaults.update(_runtime_compute_cli_overrides(args))
    engine = Engine(device, device_info, engine_defaults)
    if args.autoload:
        try:
            print(engine.load(args.weights, args.meta))
        except Exception as e:
            print(f'Autoload failed: {e}')
    app = build_app(engine, str(args.weights), str(args.meta))
    print(f'Web UI: http://{args.host}:{args.port}')
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == '__main__':
    main()
