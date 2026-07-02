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
    <div class='split'><div class='row'><label for='cycles'>Reasoning cycles</label><input id='cycles' type='number' min='1' max='64' step='1' placeholder='auto'></div><div class='row'><label for='adaptive'>Adaptive compute</label><select id='adaptive'><option value='off'>off</option><option value='on'>on</option></select></div></div>
    <div class='row'><label for='exitTol'>Adaptive exit tolerance</label><input id='exitTol' type='number' min='0' step='0.0001' value='0.001'></div>
    <div class='btns'><button id='loadBtn'>Load</button><button class='alt' id='statusBtn'>Refresh</button><button class='alt' id='clearBtn'>Clear</button><button class='alt' id='newSessionBtn'>New ID</button></div>
    <div class='status' id='statusBox'>Loading status...</div>
  </aside>
  <main class='chat'>
    <header class='head'><div><div class='head-title'>Web Chat</div><small id='metaLine'>No model loaded</small></div><div class='pillrow'><span class='pill' id='session'></span><span class='pill' id='runtimePill'>idle</span></div></header>
    <section class='msgs' id='msgs' aria-live='polite'></section>
    <section class='comp'><textarea id='prompt' placeholder='Type a message. Enter sends, Shift+Enter adds a line.'></textarea><button id='sendBtn'>Send</button><button class='alt' id='sweepBtn'>Sweep</button><div class='hint'>Drafts and the local transcript are kept in this browser session.</div><div class='quick'><button type='button' data-fill='Summarize the latest benchmark result and explain the weakest benchmark.'>Benchmark readout</button><button type='button' data-fill='Give a concise debugging checklist for this model response.'>Debug checklist</button><button type='button' data-fill='Answer as a concise analyst and include uncertainty when needed.'>Analyst mode</button></div></section>
  </main>
</div>
<script>
const el=(id)=>document.getElementById(id);
const els={msgs:el('msgs'),prompt:el('prompt'),sendBtn:el('sendBtn'),sweepBtn:el('sweepBtn'),loadBtn:el('loadBtn'),statusBtn:el('statusBtn'),clearBtn:el('clearBtn'),newSessionBtn:el('newSessionBtn'),statusBox:el('statusBox'),metaLine:el('metaLine'),session:el('session'),runtimePill:el('runtimePill'),weights:el('weights'),meta:el('meta'),style:el('style'),rt:el('rt'),showTop:el('showTop'),cycles:el('cycles'),adaptive:el('adaptive'),exitTol:el('exitTol')};
let sid=localStorage.getItem('champion-web-sid');if(!sid){sid=crypto.randomUUID?crypto.randomUUID():String(Date.now());localStorage.setItem('champion-web-sid',sid);}
const draftKey='champion-web-draft-v2';const transcriptKey=()=>('champion-web-transcript-v2-'+sid);let transcript=[];let sending=false;
function setSessionLabel(){els.session.textContent='session '+sid.slice(0,8);}
function loadTranscript(){try{transcript=JSON.parse(localStorage.getItem(transcriptKey())||'[]');}catch(_){transcript=[];}}
function saveTranscript(){localStorage.setItem(transcriptKey(),JSON.stringify(transcript.slice(-80)));}
function autoSizePrompt(){els.prompt.style.height='auto';els.prompt.style.height=Math.min(els.prompt.scrollHeight,190)+'px';}
function setBusy(active,label){sending=active;els.sendBtn.disabled=active;els.sweepBtn.disabled=active;els.loadBtn.disabled=active;els.runtimePill.textContent=label||(active?'working':'idle');}
function timingText(t){if(!t)return'';return `${t.total??'?'} ms total - ${t.infer??'?'} ms infer - ${t.rank_pick??'?'} ms rank`;}
function computeText(c){if(!c)return'';const requested=c.requested_reasoning_cycles??'default';const used=c.cycles_used??'n/a';return `compute: supported=${c.supported} requested=${requested} used=${used} adaptive=${c.adaptive_compute} applied=${c.applied}`;}
function add(kind,text,timing,top,persist=true,compute=null){const card=document.createElement('article');card.className='msg '+kind;const who=document.createElement('div');who.className='who';const label=document.createElement('span');label.textContent=kind==='user'?'You':'Champion';who.appendChild(label);if(kind==='bot'&&text){const copy=document.createElement('button');copy.className='copy';copy.type='button';copy.textContent='Copy';copy.onclick=async()=>{try{await navigator.clipboard.writeText(text);copy.textContent='Copied';setTimeout(()=>{copy.textContent='Copy';},1200);}catch(_){copy.textContent='Failed';}};who.appendChild(copy);}const body=document.createElement('div');body.textContent=text;card.appendChild(who);card.appendChild(body);const tt=timingText(timing);if(tt){const node=document.createElement('div');node.className='tim';node.textContent=tt;card.appendChild(node);}const ct=computeText(compute);if(ct){const node=document.createElement('div');node.className='tim';node.textContent=ct;card.appendChild(node);}if(Array.isArray(top)&&top.length){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent=`Top candidates (${top.length})`;details.appendChild(summary);top.forEach((candidate,index)=>{const row=document.createElement('div');const score=Number(candidate.score);const scoreText=Number.isFinite(score)?score.toFixed(3):'n/a';row.textContent=`${index+1}. (${scoreText}) ${String(candidate.text||'').slice(0,220)}`;details.appendChild(row);});card.appendChild(details);}els.msgs.appendChild(card);els.msgs.scrollTo({top:els.msgs.scrollHeight,behavior:'smooth'});if(persist){transcript.push({kind,text,timing,top,compute,ts:Date.now()});saveTranscript();}return card;}
async function jget(path){const r=await fetch(path);const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||`HTTP ${r.status}`);return d;}
async function jpost(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload||{})});const d=await r.json();if(!r.ok||d.ok===false)throw new Error(d.error||`HTTP ${r.status}`);return d;}
function renderStatus(status){const lines=[status.loaded?'Model loaded':'No model loaded','Device: '+(status.device||'unknown'),'Size: '+(status.model_size||'unknown'),'Features: '+(status.feature_mode||'unknown'),'Labels: '+(status.available_labels??'unknown'),'Runtime compute: '+(status.runtime_compute_supported?'supported':'not supported'),'Sessions: '+(status.sessions??0)];els.statusBox.textContent=lines.join('\\n');els.metaLine.textContent=status.loaded?`${status.model_size} - ${status.feature_mode} - ${status.available_labels} labels`:'Choose model files and load them';els.runtimePill.textContent=status.loaded?'ready':'idle';if(!els.weights.value&&status.weights)els.weights.value=status.weights;if(!els.meta.value&&status.meta)els.meta.value=status.meta;}
async function refresh(){try{const data=await jget('/api/status');renderStatus(data.status);}catch(err){els.statusBox.textContent='Status error: '+err.message;els.runtimePill.textContent='status error';}}
async function loadModel(){setBusy(true,'loading');els.statusBox.textContent='Loading model...';try{const data=await jpost('/api/load',{weights:els.weights.value.trim(),meta:els.meta.value.trim()});renderStatus(data);}catch(err){els.statusBox.textContent='Load error: '+err.message;els.runtimePill.textContent='load failed';}finally{setBusy(false,els.runtimePill.textContent==='load failed'?'load failed':'ready');}}
async function send(){const text=els.prompt.value.trim();if(!text||sending)return;const cycles=els.cycles.value.trim();add('user',text);els.prompt.value='';localStorage.removeItem(draftKey);autoSizePrompt();setBusy(true,'generating');const pending=add('bot','Generating response...',null,null,false);pending.classList.add('pending');try{const data=await jpost('/api/chat',{session_id:sid,message:text,style_mode:els.style.value,response_temperature:Number(els.rt.value),show_top_responses:Number(els.showTop.value),reasoning_cycles:cycles?Number(cycles):null,adaptive_compute:els.adaptive.value==='on',adaptive_exit_tol:Number(els.exitTol.value)});pending.remove();add('bot',data.response,data.timing_ms,data.top_candidates,true,data.compute);els.runtimePill.textContent=data.style_mode?'style '+data.style_mode:'ready';}catch(err){pending.remove();add('bot','Error: '+err.message);els.runtimePill.textContent='chat error';}finally{setBusy(false,els.runtimePill.textContent);}}
async function sweep(){const text=els.prompt.value.trim();if(!text||sending)return;const requested=Number(els.cycles.value);const cycles=Number.isFinite(requested)&&requested>0?[1,requested,Math.min(64,Math.max(requested+1,requested*2))]:[1,3,8];setBusy(true,'sweeping');try{const data=await jpost('/api/compute_sweep',{session_id:sid,message:text,cycles,adaptive_compute:els.adaptive.value==='on',adaptive_exit_tol:Number(els.exitTol.value)});const lines=(data.rows||[]).map((row)=>`cycles ${row.requested_cycles}: ${row.latency_ms} ms, used ${row.cycles_used??'n/a'}, label ${row.predicted_label}, confidence ${Number(row.confidence).toFixed(3)}, entropy ${Number(row.entropy).toFixed(3)}`);add('bot','Compute sweep\\n'+(lines.join('\\n')||'No sweep rows returned.'),null,null,false);els.runtimePill.textContent='sweep ready';}catch(err){add('bot','Sweep error: '+err.message,null,null,false);els.runtimePill.textContent='sweep error';}finally{setBusy(false,els.runtimePill.textContent);}}
async function clearSess(){try{await jpost('/api/clear',{session_id:sid});}catch(_){}transcript=[];localStorage.removeItem(transcriptKey());els.msgs.innerHTML='';add('bot','Session cleared.',null,null,false);}
function newSession(){sid=crypto.randomUUID?crypto.randomUUID():String(Date.now());localStorage.setItem('champion-web-sid',sid);transcript=[];setSessionLabel();els.msgs.innerHTML='';add('bot','New session started.',null,null,false);}
els.loadBtn.onclick=loadModel;els.statusBtn.onclick=refresh;els.clearBtn.onclick=clearSess;els.newSessionBtn.onclick=newSession;els.sendBtn.onclick=send;els.sweepBtn.onclick=sweep;els.prompt.value=localStorage.getItem(draftKey)||'';els.prompt.addEventListener('input',()=>{localStorage.setItem(draftKey,els.prompt.value);autoSizePrompt();});els.prompt.addEventListener('keydown',(event)=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}});
document.querySelectorAll('[data-fill]').forEach((button)=>{button.addEventListener('click',()=>{els.prompt.value=button.dataset.fill||'';localStorage.setItem(draftKey,els.prompt.value);autoSizePrompt();els.prompt.focus();});});
setSessionLabel();loadTranscript();if(transcript.length){transcript.forEach((item)=>add(item.kind,item.text,item.timing,item.top,false,item.compute));}else{add('bot','Session ready. Load a model to begin.',null,null,false);}autoSizePrompt();refresh();
</script></body></html>"""


class Engine:
    def __init__(self, device: Any, device_info: Dict[str, Any], defaults: Dict[str, Any]):
        self.device = device
        self.device_info = dict(device_info or {})
        self.defaults = dict(defaults)
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
                "runtime_compute_supported": bool(
                    self.model is not None and chat_app.model_supports_runtime_compute(self.model)
                ),
                "sessions": len(self.sessions),
            }

    def _parse_buckets(self, meta: Dict[str, Any]) -> None:
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        raw = meta.get("buckets", {})
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    label = int(k)
                except Exception:
                    continue
                if isinstance(v, list) and v:
                    buckets[label] = v
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
        raw_feature_mode = str(meta.get("feature_mode", "legacy")).strip().lower()
        feature_mode = chat_app.resolve_feature_mode(raw_feature_mode, smarter_auto=True)

        sd = chat_app.safe_load_state_dict(weights)
        inferred = chat_app.detect_model_size_from_state_dict(sd)
        resolved_model_size, _ = chat_app.resolve_runtime_model_size(
            str(self.defaults.get("model_size", "auto")),
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
            self._parse_buckets(meta)
            self.sessions.clear()
            self.recent.clear()

        return {"ok": True, "load_ms": round((time.perf_counter()-t0)*1000,1), **self.status()}

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)
            self.recent.pop(session_id, None)

    def _resolve_sweep_cycles(self, cycles: Any) -> List[int]:
        raw_cycles: List[Any]
        if cycles is None or cycles == "":
            raw_cycles = [1, 3, 8]
        elif isinstance(cycles, str):
            raw_cycles = [part.strip() for part in cycles.split(",")]
        elif isinstance(cycles, (list, tuple)):
            raw_cycles = list(cycles)
        else:
            raw_cycles = [cycles]

        resolved: List[int] = []
        seen = set()
        for value in raw_cycles:
            parsed = chat_app._coerce_optional_positive_int(
                value,
                default=None,
                max_value=chat_app.MAX_RUNTIME_REASONING_CYCLES,
            )
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            resolved.append(parsed)
            if len(resolved) >= 8:
                break
        return resolved or [1, 3, 8]

    def compute_sweep(
        self,
        session_id: str,
        user_text: str,
        cycles: Any = None,
        adaptive_compute: Optional[bool] = None,
        adaptive_exit_tol: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")
        with self.lock:
            if self.model is None:
                raise RuntimeError("No model loaded")
            model = self.model
            feature_mode = self.feature_mode
            labels = list(self.available_labels)
            history = list(self.sessions.get(session_id, []))

        context = chat_app.build_context(history, user_text=user_text, max_turns=int(self.defaults.get("max_turns", 2)))
        x = chat_app.text_to_model_input(context, feature_mode=feature_mode).to(self.device)
        adaptive = (
            chat_app._coerce_bool(self.defaults.get("adaptive_compute", False), default=False)
            if adaptive_compute is None
            else chat_app._coerce_bool(adaptive_compute, default=False)
        )
        exit_tol = (
            self.defaults.get("adaptive_exit_tol")
            if adaptive_exit_tol is None
            else adaptive_exit_tol
        )
        exit_tol = chat_app._coerce_nonnegative_float(
            exit_tol,
            default=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
        )

        rows: List[Dict[str, Any]] = []
        idx = torch.tensor(labels, dtype=torch.long, device=self.device)
        for requested_cycles in self._resolve_sweep_cycles(cycles):
            t0 = time.perf_counter()
            with torch.no_grad():
                logits_tensor, compute_metrics = chat_app.forward_with_runtime_compute(
                    model,
                    x,
                    reasoning_cycles=requested_cycles,
                    adaptive_compute=adaptive,
                    exit_tol=exit_tol,
                    return_diagnostics=True,
                )
                logits = logits_tensor[0, 0]
                avail_logits = logits.index_select(0, idx)
                probs = torch.softmax(avail_logits, dim=0)
                confidence_tensor, pred_pos_tensor = torch.max(probs, dim=0)
                entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
            pred_pos = int(pred_pos_tensor.item())
            rows.append(
                {
                    "requested_cycles": int(requested_cycles),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "cycles_used": compute_metrics.get("cycles_used"),
                    "predicted_label": int(labels[pred_pos]),
                    "confidence": round(float(confidence_tensor.item()), 6),
                    "entropy": round(entropy, 6),
                    "compute": compute_metrics,
                }
            )

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
        reasoning_cycles: Optional[int] = None,
        adaptive_compute: Optional[bool] = None,
        adaptive_exit_tol: Optional[float] = None,
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
        with torch.no_grad():
            logits_tensor, compute_metrics = chat_app.forward_with_runtime_compute(
                model,
                x,
                reasoning_cycles=(
                    self.defaults.get("reasoning_cycles")
                    if reasoning_cycles is None
                    else reasoning_cycles
                ),
                adaptive_compute=(
                    self.defaults.get("adaptive_compute", False)
                    if adaptive_compute is None
                    else adaptive_compute
                ),
                exit_tol=(
                    self.defaults.get("adaptive_exit_tol")
                    if adaptive_exit_tol is None
                    else adaptive_exit_tol
                ),
                return_diagnostics=True,
            )
            logits = logits_tensor[0, 0]
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
    ap.add_argument('--reasoning_cycles', type=int, default=None)
    ap.add_argument('--adaptive_compute', action='store_true')
    ap.add_argument('--adaptive_exit_tol', type=float, default=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL)
    args = ap.parse_args()

    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    engine = Engine(device, device_info, {
        'model_size': args.model_size,
        'max_turns': int(args.max_turns),
        'top_labels': int(args.top_labels),
        'pool_mode': str(args.pool_mode),
        'response_temperature': float(args.response_temperature),
        'temperature': float(args.temperature),
        'style_mode': str(args.style_mode),
        'creativity': float(args.creativity),
        'reasoning_cycles': chat_app._coerce_optional_positive_int(
            args.reasoning_cycles,
            default=None,
            max_value=chat_app.MAX_RUNTIME_REASONING_CYCLES,
        ),
        'adaptive_compute': bool(args.adaptive_compute),
        'adaptive_exit_tol': chat_app._coerce_nonnegative_float(
            args.adaptive_exit_tol,
            default=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
        ),
    })
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
