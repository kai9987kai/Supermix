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

try:
    from route_policy_shadow_registry import RouteShadowAssignmentRegistry
except ImportError:  # repository compatibility when only runtime_python is first on sys.path
    try:
        from source.route_policy_shadow_registry import RouteShadowAssignmentRegistry
    except ImportError:  # standalone legacy bundles may omit the shadow console modules
        RouteShadowAssignmentRegistry = None  # type: ignore[assignment,misc]


HTML = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Supermix v27 Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #05080e;
  --card-bg: rgba(18, 28, 48, 0.7);
  --accent: #3b82f6;
  --accent-glow: rgba(59, 130, 246, 0.4);
  --text: #e5edf7;
  --text-dim: #9fb1d1;
  --border: rgba(255, 255, 255, 0.1);
  --glass-border: rgba(255, 255, 255, 0.05);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  background-image: 
    radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
    radial-gradient(at 100% 100%, rgba(29, 78, 216, 0.15) 0px, transparent 50%);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  height: 100vh;
  display: flex;
  overflow: hidden;
}
.wrap {
  width: 100%;
  max-width: 1400px;
  margin: auto;
  height: 95vh;
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 20px;
  padding: 20px;
}
.card {
  background: var(--card-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.side { padding: 24px; overflow-y: auto; }
.side::-webkit-scrollbar { width: 6px; }
.side::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }

.chat { position: relative; }
.header {
  padding: 20px 24px;
  background: rgba(0,0,0,0.2);
  border-bottom: 1px solid var(--glass-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header h2 { margin: 0; font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em; }
.header small { color: var(--text-dim); font-size: 0.8rem; }

.msgs { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth; }
.msgs::-webkit-scrollbar { width: 6px; }
.msgs::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }

.msg {
  max-width: 80%;
  padding: 14px 18px;
  border-radius: 18px;
  line-height: 1.5;
  font-size: 0.95rem;
  position: relative;
  animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.msg.user {
  align-self: flex-end;
  background: var(--accent);
  color: white;
  border-bottom-right-radius: 4px;
  box-shadow: 0 4px 15px var(--accent-glow);
}
.msg.bot {
  align-self: flex-start;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--glass-border);
  border-bottom-left-radius: 4px;
  color: var(--text);
}
.msg .who { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; opacity: 0.7; font-weight: 700; }
.msg .tim { font-size: 0.7rem; margin-top: 10px; opacity: 0.5; font-style: italic; }

.comp {
  padding: 20px 24px;
  background: rgba(0,0,0,0.2);
  border-top: 1px solid var(--glass-border);
  display: flex;
  gap: 12px;
  align-items: flex-end;
}
textarea {
  flex: 1;
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  color: var(--text);
  padding: 12px 16px;
  font-family: inherit;
  font-size: 0.95rem;
  resize: none;
  min-height: 48px;
  max-height: 200px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }

button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s, filter 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}
button:hover { filter: brightness(1.1); }
button:active { transform: scale(0.96); }
button.alt { background: rgba(255,255,255,0.08); border: 1px solid var(--glass-border); }

.row { margin-bottom: 18px; }
.row label { display: block; font-size: 0.7rem; color: var(--text-dim); margin-bottom: 6px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }
select, input {
  width: 100%;
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--glass-border);
  border-radius: 10px;
  color: var(--text);
  padding: 10px 12px;
  font-family: inherit;
  font-size: 0.9rem;
}
.status {
  margin-top: 20px;
  padding: 14px;
  background: rgba(0,0,0,0.15);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  font-family: 'Courier New', monospace;
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: pre-wrap;
  max-height: 200px;
  overflow-y: auto;
}
.btns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
.btn-full { grid-column: span 2; }
.shadow-registry { margin-top:12px; padding:12px; border:1px solid rgba(45,212,191,.2); border-radius:12px; background:rgba(0,0,0,.16); }
.shadow-registry-head { display:flex; align-items:center; justify-content:space-between; gap:8px; color:var(--text); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.shadow-registry-head button { padding:7px 10px; font-size:.7rem; }
.shadow-registry-status { margin-top:8px; color:var(--text-dim); font:11px/1.45 'Courier New',monospace; white-space:pre-wrap; overflow-wrap:anywhere; }

@media (max-width: 900px) {
  .wrap { grid-template-columns: 1fr; height: auto; overflow: visible; }
  body { overflow: visible; }
  .msgs { min-height: 400px; }
}
</style></head><body>
<div class='wrap'>
  <div class='card side'>
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <div style="width:12px; height:12px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent);"></div>
        <h3 style='margin:0'>Supermix v27</h3>
    </div>
    <div style='color:var(--text-dim);font-size:0.85rem;margin-bottom:24px'>Neural Intelligence Framework</div>
    
    <div class='row'><label>Weights</label><input id='weights'></div>
    <div class='row'><label>Metadata</label><input id='meta'></div>
    
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Creative Style</label><select id='style'><option>auto</option><option>balanced</option><option>creative</option><option>concise</option><option>analyst</option></select></div>
        <div class='row'><label>Temperature</label><input id='rt' type='number' min='0' max='1' step='0.01' value='0.08'></div>
    </div>
    
    <div class='row'><label>Inference Width</label><input id='showTop' type='number' min='0' max='10' step='1' value='0'></div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Reasoning Cycles</label><input id='reasoningCycles' type='text' placeholder='default or auto'></div>
        <div class='row'><label>Exit Tolerance</label><input id='exitTol' type='number' min='0' step='0.0001' value='0.001'></div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Exit Entropy</label><input id='exitEntropy' type='number' min='0' step='0.01' value='0.2'></div>
        <div class='row'><label>Stability Tolerance</label><input id='stabilityTol' type='number' min='0' step='0.001' value='0.005'></div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Stability Patience</label><input id='stabilityPatience' type='number' min='0' max='64' step='1' value='2'></div>
        <div class='row'><label>Adaptive Compute</label><select id='adaptiveCompute'><option value='off'>off</option><option value='on'>on</option></select></div>
    </div>
    
    <div class='btns'>
        <button id='loadBtn' class="btn-full">INITIALIZE ENGINE</button>
        <button class='alt' id='statusBtn'>REFRESH</button>
        <button class='alt' id='clearBtn'>PURGE</button>
    </div>
    
    <div class='status' id='statusBox'>System idle.</div>
    <div class='shadow-registry'>
      <div class='shadow-registry-head'><span>Shadow registry - read only</span><button class='alt' id='routeShadowRefresh' type='button'>REFRESH</button></div>
      <div class='shadow-registry-status' id='routeShadowStatus'>Not loaded. Browser mutation, execution, activation, and promotion are unavailable.</div>
    </div>
  </div>
  
  <div class='card chat'>
    <div class='header'>
      <div><h2 id="metaLine">Waiting for Initialization</h2></div>
      <small id='session'></small>
    </div>
    <div class='msgs' id='msgs'></div>
    <div class='comp'>
      <textarea id='prompt' placeholder='Quantum prompt input...' rows="1"></textarea>
      <button id='sendBtn'>SEND</button>
      <button id='sweepBtn' class='alt'>SWEEP</button>
    </div>
  </div>
</div>
<script>
const el=(id)=>document.getElementById(id), msgs=el('msgs'), promptEl=el('prompt');
let sid=localStorage.getItem('champion-web-sid'); 
if(!sid){ sid=String(Date.now())+'-'+Math.random().toString(16).slice(2,10); localStorage.setItem('champion-web-sid',sid); } 
el('session').textContent='SESSION '+sid.slice(0,8).toUpperCase();

promptEl.addEventListener('input', () => {
    promptEl.style.height = 'auto';
    promptEl.style.height = (promptEl.scrollHeight) + 'px';
});

function fmtNum(value,digits=3){const n=Number(value);return Number.isFinite(n)?n.toFixed(digits):null;}
function reasoningCyclesValue(){const raw=el('reasoningCycles').value.trim();if(!raw)return null;const low=raw.toLowerCase();if(['auto','adaptive','smart'].includes(low))return 'auto';const n=Number(raw);return Number.isFinite(n)?n:raw;}
function add(kind,text,timing,top,compute){
    const d=document.createElement('div');
    d.className='msg '+kind;
    const who=document.createElement('div');
    who.className='who';
    who.textContent=kind==='user'?'Human':'Supermix';
    const body=document.createElement('div');
    body.style.whiteSpace='pre-wrap';
    body.textContent=text;
    d.appendChild(who);
    d.appendChild(body);

    if(timing){
        const cycles = timing.cycles_used !== undefined && timing.cycles_used !== null ? ` | Cycles: ${timing.cycles_used}` : '';
        const t=document.createElement('div');
        t.className='tim';
        t.textContent=`Engine Latency: ${timing.total}ms | Infer: ${timing.infer}ms${cycles}`;
        d.appendChild(t);
    }
    if(compute&&compute.applied){
        const parts=[];
        if(compute.reasoning_budget_mode==='auto') parts.push('mode auto');
        if(compute.requested_reasoning_cycles!==undefined&&compute.requested_reasoning_cycles!==null) parts.push(`requested ${compute.requested_reasoning_cycles}`);
        if(compute.cycles_used!==undefined&&compute.cycles_used!==null) parts.push(`used ${compute.cycles_used}`);
        if(compute.exit_reason) parts.push(`exit ${compute.exit_reason}`);
        const streak=fmtNum(compute.prediction_streak); if(streak) parts.push(`stable ${streak}`);
        const drift=fmtNum(compute.prediction_confidence_delta); if(drift) parts.push(`drift ${drift}`);
        const ponder=fmtNum(compute.ponder_cost); if(ponder) parts.push(`ponder ${ponder}`);
        const consistency=fmtNum(compute.consistency_loss); if(consistency) parts.push(`consistency ${consistency}`);
        const entropy=fmtNum(compute.gating_entropy); if(entropy) parts.push(`gate entropy ${entropy}`);
        const exitEntropy=fmtNum(compute.exit_entropy_threshold); if(exitEntropy) parts.push(`exit entropy ${exitEntropy}`);
        if(compute.auto_reasoning_policy&&Array.isArray(compute.auto_reasoning_policy.reasons)) parts.push(`policy ${compute.auto_reasoning_policy.reasons.slice(0,3).join(',')}`);
        if(parts.length){
            const c=document.createElement('div');
            c.className='tim';
            c.textContent='Compute: '+parts.join(' | ');
            d.appendChild(c);
        }
    }
    if(top&&top.length){
        const x=document.createElement('div');
        x.className='tim';
        x.style.borderTop='1px solid rgba(255,255,255,0.05)';
        x.style.paddingTop='8px';
        x.style.marginTop='8px';
        const title=document.createElement('b');
        title.textContent='Neural Probabilities:';
        x.appendChild(title);
        top.forEach((c,i)=>{
            const row=document.createElement('div');
            const score=Number(c.score);
            const scoreText=Number.isFinite(score)?(score*100).toFixed(1):'n/a';
            row.textContent=`${i+1}. ${String(c.text||'').slice(0,100)}... (${scoreText}%)`;
            x.appendChild(row);
        });
        d.appendChild(x);
    }
    msgs.appendChild(d);
    msgs.scrollTop=msgs.scrollHeight;
}

async function jget(path){
    const r=await fetch(path); 
    const d=await r.json(); 
    if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); 
    return d;
}
async function jpost(path,p){
    const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p||{})}); 
    const d=await r.json(); 
    if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); 
    return d;
}
async function refresh(){
    try{
        const d=await jget('/api/status'); 
        el('statusBox').textContent = 'SYSTEM STATUS:\\n' + JSON.stringify(d.status,null,2); 
        el('metaLine').textContent = d.status.loaded ? `${d.status.model_size.toUpperCase()} CORE | ${d.status.device.toUpperCase()}` : 'ENGINE OFFLINE'; 
        if(!el('weights').value&&d.status.weights) el('weights').value=d.status.weights; 
        if(!el('meta').value&&d.status.meta) el('meta').value=d.status.meta; 
        if(!el('reasoningCycles').value&&d.status.reasoning_cycles) el('reasoningCycles').value=d.status.reasoning_cycles;
        el('adaptiveCompute').value = d.status.adaptive_compute ? 'on' : 'off';
        if(d.status.adaptive_exit_tol !== undefined) el('exitTol').value = d.status.adaptive_exit_tol;
        if(d.status.adaptive_exit_entropy !== undefined) el('exitEntropy').value = d.status.adaptive_exit_entropy;
        if(d.status.prediction_stability_patience !== undefined) el('stabilityPatience').value = d.status.prediction_stability_patience;
        if(d.status.prediction_stability_tol !== undefined) el('stabilityTol').value = d.status.prediction_stability_tol;
    }catch(e){ el('statusBox').textContent='TELEMETRY ERROR: '+e.message; }
}
function renderRouteShadowRegistry(snapshot){
    const campaigns=Array.isArray(snapshot&&snapshot.campaigns)?snapshot.campaigns:[];
    const committed=campaigns.reduce((n,row)=>n+(Number(row.commitment_count)||0),0);
    const matched=campaigns.reduce((n,row)=>n+(Number(row.matched_assignment_count)||0),0);
    const processed=campaigns.reduce((n,row)=>n+(Number(row.processed_reveal_count)||0),0);
    const mismatched=campaigns.reduce((n,row)=>n+(Number(row.mismatched_assignment_count)||0),0);
    const chain=snapshot&&snapshot.event_chain;
    if(!snapshot||snapshot.available!==true){
        el('routeShadowStatus').textContent=`Not initialized at ${(snapshot&&snapshot.registry_location)||'the canonical memory path'}.\nRead only - execution, activation, and promotion unavailable.`;
        return;
    }
    const states=[...new Set(campaigns.map(row=>String(row.state||'unknown').replaceAll('_',' ')))];
    el('routeShadowStatus').textContent=
        `${snapshot.ok?'VERIFIED':'VERIFICATION FAILED'} | campaigns ${campaigns.length} | assignments ${matched}/${committed} matched | reveals ${processed} processed | mismatches ${mismatched}\n`+
        `chain ${chain&&chain.ok?'verified':'failed'} (${Number(chain&&chain.verified_events)||0} events) | states ${states.join(', ')||'none'}\n`+
        'Local chain only; no external anchor. Read only - execution, activation, and promotion unavailable.';
}
async function refreshRouteShadowRegistry(){
    const button=el('routeShadowRefresh');button.disabled=true;el('routeShadowStatus').textContent='Reading local registry...';
    try{const d=await jget('/api/route_shadow_registry/status');renderRouteShadowRegistry(d.route_shadow_registry||{});}
    catch(e){el('routeShadowStatus').textContent='REGISTRY STATUS ERROR: '+e.message;}
    finally{button.disabled=false;}
}
async function loadModel(){
    el('statusBox').textContent='SYNCING NEURAL WEIGHTS...'; 
    try{
        const d=await jpost('/api/load',{weights:el('weights').value.trim(),meta:el('meta').value.trim()}); 
        el('statusBox').textContent='SYNC COMPLETE.\\n'+JSON.stringify(d,null,2); 
        refresh();
    }catch(e){ el('statusBox').textContent='INITIALIZATION FAILED: '+e.message; }
}
async function send(){
    const text=promptEl.value.trim(); if(!text) return; 
    add('user',text); 
    promptEl.value=''; 
    promptEl.style.height = 'auto';
    try{
        const cycles = reasoningCyclesValue();
        const d=await jpost('/api/chat',{session_id:sid,message:text,style_mode:el('style').value,response_temperature:Number(el('rt').value),show_top_responses:Number(el('showTop').value),reasoning_cycles:cycles,adaptive_compute:el('adaptiveCompute').value==='on',adaptive_exit_tol:Number(el('exitTol').value),adaptive_exit_entropy:Number(el('exitEntropy').value),prediction_stability_patience:Number(el('stabilityPatience').value),prediction_stability_tol:Number(el('stabilityTol').value)});
        add('bot',d.response,d.timing_ms,d.top_candidates,d.compute);
    }catch(e){ add('bot','CORE ERROR: '+e.message); }
}
async function sweepCompute(){
    const text=promptEl.value.trim(); if(!text) return;
    try{
        const d=await jpost('/api/compute_sweep',{session_id:sid,message:text,cycles:[1,3,8],adaptive_compute:el('adaptiveCompute').value==='on',adaptive_exit_tol:Number(el('exitTol').value),adaptive_exit_entropy:Number(el('exitEntropy').value),prediction_stability_patience:Number(el('stabilityPatience').value),prediction_stability_tol:Number(el('stabilityTol').value)});
        const lines=['Compute sweep for draft prompt:'];
        d.rows.forEach((row)=>{
            const conf=fmtNum(row.confidence);
            const entropy=fmtNum(row.entropy);
            const reason=row.compute&&row.compute.exit_reason?` | exit ${row.compute.exit_reason}`:'';
            lines.push(`cycles ${row.requested_cycles}: ${row.latency_ms}ms | used ${row.cycles_used} | label ${row.predicted_label} | conf ${conf||'n/a'} | entropy ${entropy||'n/a'}${reason}`);
        });
        add('bot',lines.join('\\n'),null,null,d.rows[d.rows.length-1]?.compute||null);
    }catch(e){ add('bot','SWEEP ERROR: '+e.message); }
}
async function clearSess(){
    try{
        await jpost('/api/clear',{session_id:sid}); 
        msgs.innerHTML=''; 
        add('bot','Neural cache purged. Ready for fresh session.');
    }catch(e){ add('bot','PURGE ERROR: '+e.message); }
}
el('loadBtn').onclick=loadModel; el('statusBtn').onclick=refresh; el('clearBtn').onclick=clearSess; el('sendBtn').onclick=send; el('sweepBtn').onclick=sweepCompute; el('routeShadowRefresh').onclick=refreshRouteShadowRegistry;
promptEl.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault();send(); } }); 
refresh(); refreshRouteShadowRegistry();
</script></body></html>"""


_RUNTIME_COMPUTE_DEFAULT_KEYS = (
    "reasoning_cycles",
    "adaptive_compute",
    "adaptive_exit_tol",
    "adaptive_exit_entropy",
    "prediction_stability_patience",
    "prediction_stability_tol",
)


def _library_runtime_compute_defaults() -> Dict[str, Any]:
    return {
        "reasoning_cycles": None,
        "adaptive_compute": False,
        "adaptive_exit_tol": 1e-3,
        "adaptive_exit_entropy": chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        "prediction_stability_patience": chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE,
        "prediction_stability_tol": chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
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
        if RouteShadowAssignmentRegistry is None:
            return {
                "ok": False,
                "available": True,
                "status": "reader_unavailable",
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

        return {"ok": True, "session_id": session_id, "rows": rows}

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
            model_out, compute_diag = chat_app.forward_with_runtime_compute(
                model,
                x,
                reasoning_cycles=reasoning_cycles if reasoning_cycles is not None else self.defaults.get("reasoning_cycles"),
                adaptive_compute=adaptive_compute if adaptive_compute is not None else self.defaults.get("adaptive_compute", False),
                exit_tol=adaptive_exit_tol if adaptive_exit_tol is not None else self.defaults.get("adaptive_exit_tol", 1e-3),
                exit_entropy_threshold=adaptive_exit_entropy if adaptive_exit_entropy is not None else self.defaults.get("adaptive_exit_entropy", chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
                prediction_stability_patience=prediction_stability_patience if prediction_stability_patience is not None else self.defaults.get("prediction_stability_patience", chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE),
                prediction_stability_tol=prediction_stability_tol if prediction_stability_tol is not None else self.defaults.get("prediction_stability_tol", chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
                auto_reasoning_context=context,
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

        return {
            "ok": True,
            "session_id": session_id,
            "response": resp,
            "style_mode": resolved_style,
            "timing_ms": {
                "infer": round(t_infer * 1000, 1),
                "rank_pick": round(t_rank * 1000, 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
                "cycles_used": compute_diag.get("cycles_used"),
            },
            "compute": compute_diag,
            "top_candidates": top_candidates,
        }


def build_app(engine: Engine, default_weights: str, default_meta: str):
    app = Flask(__name__)

    @app.get('/')
    def index():
        html = HTML.replace(
            "<input id='weights'></div>",
            f"<input id='weights' value='{escape(default_weights, quote=True)}'></div>",
        )
        html = html.replace(
            "<input id='meta'></div>",
            f"<input id='meta' value='{escape(default_meta, quote=True)}'></div>",
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
