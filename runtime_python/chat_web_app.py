import argparse
import threading
import time
import uuid
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
import torch

import chat_app
from chat_memory import ChatMemoryDB, render_memory_block
from conversation_state import (
    build_conversation_state,
    conversation_state_diagnostics,
)
from device_utils import configure_torch_runtime, resolve_device
from grounding_runtime import (
    build_evidence_bundle,
    finalize_grounded_response,
    plan_grounding,
)
from interaction_planner import (
    finalize_response_for_interaction,
    interaction_plan_diagnostics,
    plan_interaction,
)
from llm_database import LLMDatabase
from prompt_understanding import (
    analyze_prompt,
    evaluate_response_constraints,
    prompt_understanding_diagnostics,
)

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
        <div class='row'><label>Adaptive Compute</label><select id='adaptiveCompute'><option value='off'>off</option><option value='on'>on</option></select></div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Progressive Auto Compute</label><select id='autoCompute'><option value='off'>off</option><option value='on'>on</option></select></div>
        <div class='row'><label>Exit Tolerance</label><input id='exitTol' type='number' min='0' step='0.0001' value='0.001'></div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Exit Entropy</label><input id='exitEntropy' type='number' min='0' step='0.01' value='0.2'></div>
        <div class='row'><label>Stability Tolerance</label><input id='stabilityTol' type='number' min='0' step='0.001' value='0.005'></div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
        <div class='row'><label>Stability Patience</label><input id='stabilityPatience' type='number' min='0' max='64' step='1' value='2'></div>
        <div class='row'><label>Stability Margin</label><input id='stabilityMargin' type='number' min='0' step='0.0001' value='0.0005' title='v51 checkpoint/workload-calibrated default; explicit overrides are supported'></div>
    </div>
    <div class='row'><label>Decision Rank Depth</label><input id='stabilityRankDepth' type='number' min='0' max='10' step='1' value='3'><small style='color:var(--text-dim)'>Verify ordered top ranks; 0 intentionally disables rank verification.</small></div>
    
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
function addOptionalFiniteNumber(payload,key,input){const raw=input.value.trim();if(raw==='')return;const value=Number(raw);if(Number.isFinite(value))payload[key]=value;}
function interactionText(interaction){if(!interaction)return'';const guard=interaction.response_guard||{};return `Interaction: intent ${interaction.intent||'conversation'} | strategy ${interaction.strategy||'direct_then_offer_depth'} | risk ${interaction.risk_tier||'low'} | guard ${guard.reason||'candidate_aligned'}`;}
function understandingText(understanding){if(!understanding)return'';const acts=Array.isArray(understanding.objective_acts)?understanding.objective_acts:(Array.isArray(understanding.acts)?understanding.acts:[]);const ambiguity=understanding.ambiguity||{};const context=understanding.context||{};const normalization=understanding.normalization||{};const safety=understanding.safety||{};const audit=understanding.response_constraint_audit||{};const parts=[`acts ${acts.slice(0,3).join('+')||understanding.primary_act||'direct'}`,`constraints ${understanding.constraint_count??0}`,`ambiguity ${ambiguity.status||understanding.ambiguity_status||understanding.decision||'clear'}`,`turn ${context.turn_relation||understanding.turn_relation||'standalone'}`];if(audit.accepted===true)parts.push('contract pass');else if(audit.accepted===false)parts.push(`contract ${Array.isArray(audit.violations)?audit.violations.length:1} issue(s)`);if(Number(normalization.correction_count||0)>0||safety.typo_recovery_applied||understanding.typo_recovery_applied||understanding.cue_typos_recovered)parts.push('cue typo recovered');return `Understanding: ${parts.join(' | ')}`;}
function answerReceiptText(receipt){if(!receipt||receipt.kind==='none')return'';const verification=receipt.verification||{};const epistemics=receipt.epistemics||{};const parts=[`receipt ${receipt.decision||'not_attempted'}`];if(receipt.problem_class)parts.push(receipt.problem_class);if(receipt.method)parts.push(receipt.method);if(verification.passed)parts.push(verification.independent?'independently verified':'deterministically verified');if(epistemics.model_conditional)parts.push('model-conditional, not calibrated');return parts.join(' | ');}
function groundingText(grounding){if(!grounding)return'';const diagnostics=grounding.diagnostics||{};const guard=grounding.response_guard||{};const base=`Grounding: evidence ${diagnostics.evidence_count??0} | ${diagnostics.sufficiency||'no_evidence'} | guard ${guard.reason||'audit_only'}`;const receipt=answerReceiptText(grounding.answer_receipt);return receipt?`${base} | ${receipt}`:base;}
function add(kind,text,timing,top,compute,interaction,promptUnderstanding,grounding){
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
        if(compute.prediction_verifier_active===true) parts.push('verifier active');
        else if(compute.adaptive_compute) parts.push('verifier inactive');
        if(compute.reasoning_budget_mode==='auto') parts.push('mode auto');
        if(compute.requested_reasoning_cycles!==undefined&&compute.requested_reasoning_cycles!==null) parts.push(`requested ${compute.requested_reasoning_cycles}`);
        if(compute.cycles_used!==undefined&&compute.cycles_used!==null) parts.push(`used ${compute.cycles_used}`);
        if(compute.decision_reference_cycles!==undefined&&compute.decision_reference_cycles!==null) parts.push(`reference ${compute.decision_reference_cycles}`);
        if(compute.exit_reason) parts.push(`exit ${compute.exit_reason}`);
        const streak=fmtNum(compute.prediction_streak); if(streak) parts.push(`stable ${streak}`);
        const drift=fmtNum(compute.prediction_confidence_delta); if(drift) parts.push(`drift ${drift}`);
        const observedMargin=fmtNum(compute.prediction_margin,6); if(observedMargin) parts.push(`top-1 margin ${observedMargin}`);
        const decisionMargin=fmtNum(compute.prediction_decision_margin,6); if(decisionMargin) parts.push(`decision margin ${decisionMargin}`);
        const marginFloor=fmtNum(compute.prediction_stability_margin,6); if(marginFloor) parts.push(`margin floor ${marginFloor}`);
        if(compute.prediction_rank_depth!==undefined&&compute.prediction_rank_depth!==null) parts.push(`verified depth ${compute.prediction_rank_depth}`);
        if(compute.prediction_stability_rank_depth!==undefined&&compute.prediction_stability_rank_depth!==null) parts.push(`requested depth ${compute.prediction_stability_rank_depth}`);
        if(compute.prediction_class_count!==undefined&&compute.prediction_class_count!==null) parts.push(`verifier ${compute.prediction_class_count} classes`);
        if(compute.prediction_class_selection_valid===false) parts.push('verifier scope invalid');
        const ponder=fmtNum(compute.ponder_cost); if(ponder) parts.push(`ponder ${ponder}`);
        const consistency=fmtNum(compute.consistency_loss); if(consistency) parts.push(`consistency ${consistency}`);
        const entropy=fmtNum(compute.gating_entropy); if(entropy) parts.push(`gate entropy ${entropy}`);
        const exitEntropy=fmtNum(compute.exit_entropy_threshold); if(exitEntropy) parts.push(`exit entropy ${exitEntropy}`);
        if(compute.auto_reasoning_policy&&Array.isArray(compute.auto_reasoning_policy.reasons)) parts.push(`policy ${compute.auto_reasoning_policy.reasons.slice(0,3).join(',')}`);
        if(compute.auto_compute_plan){
            const plan=compute.auto_compute_plan;
            parts.push(`budget ${plan.selected_reasoning_cycles} (${plan.reason})`);
            parts.push(`${plan.forward_evaluations}/${plan.legacy_forward_evaluations} forwards`);
            if(plan.reused_probe_output) parts.push('probe reused');
            const rows=Array.isArray(plan.rows)?plan.rows:[];
            const selectedIndex=Number(plan.selected_index);
            const selectedRow=Number.isInteger(selectedIndex)&&selectedIndex>=0&&selectedIndex<rows.length?rows[selectedIndex]:(rows.length?rows[rows.length-1]:null);
            const shadow=selectedRow?selectedRow.mutual_stability_shadow:null;
            const js=shadow?fmtNum(shadow.js_divergence,6):null;
            if(js) parts.push(`shadow JSD ${js}`);
        }
        if(parts.length){
            const c=document.createElement('div');
            c.className='tim';
            c.textContent='Compute: '+parts.join(' | ');
            d.appendChild(c);
        }
    }
    const understandingSummary=understandingText(promptUnderstanding);
    if(understandingSummary){
        const node=document.createElement('div');
        node.className='tim';
        node.textContent=understandingSummary;
        d.appendChild(node);
    }
    const interactionSummary=interactionText(interaction);
    if(interactionSummary){
        const node=document.createElement('div');
        node.className='tim';
        node.textContent=interactionSummary;
        d.appendChild(node);
    }
    const groundingSummary=groundingText(grounding);
    if(groundingSummary){
        const node=document.createElement('div');
        node.className='tim';
        node.textContent=groundingSummary;
        d.appendChild(node);
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
        el('autoCompute').value = d.status.auto_compute ? 'on' : 'off';
        if(d.status.adaptive_exit_tol !== undefined) el('exitTol').value = d.status.adaptive_exit_tol;
        if(d.status.adaptive_exit_entropy !== undefined) el('exitEntropy').value = d.status.adaptive_exit_entropy;
        if(d.status.prediction_stability_patience !== undefined) el('stabilityPatience').value = d.status.prediction_stability_patience;
        if(d.status.prediction_stability_tol !== undefined) el('stabilityTol').value = d.status.prediction_stability_tol;
        if(d.status.prediction_stability_margin !== undefined) el('stabilityMargin').value = d.status.prediction_stability_margin;
        if(d.status.prediction_stability_rank_depth !== undefined) el('stabilityRankDepth').value = d.status.prediction_stability_rank_depth;
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
        const payload={session_id:sid,message:text,style_mode:el('style').value,response_temperature:Number(el('rt').value),show_top_responses:Number(el('showTop').value),reasoning_cycles:cycles,adaptive_compute:el('adaptiveCompute').value==='on',auto_compute:el('autoCompute').value==='on',adaptive_exit_tol:Number(el('exitTol').value),adaptive_exit_entropy:Number(el('exitEntropy').value),prediction_stability_patience:Number(el('stabilityPatience').value),prediction_stability_tol:Number(el('stabilityTol').value)};
        addOptionalFiniteNumber(payload,'prediction_stability_margin',el('stabilityMargin'));
        addOptionalFiniteNumber(payload,'prediction_stability_rank_depth',el('stabilityRankDepth'));
        const d=await jpost('/api/chat',payload);
        add('bot',d.response,d.timing_ms,d.top_candidates,d.compute,d.interaction,d.prompt_understanding,d.grounding);
    }catch(e){ add('bot','CORE ERROR: '+e.message); }
}
async function sweepCompute(){
    const text=promptEl.value.trim(); if(!text) return;
    try{
        const payload={session_id:sid,message:text,cycles:[1,3,8],adaptive_compute:el('adaptiveCompute').value==='on',adaptive_exit_tol:Number(el('exitTol').value),adaptive_exit_entropy:Number(el('exitEntropy').value),prediction_stability_patience:Number(el('stabilityPatience').value),prediction_stability_tol:Number(el('stabilityTol').value)};
        addOptionalFiniteNumber(payload,'prediction_stability_margin',el('stabilityMargin'));
        addOptionalFiniteNumber(payload,'prediction_stability_rank_depth',el('stabilityRankDepth'));
        const d=await jpost('/api/compute_sweep',payload);
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


# Bounded so a long-lived server cannot accumulate one lock per session id.
MAX_SESSION_TURN_LOCKS = 512


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
    "core_top_k",
    "verifier_adaptive_compute",
    "verifier_continue_threshold",
    "max_verifier_cycles",
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
        # v52 controls default to off: sparse dispatch is not always faster on
        # small CPU batches, and the verifier head is untrained on v50 imports.
        "core_top_k": None,
        "verifier_adaptive_compute": False,
        "verifier_continue_threshold": chat_app.DEFAULT_VERIFIER_CONTINUE_THRESHOLD,
        "max_verifier_cycles": chat_app.DEFAULT_MAX_VERIFIER_CYCLES,
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
        "core_top_k": chat_app._coerce_optional_positive_int(
            values.get("core_top_k"),
            chat_app.MAX_RUNTIME_CORE_TOP_K,
        ),
        "verifier_adaptive_compute": chat_app._coerce_bool(
            values.get("verifier_adaptive_compute")
        ),
        "verifier_continue_threshold": chat_app._coerce_unit_interval(
            values.get("verifier_continue_threshold"),
            chat_app.DEFAULT_VERIFIER_CONTINUE_THRESHOLD,
        ),
        "max_verifier_cycles": chat_app._coerce_nonnegative_int(
            values.get("max_verifier_cycles"),
            chat_app.DEFAULT_MAX_VERIFIER_CYCLES,
            chat_app.MAX_RUNTIME_REASONING_CYCLES,
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
        "prediction_stability_margin": getattr(args, "prediction_stability_margin", None),
        "prediction_stability_rank_depth": getattr(args, "prediction_stability_rank_depth", None),
        "auto_compute": getattr(args, "auto_compute", None),
        "core_top_k": getattr(args, "core_top_k", None),
        "verifier_adaptive_compute": getattr(args, "verifier_adaptive_compute", None),
        "verifier_continue_threshold": getattr(args, "verifier_continue_threshold", None),
        "max_verifier_cycles": getattr(args, "max_verifier_cycles", None),
    }
    return {key: value for key, value in values.items() if value is not None}


class Engine:
    def __init__(self, device: Any, device_info: Dict[str, Any], defaults: Dict[str, Any]):
        self.device = device
        self.device_info = dict(device_info or {})
        self._constructor_defaults = dict(defaults or {})
        self.defaults = self._build_effective_defaults({})
        self.lock = threading.RLock()
        # Serializes model inference. The heads store their telemetry on
        # themselves (`last_*`), so concurrent forwards would report each
        # other's metrics.
        self.inference_lock = threading.Lock()
        self.session_turn_locks: Dict[str, threading.Lock] = {}
        self.model = None
        self.weights_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.feature_mode = "legacy"
        self.model_size = "base"
        self.buckets: Dict[int, List[Dict[str, Any]]] = {}
        self.available_labels: List[int] = list(range(chat_app.MODEL_CLASSES))
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}
        self.recent: Dict[str, List[str]] = {}
        llm_db_path = str(self._constructor_defaults.get("llm_db") or "").strip()
        self.llm_db_path = str(Path(llm_db_path).expanduser().resolve()) if llm_db_path else ""
        self.llm_db: Optional[LLMDatabase] = (
            LLMDatabase(self.llm_db_path)
            if self.llm_db_path and Path(self.llm_db_path).is_file()
            else None
        )
        memory_db_path = str(self._constructor_defaults.get("memory_db") or "").strip()
        self.memory_db_path = (
            str(Path(memory_db_path).expanduser().resolve()) if memory_db_path else ""
        )
        self.memory_db: Optional[ChatMemoryDB] = (
            ChatMemoryDB(self.memory_db_path)
            if self.memory_db_path
            and not bool(self._constructor_defaults.get("disable_memory", False))
            else None
        )
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

    def _session_turn_lock(self, session_id: str) -> "threading.Lock":
        """One turn at a time per session.

        `chat` snapshots the session history, runs inference outside the engine
        lock, then appends the finished turn. Two concurrent requests for the
        same session would both snapshot the same history and the earlier turn
        would be lost, which also corrupts the derived conversation state.
        """

        with self.lock:
            lock = self.session_turn_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self.session_turn_locks[session_id] = lock
                if len(self.session_turn_locks) > MAX_SESSION_TURN_LOCKS:
                    for stale in list(self.session_turn_locks)[:-MAX_SESSION_TURN_LOCKS]:
                        if not self.session_turn_locks[stale].locked():
                            del self.session_turn_locks[stale]
        return lock

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
                "knowledge": {
                    "llm_db_available": self.llm_db is not None,
                    "persistent_memory_available": self.memory_db is not None,
                    "llm_db_file": Path(self.llm_db_path).name if self.llm_db_path else None,
                    "memory_db_file": Path(self.memory_db_path).name if self.memory_db_path else None,
                },
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

        with self.inference_lock, torch.no_grad():
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
                    core_top_k=self.defaults.get("core_top_k"),
                    verifier_adaptive_compute=self.defaults.get("verifier_adaptive_compute", False),
                    verifier_continue_threshold=self.defaults.get(
                        "verifier_continue_threshold", chat_app.DEFAULT_VERIFIER_CONTINUE_THRESHOLD
                    ),
                    max_verifier_cycles=self.defaults.get(
                        "max_verifier_cycles", chat_app.DEFAULT_MAX_VERIFIER_CYCLES
                    ),
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
        interaction_enabled: bool = True,
        interaction_plan: Optional[Dict[str, Any]] = None,
        interaction_user_text: Optional[str] = None,
        prompt_profile: Optional[Dict[str, Any]] = None,
        grounding_enabled: bool = True,
        grounding_plan: Optional[Dict[str, Any]] = None,
        conversation_enabled: bool = True,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")
        with self._session_turn_lock(session_id):
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
            t_retrieval = 0.0
            interaction_request_text = str(interaction_user_text or user_text)
            recent_turns = [
                {
                    "id": f"turn-{index + 1}",
                    "user": str(prior_user or ""),
                    "assistant": str(prior_assistant or ""),
                }
                for index, (prior_user, prior_assistant) in enumerate(history[-4:])
            ]
            recent_user_messages = [
                str(prior_user or "")
                for prior_user, _ in history[-4:]
                if str(prior_user or "").strip()
            ]
            recent_assistant_messages = [
                str(message or "")
                for message in recent_msgs[-4:]
                if str(message or "").strip()
            ]
            if prompt_profile is None:
                prompt_profile = analyze_prompt(
                    interaction_request_text,
                    recent_turns=recent_turns,
                    recent_user_messages=recent_user_messages,
                    recent_assistant_messages=recent_assistant_messages,
                )
            else:
                prompt_profile = dict(prompt_profile)
            if not interaction_enabled:
                interaction_plan = None
            elif interaction_plan is None:
                interaction_plan = plan_interaction(
                    interaction_request_text,
                    recent_assistant_messages=recent_assistant_messages,
                    context={
                        "recent_user_messages": recent_user_messages,
                        "recent_turns": recent_turns,
                    },
                    prompt_profile=prompt_profile,
                )
            if not grounding_enabled:
                grounding_plan = None
            elif grounding_plan is None:
                grounding_plan = plan_grounding(
                    interaction_request_text,
                    interaction_plan=interaction_plan,
                    prompt_profile=prompt_profile,
                )
            # Accumulated over the whole session rather than the bounded window the
            # planner sees. A controlled evaluation can switch it off entirely.
            conversation_state = (
                build_conversation_state(history, current_user_text=interaction_request_text)
                if conversation_enabled
                else None
            )

            retrieval_started = time.perf_counter()
            memory_rows: List[Dict[str, Any]] = []
            if self.memory_db is not None:
                memory_rows = self.memory_db.query(
                    interaction_request_text,
                    top_k=max(1, int(self.defaults.get("memory_top_k", 4))),
                    pool_size=max(1, int(self.defaults.get("memory_pool_size", 400))),
                    recency_half_life_hours=max(
                        1.0,
                        float(self.defaults.get("memory_recency_half_life_hours", 168.0)),
                    ),
                )
            db_candidates: List[Dict[str, Any]] = []
            if self.llm_db is not None:
                db_query = chat_app._build_db_query(
                    user=interaction_request_text,
                    history=history,
                    memory_rows=memory_rows,
                    max_turns=max(0, int(self.defaults.get("db_query_context_turns", 2))),
                    prompt_profile=prompt_profile,
                    recent_turns=recent_turns,
                )
                db_candidates = self.llm_db.query(
                    db_query or interaction_request_text,
                    top_k=max(1, int(self.defaults.get("db_top_k", 120))),
                    exact_user_text=interaction_request_text,
                )
            exact_db_candidates = [
                row for row in db_candidates if bool(row.get("exact_user_match", False))
            ]
            grounded_db_candidates = exact_db_candidates or db_candidates
            evidence_rows = [
                {
                    "title": str(row.get("source_title") or ""),
                    "text": str(row.get("text") or ""),
                    "source": str(row.get("source_uri") or "local_llm_db"),
                    "source_type": str(row.get("source_type") or "local_dataset"),
                    "score": float(row.get("bucket_score") or 0.0),
                }
                for row in grounded_db_candidates
                if str(row.get("text") or "").strip()
            ]
            evidence_bundle = (
                build_evidence_bundle(
                    interaction_request_text,
                    evidence_rows,
                    interaction_plan=interaction_plan,
                    max_items=int((grounding_plan or {}).get("max_evidence_items") or 6),
                    grounding_plan=grounding_plan,
                    prompt_profile=prompt_profile,
                )
                if grounding_enabled
                else None
            )
            t_retrieval += time.perf_counter() - retrieval_started

            context = chat_app.build_context(history, user_text=user_text, max_turns=int(self.defaults.get("max_turns", 2)))
            if memory_rows:
                memory_block = render_memory_block(memory_rows)
                if memory_block:
                    context = memory_block + "\n" + context
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
                with self.inference_lock:
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
                with self.inference_lock, torch.no_grad():
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
                        core_top_k=self.defaults.get("core_top_k"),
                        verifier_adaptive_compute=self.defaults.get("verifier_adaptive_compute", False),
                        verifier_continue_threshold=self.defaults.get(
                            "verifier_continue_threshold", chat_app.DEFAULT_VERIFIER_CONTINUE_THRESHOLD
                        ),
                        max_verifier_cycles=self.defaults.get(
                            "max_verifier_cycles", chat_app.DEFAULT_MAX_VERIFIER_CYCLES
                        ),
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
            if not exact_db_candidates:
                for pos in top_pos:
                    label = labels[int(pos)]
                    bucket_score = float(probs[int(pos)].item())
                    for row in buckets.get(label, []):
                        m = dict(row)
                        m["bucket_score"] = bucket_score
                        m["_source"] = "model"
                        pooled.append(m)
            for row in grounded_db_candidates:
                merged = dict(row)
                merged["bucket_score"] = float(merged.get("bucket_score", 0.0)) * float(
                    self.defaults.get("db_score_scale", 1.0)
                )
                merged["_source"] = "llm_db"
                pooled.append(merged)
            for row in ([] if exact_db_candidates else memory_rows):
                text = str(row.get("assistant_text", "")).strip()
                vec = row.get("assistant_vec")
                ctx_vec = row.get("user_vec")
                if not text or not isinstance(vec, list) or not isinstance(ctx_vec, list):
                    continue
                pooled.append(
                    {
                        "text": text,
                        "count": 1,
                        "vec": vec,
                        "ctx_vec": ctx_vec,
                        "bucket_score": float(
                            max(0.0, float(row.get("score", 0.0)))
                            * float(self.defaults.get("memory_score_scale", 0.45))
                        ),
                        "_source": "memory",
                    }
                )
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

            resolved_style = chat_app.infer_style_mode(
                user_text,
                requested_mode=style_mode or str(self.defaults.get("style_mode", "auto")),
                conversation_state=conversation_state,
            )
            top_candidates: List[Dict[str, Any]] = []
            show_n = max(0, int(show_top_responses))
            if show_n > 0 and pooled:
                tt = time.perf_counter()
                ranked, scores = chat_app.rank_response_candidates(
                    pooled,
                    query_text=user_text,
                    recent_assistant_messages=recent_msgs,
                    style_mode=resolved_style,
                    interaction_plan=interaction_plan,
                    conversation_state=conversation_state,
                )
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
                interaction_plan=interaction_plan,
                conversation_state=conversation_state,
            )
            t_rank += time.perf_counter() - tt
            resp = chat_app.cleanup_response_text(resp) or "I do not have a trained response for that yet."
            grounding_diag: Optional[Dict[str, Any]] = None
            if grounding_plan is not None:
                grounding_guard = finalize_grounded_response(
                    resp,
                    interaction_request_text,
                    grounding_plan=grounding_plan,
                    evidence_bundle=evidence_bundle,
                    prompt_profile=prompt_profile,
                )
                resp = str(grounding_guard["text"])
                grounding_diag = {
                    "schema_version": str((evidence_bundle or {}).get("schema_version") or ""),
                    "plan": grounding_plan,
                    "source_ids": [
                        str(row.get("id") or "")
                        for row in (evidence_bundle or {}).get("evidence", [])
                    ],
                    "diagnostics": dict(grounding_guard.get("grounding") or {}),
                    "response_guard": {
                        "changed": bool(grounding_guard.get("changed", False)),
                        "reason": str(grounding_guard.get("reason") or "audit_only"),
                    },
                    "answer_receipt": dict(
                        grounding_guard.get("answer_receipt") or {}
                    ),
                    "authority": dict(grounding_guard.get("authority") or {}),
                }
            interaction_diag: Optional[Dict[str, Any]] = None
            if interaction_plan is not None:
                response_guard = finalize_response_for_interaction(
                    resp,
                    interaction_request_text,
                    interaction_plan,
                    relevance_context=history[-1][0] if history else "",
                )
                resp = str(response_guard["text"])
                interaction_diag = interaction_plan_diagnostics(interaction_plan)
                interaction_diag["response_guard"] = {
                    "changed": bool(response_guard.get("changed", False)),
                    "reason": str(response_guard.get("reason", "candidate_aligned")),
                    "audit": dict(response_guard.get("audit", {})),
                }
            understanding_diag = prompt_understanding_diagnostics(prompt_profile)
            understanding_diag["response_constraint_audit"] = (
                evaluate_response_constraints(
                    resp,
                    interaction_request_text,
                    prompt_profile,
                )
            )

            with self.lock:
                hist = self.sessions.setdefault(session_id, [])
                hist.append((user_text, resp))
                if len(hist) > 40:
                    del hist[:-40]
                recent = self.recent.setdefault(session_id, [])
                recent.append(resp)
                if len(recent) > 24:
                    del recent[:-24]
            if self.memory_db is not None:
                self.memory_db.add_turn(interaction_request_text, resp)

            timing_ms = {
                "retrieval": round(t_retrieval * 1000, 1),
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
                "prompt_understanding": understanding_diag,
                "interaction": interaction_diag,
                "conversation": (
                    conversation_state_diagnostics(conversation_state)
                    if conversation_state is not None
                    else None
                ),
                "grounding": grounding_diag,
                "knowledge": {
                    "llm_db_enabled": self.llm_db is not None,
                    "memory_enabled": self.memory_db is not None,
                    "llm_db_hits": len(db_candidates),
                    "memory_hits": len(memory_rows),
                },
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
                auto_compute=p.get('auto_compute'),
                prediction_stability_margin=p.get('prediction_stability_margin'),
                prediction_stability_rank_depth=p.get('prediction_stability_rank_depth'),
                grounding_enabled=bool(p.get('grounding_enabled', True)),
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
    ap.add_argument('--llm_db', default='llm_chat.db', help='Optional local retrieval database.')
    ap.add_argument('--db_top_k', type=int, default=120)
    ap.add_argument('--db_query_context_turns', type=int, default=2)
    ap.add_argument('--db_score_scale', type=float, default=1.0)
    ap.add_argument('--memory_db', default='chat_memory.db', help='Persistent local chat memory.')
    ap.add_argument('--memory_top_k', type=int, default=4)
    ap.add_argument('--memory_pool_size', type=int, default=400)
    ap.add_argument('--memory_recency_half_life_hours', type=float, default=168.0)
    ap.add_argument('--memory_score_scale', type=float, default=0.45)
    ap.add_argument('--disable_memory', action='store_true')
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
    ap.add_argument(
        '--core_top_k',
        type=int,
        default=None,
        help='v52 only: execute just the top-k routed recurrent cores (off by default).',
    )
    ap.add_argument(
        '--verifier_adaptive_compute',
        action='store_true',
        default=None,
        help='v52 only: let the quality head request extra recursive cycles.',
    )
    ap.add_argument('--verifier_continue_threshold', type=float, default=None)
    ap.add_argument('--max_verifier_cycles', type=int, default=None)
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
        'llm_db': str(args.llm_db),
        'db_top_k': int(args.db_top_k),
        'db_query_context_turns': int(args.db_query_context_turns),
        'db_score_scale': float(args.db_score_scale),
        'memory_db': str(args.memory_db),
        'memory_top_k': int(args.memory_top_k),
        'memory_pool_size': int(args.memory_pool_size),
        'memory_recency_half_life_hours': float(args.memory_recency_half_life_hours),
        'memory_score_scale': float(args.memory_score_scale),
        'disable_memory': bool(args.disable_memory),
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
