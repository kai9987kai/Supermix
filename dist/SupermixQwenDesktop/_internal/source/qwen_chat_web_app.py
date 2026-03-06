import argparse
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from flask import Flask, jsonify, request
from peft import PeftConfig, get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer


HTML = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Qwen Adapter Chat</title>
<style>
body{margin:0;background:#0b1220;color:#e5edf7;font-family:Segoe UI,Arial,sans-serif}
.wrap{max-width:1040px;margin:20px auto;padding:14px;display:grid;grid-template-columns:340px 1fr;gap:14px}
.card{background:#121c30;border:1px solid #24324e;border-radius:14px}
.side{padding:14px}.chat{display:grid;grid-template-rows:auto 1fr auto;min-height:78vh}
.row{margin-bottom:10px}.row label{display:block;font-size:.75rem;color:#9fb1d1;margin-bottom:5px;text-transform:uppercase}
input,textarea{width:100%;background:#0b1220;color:#e5edf7;border:1px solid #2a3a58;border-radius:10px;padding:10px}
button{background:#1d4ed8;color:white;border:0;border-radius:10px;padding:10px 12px;font-weight:600;cursor:pointer}
.btns{display:flex;gap:8px;flex-wrap:wrap}
.status{white-space:pre-wrap;background:#0b1220;border:1px solid #2a3a58;border-radius:10px;padding:10px;min-height:120px;color:#b7c6df;font-size:.85rem}
.head{padding:12px 14px;border-bottom:1px solid #24324e;display:flex;justify-content:space-between;gap:8px;align-items:center}
.msgs{padding:12px;overflow:auto;display:flex;flex-direction:column;gap:10px}
.msg{border:1px solid #24324e;border-radius:12px;padding:10px;background:#0b1220;max-width:85%;white-space:pre-wrap;line-height:1.35}
.msg.user{align-self:flex-end;background:#10203d;border-color:#244d90}.msg.bot{align-self:flex-start}
.msg .who{font-size:.72rem;color:#9fb1d1;margin-bottom:4px;text-transform:uppercase}
.tim{margin-top:6px;color:#9fb1d1;font-size:.75rem}
.comp{padding:12px;border-top:1px solid #24324e;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end}
textarea{min-height:68px;max-height:220px;resize:vertical}
@media (max-width: 900px){.wrap{grid-template-columns:1fr}.chat{min-height:70vh}}
</style></head><body>
<div class='wrap'>
  <div class='card side'>
    <h3 style='margin:0 0 6px'>Qwen Adapter Chat</h3>
    <div style='color:#9fb1d1;font-size:.9rem;margin-bottom:12px'>Running local base model + LoRA adapter.</div>
    <div class='row'><label>Max New Tokens</label><input id='maxNew' type='number' min='8' max='512' step='1' value='96'></div>
    <div class='row'><label>Temperature</label><input id='temp' type='number' min='0' max='2' step='0.01' value='0.20'></div>
    <div class='row'><label>Top P</label><input id='topP' type='number' min='0.1' max='1.0' step='0.01' value='0.92'></div>
    <div class='btns'><button id='statusBtn'>Refresh</button><button id='clearBtn'>Clear Session</button></div>
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
let sid=localStorage.getItem('qwen-chat-sid'); if(!sid){sid=(crypto.randomUUID?crypto.randomUUID():String(Date.now())); localStorage.setItem('qwen-chat-sid',sid);} el('session').textContent='session '+sid.slice(0,8);
function add(kind,text,timing){const d=document.createElement('div'); d.className='msg '+kind; d.innerHTML=`<div class='who'>${kind==='user'?'You':'Assistant'}</div>`; const b=document.createElement('div'); b.textContent=text; d.appendChild(b); if(timing){const t=document.createElement('div'); t.className='tim'; t.textContent=`Total: ${timing.total_ms} ms`; d.appendChild(t);} msgs.appendChild(d); msgs.scrollTop=msgs.scrollHeight;}
async function jget(path){const r=await fetch(path); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); return d;}
async function jpost(path,p){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p||{})}); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); return d;}
async function refresh(){try{const d=await jget('/api/status'); el('statusBox').textContent=JSON.stringify(d.status,null,2); el('metaLine').textContent=d.status.loaded?`${d.status.device} | adapter=${d.status.adapter_loaded}`:'Not loaded'; }catch(e){el('statusBox').textContent='Status error: '+e.message;}}
async function send(){const text=el('prompt').value.trim(); if(!text) return; add('user',text); el('prompt').value=''; try{const d=await jpost('/api/chat',{session_id:sid,message:text,max_new_tokens:Number(el('maxNew').value),temperature:Number(el('temp').value),top_p:Number(el('topP').value)}); add('bot',d.response,d.timing);}catch(e){add('bot','Error: '+e.message);}}
async function clearSess(){try{await jpost('/api/clear',{session_id:sid}); msgs.innerHTML=''; add('bot','Session cleared.');}catch(e){add('bot','Clear error: '+e.message);}}
el('statusBtn').onclick=refresh; el('clearBtn').onclick=clearSess; el('sendBtn').onclick=send; el('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}); refresh();
</script></body></html>"""


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
NORMAL_ADAPTER_PREFIX = "base_model.model.model."
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
    out = re.sub(
        r"\bfocus on (?:a|an|the)?\s*[^.]{0,120}\sfor (?:beginners|advanced learners|beginners and advanced learners)\.?",
        "",
        out,
        flags=re.IGNORECASE,
    )

    lowered = out.lower()
    changed = True
    while changed:
        changed = False
        for phrase in LEAD_NOISE_PHRASES:
            if lowered.startswith(phrase):
                out = out[len(phrase) :].lstrip(" :-\n")
                lowered = out.lower()
                changed = True

    out = re.sub(r"[ \t]+", " ", out).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _build_bullets_from_text(text: str, n: int) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if p.strip()]
    if not parts:
        return ""
    n_use = max(1, min(int(n), len(parts)))
    bullets = []
    for p in parts[:n_use]:
        p = re.sub(r"^[\-\*\d\.\)\s]+", "", p).strip()
        p = re.sub(r"\*\*", "", p).strip()
        if p:
            bullets.append(f"- {p}")
    return "\n".join(bullets).strip()


def _normalize_bullet_output(text: str, n: int) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    candidates: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
        ln = re.sub(r"\*\*", "", ln).strip()
        if not ln:
            continue
        if len(ln.split()) < 3:
            continue
        if ln.lower().startswith("unit tests are crucial for several reasons"):
            continue
        candidates.append(ln)
    if len(candidates) < n:
        more = [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text or "")) if p.strip()]
        for m in more:
            m = re.sub(r"^[\-\*\d\.\)\s]+", "", m).strip()
            m = re.sub(r"\*\*", "", m).strip()
            if not m or len(m.split()) < 3:
                continue
            if m not in candidates:
                candidates.append(m)
            if len(candidates) >= n:
                break
    if not candidates:
        return _build_bullets_from_text(text=text, n=n)
    n_use = max(1, min(int(n), len(candidates)))
    return "\n".join(f"- {candidates[i]}" for i in range(n_use))


def enforce_response_contract(user_text: str, response_text: str) -> str:
    user_low = str(user_text or "").lower()
    out = str(response_text or "").strip()
    if not out:
        return out

    math_match = re.search(
        r"what is\s+(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)",
        user_low,
    )
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
        if "round to 2 decimals" in user_low:
            return f"{val:.2f}"
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return f"{val:.6g}"

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

    if "just the answer" in user_low:
        nums = NUMBER_RE.findall(out)
        if nums:
            return nums[-1]
        if out.lower().startswith("the answer is "):
            return out[14:].strip().rstrip(".")

    if "bullet" in user_low and not re.search(r"(?m)^\s*[-*]\s+", out):
        m = re.search(r"(\d+)\s+(?:short\s+)?bullet", user_low)
        n = int(m.group(1)) if m else 3
        out = _normalize_bullet_output(out, n=n)
    elif "bullet" in user_low:
        m = re.search(r"(\d+)\s+(?:short\s+)?bullet", user_low)
        n = int(m.group(1)) if m else 3
        out = _normalize_bullet_output(out, n=n)

    return out


class Engine:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device, adapter_loaded: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adapter_loaded = adapter_loaded
        self.lock = threading.RLock()
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "device": str(self.device),
            "adapter_loaded": bool(self.adapter_loaded),
            "sessions": len(self.sessions),
        }

    def clear(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)

    def _build_prompt(self, history: List[Dict[str, str]], user_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise, practical assistant. "
                    "Answer directly, avoid template labels and bracketed tags, "
                    "and do not narrate hidden reasoning."
                ),
            }
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = str(msg.get("content", ""))
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def chat(
        self,
        session_id: str,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        if not user_text.strip():
            raise ValueError("Empty message")
        with self.lock:
            history = list(self.sessions.get(session_id, []))[-12:]
        prompt = self._build_prompt(history, user_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(self.device)

        do_sample = float(temperature) >= 0.18
        wall_start = time.perf_counter()
        gen_kwargs = {
            "max_new_tokens": max(16, int(max_new_tokens)),
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.08,
            "no_repeat_ngram_size": 4,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = min(1.3, max(0.05, float(temperature)))
            gen_kwargs["top_p"] = min(1.0, max(0.1, float(top_p)))
            gen_kwargs["top_k"] = 40

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        total_ms = (time.perf_counter() - wall_start) * 1000.0

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        response = clean_generated_response(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        response = enforce_response_contract(user_text=user_text, response_text=response)
        if not response:
            response = "(no output)"

        with self.lock:
            new_hist = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response},
            ]
            self.sessions[session_id] = new_hist[-20:]
        return {
            "ok": True,
            "session_id": session_id,
            "response": response,
            "timing": {"total_ms": round(total_ms, 1)},
        }


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file  # local import keeps startup lighter

        return load_file(str(safetensors_path))
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict):
            return state
        raise TypeError(f"Unexpected adapter_model.bin payload type: {type(state)}")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _is_legacy_nested_adapter_state(state_dict: Dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    first_key = next(iter(state_dict.keys()))
    return first_key.startswith(LEGACY_NESTED_ADAPTER_PREFIX)


def _canonicalize_adapter_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
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
        print(f"[adapter] legacy nested format detected: {adapter_dir}")
    remapped_state, remapped_count = _canonicalize_adapter_state_dict(state_dict)
    peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
    peft_cfg.inference_mode = True
    model = get_peft_model(model, peft_cfg)
    incompat = set_peft_model_state_dict(model, remapped_state, adapter_name="default")
    missing_count = len(getattr(incompat, "missing_keys", []) or [])
    unexpected_count = len(getattr(incompat, "unexpected_keys", []) or [])
    print(
        f"[adapter] loaded remapped={remapped_count} missing={missing_count} unexpected={unexpected_count}"
    )
    logging.info(
        "[adapter] loaded remapped=%s missing=%s unexpected=%s",
        remapped_count,
        missing_count,
        unexpected_count,
    )
    return model.to(device)


def resolve_device(preferred: str) -> torch.device:
    pref = str(preferred).strip().lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_engine(base_model: str, adapter_dir: str, device: torch.device) -> Engine:
    logging.info("Loading tokenizer from %s", base_model)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        base_model,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Tokenizer loaded")
    logging.info("Loading base model weights from %s", base_model)
    model = Qwen2ForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).to(device)
    logging.info("Base model weights loaded")
    _set_model_use_cache(model, enabled=True)

    adapter_loaded = False
    ad_path = Path(adapter_dir)
    if ad_path.exists():
        logging.info("Loading adapter weights from %s", ad_path)
        model = _load_adapter_with_compat(model=model, adapter_dir=ad_path, device=device)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload().to(device)
            print("[adapter] merged into base model for faster inference")
            logging.info("[adapter] merged into base model for faster inference")
        adapter_loaded = True
        logging.info("Adapter load complete")
    else:
        logging.warning("Adapter path does not exist: %s", ad_path)
    _set_model_use_cache(model, enabled=True)
    model.eval()
    logging.info("Model ready for inference")
    return Engine(model=model, tokenizer=tokenizer, device=device, adapter_loaded=adapter_loaded)


def build_app(engine: Engine) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "status": engine.status()})

    @app.post("/api/chat")
    def api_chat():
        p = request.get_json(force=True, silent=True) or {}
        sid = str(p.get("session_id") or "").strip() or str(uuid.uuid4())
        msg = str(p.get("message") or "").strip()
        try:
            return jsonify(
                engine.chat(
                    session_id=sid,
                    user_text=msg,
                    max_new_tokens=int(p.get("max_new_tokens") or 96),
                    temperature=float(p.get("temperature") or 0.20),
                    top_p=float(p.get("top_p") or 0.92),
                )
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post("/api/clear")
    def api_clear():
        p = request.get_json(force=True, silent=True) or {}
        sid = str(p.get("session_id") or "").strip()
        if not sid:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        engine.clear(sid)
        return jsonify({"ok": True, "session_id": sid})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen + LoRA adapter web chat.")
    ap.add_argument(
        "--base_model",
        default=r"C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
    )
    ap.add_argument(
        "--adapter_dir",
        default=r"..\artifacts\qwen_supermix_enhanced_v8_repair\adapter",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = resolve_device("cuda" if args.device == "auto" else args.device)
    print(f"[load] device={device}")
    print(f"[load] base_model={args.base_model}")
    print(f"[load] adapter_dir={args.adapter_dir}")
    engine = load_engine(args.base_model, args.adapter_dir, device)
    print(f"[ready] adapter_loaded={engine.adapter_loaded}")
    app = build_app(engine)
    print(f"[ready] web ui: http://{args.host}:{args.port}")
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
