import argparse
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from flask import Flask, jsonify, request, send_from_directory

import chat_app
from chat_web_app import Engine as ChatEngine
from device_utils import configure_torch_runtime, resolve_device


DEFAULT_IMAGE_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, watermark, signature, "
    "extra fingers, malformed hands, duplicate subjects, disfigured face"
)
PROMPT_STYLE_SUFFIXES = {
    "auto": "",
    "photo": "photorealistic, natural lighting, high detail, sharp focus",
    "cinematic": "cinematic composition, dramatic lighting, rich atmosphere, high detail",
    "illustration": "detailed illustration, clean shapes, polished color palette",
    "anime": "anime style, expressive character design, crisp line art, vibrant color",
}
SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")
PROMPT_TOKEN_RE = re.compile(r"[a-z0-9']+")
PROMPT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


HTML = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Champion Image Variant</title>
<style>
body{margin:0;background:#08111d;color:#e7eef8;font-family:"Segoe UI Variable Text","Segoe UI",sans-serif}
.wrap{max-width:1360px;margin:18px auto;padding:0 14px;display:grid;grid-template-columns:380px 1fr;gap:16px}
.panel{background:linear-gradient(180deg,rgba(16,29,45,.96),rgba(8,17,29,.98));border:1px solid rgba(145,173,214,.16);border-radius:22px;box-shadow:0 24px 60px rgba(0,0,0,.28)}
.side{padding:18px;display:grid;gap:14px;align-content:start}
.hero{padding:18px;border-radius:18px;border:1px solid rgba(104,180,255,.18);background:linear-gradient(145deg,rgba(27,58,93,.92),rgba(10,20,33,.95))}
.hero h1{margin:0;font-size:30px;line-height:1.04;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
.hero p{margin:10px 0 0;color:#b9cbe1;line-height:1.55}
.card{padding:15px;border-radius:18px;border:1px solid rgba(145,173,214,.12);background:rgba(255,255,255,.03)}
.card h2{margin:0 0 12px;font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9fb5d2}
.field{display:grid;gap:6px;margin-bottom:10px}
.field label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#9fb5d2;font-weight:700}
input,select,textarea{width:100%;box-sizing:border-box;background:#07111b;color:#e7eef8;border:1px solid rgba(145,173,214,.18);border-radius:12px;padding:11px 12px;font:inherit}
textarea{resize:vertical;min-height:92px}
input:focus,select:focus,textarea:focus{outline:none;border-color:rgba(104,180,255,.52);box-shadow:0 0 0 3px rgba(104,180,255,.12)}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{border:0;border-radius:12px;padding:10px 13px;font-weight:700;cursor:pointer}
.primary{background:linear-gradient(135deg,#2c74c7,#69b4ff);color:white}
.secondary{background:#22324b;color:#e7eef8}
.status{white-space:pre-wrap;background:#07111b;border:1px solid rgba(145,173,214,.16);border-radius:12px;padding:12px;min-height:88px;color:#c0d0e4;font-family:Consolas,"Cascadia Code",monospace;font-size:12px;line-height:1.45}
.main{display:grid;grid-template-rows:auto auto 1fr;min-height:84vh}
.head{padding:18px 22px;border-bottom:1px solid rgba(145,173,214,.14);display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.head h2{margin:0;font-size:24px;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
.head p{margin:8px 0 0;color:#a9bfd9;line-height:1.55}
.pill{display:inline-flex;align-items:center;padding:8px 11px;border-radius:999px;border:1px solid rgba(145,173,214,.16);background:rgba(255,255,255,.04);font-size:12px;color:#d7e5f6}
.composer{padding:18px 22px;border-bottom:1px solid rgba(145,173,214,.12);display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end}
.composer textarea{min-height:100px}
.note{color:#9fb5d2;font-size:12px;line-height:1.45}
.content{padding:22px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px}
.canvas{border-radius:20px;border:1px solid rgba(145,173,214,.14);background:rgba(255,255,255,.02);padding:16px;min-height:560px;display:grid;place-items:center}
.canvas.empty{color:#9fb5d2;text-align:center;line-height:1.6}
.canvas img{max-width:100%;max-height:760px;display:block;border-radius:16px;box-shadow:0 22px 56px rgba(0,0,0,.35)}
.meta{display:grid;gap:12px}
.meta-card{border-radius:18px;border:1px solid rgba(145,173,214,.12);background:rgba(255,255,255,.03);padding:14px}
.meta-card h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#9fb5d2}
.meta-card pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:Consolas,"Cascadia Code",monospace;font-size:12px;line-height:1.5;color:#d4e2f3}
.history{display:grid;gap:10px}
.thumb{display:grid;gap:8px;padding:10px;border-radius:14px;background:#07111b;border:1px solid rgba(145,173,214,.12);cursor:pointer}
.thumb img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:12px}
.thumb .caption{font-size:12px;color:#c5d4e6;line-height:1.45}
@media (max-width:1080px){.wrap{grid-template-columns:1fr}.main{min-height:auto}.content{grid-template-columns:1fr}}
</style></head><body>
<div class='wrap'>
  <aside class='panel side'>
    <section class='hero'>
      <div style='font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#69b4ff;font-weight:700;margin-bottom:10px'>Current model + image renderer</div>
      <h1>Champion Image Variant</h1>
      <p>This variant keeps your current text model in the loop by using it to rewrite image prompts, then renders the final image with a local diffusion pipeline.</p>
    </section>
    <section class='card'>
      <h2>Text Model</h2>
      <div class='field'><label>Weights</label><input id='weights'></div>
      <div class='field'><label>Metadata</label><input id='meta'></div>
      <div class='btns'>
        <button class='primary' id='loadBtn'>Load Text Model</button>
        <button class='secondary' id='statusBtn'>Refresh</button>
      </div>
    </section>
    <section class='card'>
      <h2>Image Settings</h2>
      <div class='field'><label>Image Model</label><input id='imageModel'></div>
      <div class='field'><label>Style</label><select id='style'><option value='auto'>Auto</option><option value='photo'>Photo</option><option value='cinematic'>Cinematic</option><option value='illustration'>Illustration</option><option value='anime'>Anime</option></select></div>
      <div class='field'><label>Negative Prompt</label><textarea id='negativePrompt'></textarea></div>
      <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px'>
        <div class='field'><label>Width</label><input id='width' type='number' min='256' max='1024' step='64' value='512'></div>
        <div class='field'><label>Height</label><input id='height' type='number' min='256' max='1024' step='64' value='512'></div>
        <div class='field'><label>Steps</label><input id='steps' type='number' min='1' max='4' step='1' value='2'></div>
      </div>
      <div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px'>
        <div class='field'><label>Seed</label><input id='seed' type='number' min='0' step='1' value='48'></div>
        <div class='field'><label>Guidance</label><input id='guidance' type='number' min='0' max='12' step='0.1' value='0.0'></div>
      </div>
      <div class='field'><label><input id='useRefiner' type='checkbox' checked style='width:auto;margin-right:8px'>Use current text model to refine prompts</label></div>
    </section>
    <section class='card'>
      <h2>Status</h2>
      <div class='status' id='statusBox'>Loading status...</div>
    </section>
  </aside>
  <main class='panel main'>
    <header class='head'>
      <div>
        <h2>Generate Image</h2>
        <p>Type what you want, then this variant will optionally rewrite the prompt through the loaded Champion model before rendering with SDXL Turbo.</p>
      </div>
      <div class='pill' id='runtimeBadge'>runtime pending</div>
    </header>
    <section class='composer'>
      <div>
        <textarea id='prompt' placeholder='Example: A rain-soaked neon alley with a silver fox detective in a trench coat, cinematic, moody, detailed.'></textarea>
        <div class='note'>SDXL Turbo works best around 512x512 with guidance scale 0.0 and 1-4 inference steps.</div>
      </div>
      <div style='display:grid;gap:8px;align-content:end'>
        <button class='primary' id='generateBtn'>Generate</button>
        <button class='secondary' id='clearBtn'>Clear Preview</button>
      </div>
    </section>
    <section class='content'>
      <div class='canvas empty' id='canvas'>No image yet. Load the text model, then generate one.</div>
      <div class='meta'>
        <div class='meta-card'>
          <h3>Prompt Details</h3>
          <pre id='promptMeta'>Waiting for first render...</pre>
        </div>
        <div class='meta-card'>
          <h3>Recent Images</h3>
          <div class='history' id='history'></div>
        </div>
      </div>
    </section>
  </main>
</div>
<script>
const el=(id)=>document.getElementById(id);
function escapeHtml(s){return (s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}
async function jget(path){const r=await fetch(path); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); return d;}
async function jpost(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload||{})}); const d=await r.json(); if(!r.ok||d.ok===false) throw new Error(d.error||`HTTP ${r.status}`); return d;}
function renderPreview(data){
  const canvas=el('canvas');
  canvas.classList.remove('empty');
  canvas.innerHTML=`<img src="${data.image_url}" alt="generated image">`;
  el('promptMeta').textContent=JSON.stringify({
    prompt_used:data.prompt_used,
    refined_prompt:data.refined_prompt,
    output_path:data.output_path,
    image_model:data.image_model,
    timing_ms:data.timing_ms,
  }, null, 2);
}
function addHistory(item){
  const box=document.createElement('button');
  box.className='thumb';
  box.innerHTML=`<img src="${item.image_url}" alt=""><div class='caption'>${escapeHtml(item.prompt_used || item.original_prompt || '')}</div>`;
  box.onclick=()=>renderPreview(item);
  const history=el('history');
  history.prepend(box);
  while(history.children.length>6){history.removeChild(history.lastChild);}
}
async function refresh(){
  try{
    const d=await jget('/api/status');
    el('statusBox').textContent=JSON.stringify(d.status,null,2);
    el('runtimeBadge').textContent=d.status.image_pipeline_loaded ? `image ready: ${d.status.image_model_id}` : 'image pipeline idle';
    if(!el('weights').value && d.status.text.weights) el('weights').value=d.status.text.weights;
    if(!el('meta').value && d.status.text.meta) el('meta').value=d.status.text.meta;
    if(!el('imageModel').value) el('imageModel').value=d.status.image_model_id || '';
    if(!el('negativePrompt').value) el('negativePrompt').value=d.status.default_negative_prompt || '';
    if(Array.isArray(d.status.recent_images)){
      el('history').innerHTML='';
      d.status.recent_images.forEach(addHistory);
    }
  }catch(e){
    el('statusBox').textContent='Status error: '+e.message;
  }
}
async function loadModel(){
  el('statusBox').textContent='Loading text model...';
  try{
    const d=await jpost('/api/load_text',{weights:el('weights').value.trim(),meta:el('meta').value.trim()});
    el('statusBox').textContent='Loaded text model.\\n'+JSON.stringify(d,null,2);
    refresh();
  }catch(e){
    el('statusBox').textContent='Load error: '+e.message;
  }
}
async function generate(){
  const prompt=el('prompt').value.trim();
  if(!prompt) return;
  el('generateBtn').disabled=true;
  el('generateBtn').textContent='Generating...';
  try{
    const d=await jpost('/api/generate_image',{
      prompt,
      image_model:el('imageModel').value.trim(),
      negative_prompt:el('negativePrompt').value,
      style:el('style').value,
      width:Number(el('width').value),
      height:Number(el('height').value),
      steps:Number(el('steps').value),
      seed:el('seed').value === '' ? null : Number(el('seed').value),
      guidance_scale:Number(el('guidance').value),
      use_text_refiner:el('useRefiner').checked,
    });
    renderPreview(d);
    addHistory(d);
    refresh();
  }catch(e){
    el('promptMeta').textContent='Generation error: '+e.message;
  }finally{
    el('generateBtn').disabled=false;
    el('generateBtn').textContent='Generate';
  }
}
el('loadBtn').onclick=loadModel;
el('statusBtn').onclick=refresh;
el('generateBtn').onclick=generate;
el('clearBtn').onclick=()=>{el('canvas').className='canvas empty'; el('canvas').textContent='Preview cleared.';};
el('prompt').addEventListener('keydown', e=>{if(e.key==='Enter' && (e.ctrlKey || e.metaKey)){e.preventDefault(); generate();}});
refresh();
</script></body></html>"""


def _coerce_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _slugify(text: str) -> str:
    cooked = SAFE_NAME_RE.sub("-", _coerce_text(text).lower()).strip("-")
    return cooked[:48] or "image"


def _cleanup_prompt(text: str) -> str:
    out = _coerce_text(text)
    if not out:
        return ""
    lowered = out.lower()
    for prefix in ("prompt:", "image prompt:", "refined prompt:", "assistant:"):
        if lowered.startswith(prefix):
            out = out[len(prefix):].strip()
            lowered = out.lower()
    return " ".join(out.split())


def _prompt_keywords(text: str) -> set[str]:
    tokens = {
        token
        for token in PROMPT_TOKEN_RE.findall(_coerce_text(text).lower())
        if len(token) >= 3 and token not in PROMPT_STOP_WORDS
    }
    return tokens


def _prompt_is_related(source: str, candidate: str) -> bool:
    source_tokens = _prompt_keywords(source)
    candidate_tokens = _prompt_keywords(candidate)
    if not source_tokens or not candidate_tokens:
        return False
    overlap = source_tokens & candidate_tokens
    return len(overlap) >= min(2, len(source_tokens)) or (len(overlap) / len(source_tokens)) >= 0.4


class ImageVariantEngine:
    def __init__(self, text_engine: ChatEngine, output_dir: Path, default_image_model: str, default_negative_prompt: str):
        self.text_engine = text_engine
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_image_model = default_image_model
        self.default_negative_prompt = default_negative_prompt
        self.lock = threading.RLock()
        self.pipeline = None
        self.pipeline_model_id = ""
        self.recent_images = []

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "text": self.text_engine.status(),
                "image_pipeline_loaded": self.pipeline is not None,
                "image_model_id": self.pipeline_model_id or self.default_image_model,
                "default_negative_prompt": self.default_negative_prompt,
                "output_dir": str(self.output_dir),
                "recent_images": list(self.recent_images),
            }

    def _image_device(self) -> str:
        device = getattr(self.text_engine, "device", torch.device("cpu"))
        if isinstance(device, torch.device):
            return "cuda" if device.type == "cuda" else "cpu"
        return "cuda" if str(device).startswith("cuda") else "cpu"

    def _ensure_pipeline(self, image_model: str):
        target_model = _coerce_text(image_model) or self.default_image_model
        with self.lock:
            if self.pipeline is not None and self.pipeline_model_id == target_model:
                return self.pipeline, self.pipeline_model_id

            os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
            import diffusers.utils as diffusers_utils
            import diffusers.utils.import_utils as diffusers_import_utils

            for module in (diffusers_utils, diffusers_import_utils):
                for attr in (
                    "is_flash_attn_available",
                    "is_flash_attn_3_available",
                    "is_aiter_available",
                    "is_sageattention_available",
                ):
                    if hasattr(module, attr):
                        setattr(module, attr, lambda *args, **kwargs: False)

            # Diffusers 0.37.x registers custom CUDA ops on import. On the
            # current RunPod image that import path trips over torch 2.4.1's
            # schema inference, so we temporarily disable those registrations
            # and force native attention instead.
            original_custom_op = torch.library.custom_op
            original_register_fake = torch.library.register_fake

            def _custom_op_no_op(name, fn=None, /, *, mutates_args=(), device_types=None, schema=None):
                def wrap(func):
                    return func

                return wrap if fn is None else fn

            def _register_fake_no_op(op, fn=None, /, *, lib=None, _stacklevel=1):
                def wrap(func):
                    return func

                return wrap if fn is None else fn

            torch.library.custom_op = _custom_op_no_op
            torch.library.register_fake = _register_fake_no_op
            try:
                from diffusers import AutoPipelineForText2Image
            finally:
                torch.library.custom_op = original_custom_op
                torch.library.register_fake = original_register_fake

            device_name = self._image_device()
            load_kwargs: Dict[str, Any] = {
                "torch_dtype": torch.float16 if device_name == "cuda" else torch.float32,
            }
            if device_name == "cuda":
                load_kwargs["variant"] = "fp16"
            try:
                pipeline = AutoPipelineForText2Image.from_pretrained(target_model, **load_kwargs)
            except Exception:
                load_kwargs.pop("variant", None)
                pipeline = AutoPipelineForText2Image.from_pretrained(target_model, **load_kwargs)

            pipeline = pipeline.to(device_name)
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            self.pipeline = pipeline
            self.pipeline_model_id = target_model
            return pipeline, target_model

    def _refine_prompt_with_text_model(self, prompt: str) -> str:
        session_id = f"image-refiner-{uuid.uuid4()}"
        try:
            result = self.text_engine.chat(
                session_id=session_id,
                user_text=(
                    "Rewrite this request as one compact text-to-image prompt. "
                    "Keep the subject, add concrete visual details, composition, lighting, "
                    "camera or art cues when helpful, and do not explain anything. "
                    f"Request: {prompt}"
                ),
                style_mode="creative",
                response_temperature=0.05,
                show_top_responses=0,
            )
            return _cleanup_prompt(str(result.get("response", "")))
        finally:
            self.text_engine.clear(session_id)

    def _build_final_prompt(self, original_prompt: str, style: str, use_text_refiner: bool) -> Dict[str, str]:
        base = _cleanup_prompt(original_prompt)
        refined = ""
        if use_text_refiner and self.text_engine.status().get("loaded"):
            try:
                candidate = self._refine_prompt_with_text_model(base)
                if _prompt_is_related(base, candidate):
                    refined = candidate
            except Exception:
                refined = ""
        style_suffix = PROMPT_STYLE_SUFFIXES.get(style, "")
        prompt_used = refined or base
        extras = [part for part in (prompt_used, style_suffix) if part]
        final_prompt = ", ".join(extras) if extras else base
        if "high detail" not in final_prompt.lower():
            final_prompt = f"{final_prompt}, high detail, coherent composition"
        return {"base_prompt": base, "refined_prompt": refined, "prompt_used": final_prompt}

    def generate_image(
        self,
        *,
        prompt: str,
        image_model: str,
        negative_prompt: str,
        style: str,
        width: int,
        height: int,
        steps: int,
        seed: Optional[int],
        guidance_scale: float,
        use_text_refiner: bool,
    ) -> Dict[str, Any]:
        base_prompt = _coerce_text(prompt)
        if not base_prompt:
            raise ValueError("Prompt is required")

        pipeline, resolved_model = self._ensure_pipeline(image_model)
        prompt_payload = self._build_final_prompt(base_prompt, style=style, use_text_refiner=use_text_refiner)

        width = max(256, min(1024, int(width)))
        height = max(256, min(1024, int(height)))
        steps = max(1, min(4, int(steps)))
        seed_value = None if seed is None else max(0, int(seed))
        guidance_value = max(0.0, float(guidance_scale))
        negative_value = _coerce_text(negative_prompt) or self.default_negative_prompt

        if "turbo" in resolved_model.lower():
            guidance_value = 0.0
            negative_value = ""

        generator = None
        device_name = self._image_device()
        if seed_value is not None:
            generator = torch.Generator(device=device_name).manual_seed(seed_value)

        started = time.perf_counter()
        image = pipeline(
            prompt=prompt_payload["prompt_used"],
            negative_prompt=negative_value or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_value,
            generator=generator,
        ).images[0]
        total_ms = round((time.perf_counter() - started) * 1000, 1)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        basename = f"{stamp}_{_slugify(prompt_payload['base_prompt'])}_{uuid.uuid4().hex[:8]}"
        image_path = self.output_dir / f"{basename}.png"
        meta_path = self.output_dir / f"{basename}.json"
        image.save(image_path)

        metadata = {
            "original_prompt": prompt_payload["base_prompt"],
            "refined_prompt": prompt_payload["refined_prompt"],
            "prompt_used": prompt_payload["prompt_used"],
            "negative_prompt": negative_value,
            "image_model": resolved_model,
            "style": style,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed_value,
            "guidance_scale": guidance_value,
            "timing_ms": total_ms,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        result = {
            "ok": True,
            "original_prompt": prompt_payload["base_prompt"],
            "refined_prompt": prompt_payload["refined_prompt"],
            "prompt_used": prompt_payload["prompt_used"],
            "image_model": resolved_model,
            "output_path": str(image_path),
            "metadata_path": str(meta_path),
            "image_url": f"/generated/{image_path.name}",
            "timing_ms": total_ms,
        }
        with self.lock:
            self.recent_images = [result, *self.recent_images[:5]]
        return result


def build_app(hybrid_engine: ImageVariantEngine, default_weights: str, default_meta: str):
    app = Flask(__name__)

    @app.get("/")
    def index():
        html = HTML.replace("<input id='weights'></div>", f"<input id='weights' value='{default_weights}'></div>")
        html = html.replace("<input id='meta'></div>", f"<input id='meta' value='{default_meta}'></div>")
        html = html.replace("<input id='imageModel'></div>", f"<input id='imageModel' value='{hybrid_engine.default_image_model}'></div>")
        html = html.replace("<textarea id='negativePrompt'></textarea>", f"<textarea id='negativePrompt'>{hybrid_engine.default_negative_prompt}</textarea>")
        return html

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "status": hybrid_engine.status()})

    @app.post("/api/load_text")
    def api_load_text():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(hybrid_engine.text_engine.load(str(payload.get("weights") or "").strip(), str(payload.get("meta") or "").strip()))
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/generate_image")
    def api_generate_image():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(
                hybrid_engine.generate_image(
                    prompt=str(payload.get("prompt") or "").strip(),
                    image_model=str(payload.get("image_model") or "").strip(),
                    negative_prompt=str(payload.get("negative_prompt") or "").strip(),
                    style=str(payload.get("style") or "auto").strip().lower(),
                    width=int(payload.get("width") or 512),
                    height=int(payload.get("height") or 512),
                    steps=int(payload.get("steps") or 2),
                    seed=None if payload.get("seed") in (None, "") else int(payload.get("seed")),
                    guidance_scale=float(payload.get("guidance_scale") or 0.0),
                    use_text_refiner=bool(payload.get("use_text_refiner", True)),
                )
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/generated/<path:filename>")
    def generated_file(filename: str):
        return send_from_directory(hybrid_engine.output_dir, filename)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-capable Champion variant that uses the current text model as a prompt refiner.")
    parser.add_argument("--weights", default="champion_model_chat_v28_expert.pth")
    parser.add_argument("--meta", default="chat_model_meta_v28_expert.json")
    parser.add_argument("--autoload", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--torch_interop_threads", type=int, default=0)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--model_size", choices=["auto", *chat_app.VALID_RUNTIME_MODEL_SIZES], default="auto")
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--top_labels", type=int, default=3)
    parser.add_argument("--pool_mode", choices=["all", "topk"], default="all")
    parser.add_argument("--response_temperature", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--style_mode", choices=["auto", "balanced", "creative", "concise", "analyst"], default="auto")
    parser.add_argument("--creativity", type=float, default=0.2)
    parser.add_argument("--image_model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output_dir", default="generated_images")
    args = parser.parse_args()

    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    text_engine = ChatEngine(
        device,
        device_info,
        {
            "model_size": args.model_size,
            "max_turns": int(args.max_turns),
            "top_labels": int(args.top_labels),
            "pool_mode": str(args.pool_mode),
            "response_temperature": float(args.response_temperature),
            "temperature": float(args.temperature),
            "style_mode": str(args.style_mode),
            "creativity": float(args.creativity),
        },
    )
    if args.autoload:
        try:
            print(text_engine.load(args.weights, args.meta))
        except Exception as exc:
            print(f"Autoload failed: {exc}")

    hybrid_engine = ImageVariantEngine(
        text_engine=text_engine,
        output_dir=Path(args.output_dir).resolve(),
        default_image_model=str(args.image_model),
        default_negative_prompt=str(args.negative_prompt),
    )
    app = build_app(hybrid_engine, str(args.weights), str(args.meta))
    print(f"Image variant UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=int(args.port), threaded=True)


if __name__ == "__main__":
    main()
