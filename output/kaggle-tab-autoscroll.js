/*
Supermix Kaggle tab auto-scroll helper.

Usage in the Kaggle notebook tab:
1. Open DevTools console.
2. Paste the full file contents and press Enter.
3. To stop it later, run:
   window.__supermixKaggleAutoScrollStop?.();

This only scrolls the current page. It does not move the mouse or take over
the rest of the computer. It is a convenience hack, not a reliable way to
defeat Kaggle timeout policy.
*/

(() => {
  if (typeof window === "undefined" || typeof document === "undefined") {
    throw new Error("Browser window not available.");
  }

  if (typeof window.__supermixKaggleAutoScrollStop === "function") {
    window.__supermixKaggleAutoScrollStop();
  }

  const state = {
    active: true,
    direction: 1,
    timer: null,
    pauseUntil: 0,
    tickBaseMs: 2600,
    tickJitterMs: 700,
    manualPauseMs: 20000,
    edgeFlipPx: 220,
    minStepPx: 26,
    maxStepPx: 120,
    target: null,
  };

  const interactionEvents = ["wheel", "touchstart", "keydown", "mousedown"];

  function log(message) {
    console.log(`[supermix-kaggle-autoscroll] ${message}`);
  }

  function root() {
    return document.scrollingElement || document.documentElement || document.body;
  }

  function isScrollable(el) {
    if (!el || typeof el.scrollTop !== "number") {
      return false;
    }
    const style = window.getComputedStyle(el);
    const overflowY = style.overflowY;
    const allowsScroll =
      overflowY === "auto" ||
      overflowY === "scroll" ||
      overflowY === "overlay" ||
      el === document.scrollingElement ||
      el === document.documentElement ||
      el === document.body;
    return allowsScroll && el.scrollHeight - el.clientHeight > 80;
  }

  function scrollableCandidates() {
    const candidates = [root()];
    for (const el of document.querySelectorAll("main, section, div, [role='main']")) {
      candidates.push(el);
    }
    return candidates.filter(isScrollable);
  }

  function pickTarget() {
    const candidates = scrollableCandidates();
    if (!candidates.length) {
      return root();
    }
    candidates.sort((a, b) => {
      const aRange = Math.max(0, a.scrollHeight - a.clientHeight);
      const bRange = Math.max(0, b.scrollHeight - b.clientHeight);
      const aArea = a.clientHeight * a.clientWidth;
      const bArea = b.clientHeight * b.clientWidth;
      return bRange - aRange || bArea - aArea;
    });
    return candidates[0];
  }

  function clamp(value, minValue, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
  }

  function onManualInteraction() {
    state.pauseUntil = Date.now() + state.manualPauseMs;
  }

  function nextDelay() {
    const jitter = Math.round((Math.random() * 2 - 1) * state.tickJitterMs);
    return Math.max(1200, state.tickBaseMs + jitter);
  }

  function schedule() {
    if (!state.active) {
      return;
    }
    clearTimeout(state.timer);
    state.timer = window.setTimeout(tick, nextDelay());
  }

  function tick() {
    if (!state.active) {
      return;
    }

    if (document.hidden || Date.now() < state.pauseUntil) {
      schedule();
      return;
    }

    const el = pickTarget();
    if (!el) {
      schedule();
      return;
    }
    state.target = el;

    const viewportHeight = Math.max(1, el.clientHeight || window.innerHeight || 1);
    const maxScrollTop = Math.max(0, el.scrollHeight - viewportHeight);
    if (maxScrollTop <= 0) {
      schedule();
      return;
    }

    const currentTop = el.scrollTop;
    const distanceToEdge = state.direction > 0 ? (maxScrollTop - currentTop) : currentTop;
    if (distanceToEdge <= state.edgeFlipPx) {
      state.direction *= -1;
    }

    const scaledStep = Math.round(Math.max(state.minStepPx, Math.min(state.maxStepPx, distanceToEdge * 0.18)));
    const nextTop = clamp(currentTop + state.direction * scaledStep, 0, maxScrollTop);
    if (typeof el.scrollTo === "function") {
      el.scrollTo({ top: nextTop, behavior: "smooth" });
    } else {
      el.scrollTop = nextTop;
    }

    schedule();
  }

  function stop() {
    if (!state.active) {
      return;
    }
    state.active = false;
    clearTimeout(state.timer);
    for (const eventName of interactionEvents) {
      window.removeEventListener(eventName, onManualInteraction, { passive: true });
    }
    delete window.__supermixKaggleAutoScrollStop;
    log("stopped");
  }

  for (const eventName of interactionEvents) {
    window.addEventListener(eventName, onManualInteraction, { passive: true });
  }

  window.__supermixKaggleAutoScrollStop = stop;
  window.__supermixKaggleAutoScrollState = state;
  log("started");
  log("this is a convenience hack, not a guaranteed anti-timeout solution");
  tick();
  schedule();
})();
