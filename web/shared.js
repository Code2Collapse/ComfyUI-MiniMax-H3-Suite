/**
 * MiniMax H3 Suite — shared DOM-widget helpers.
 * Theme vars read from comfyui_frontend_package 1.45.21:
 *   groupNode-CZraG87T.css — --bg-color, --fg-color, --comfy-menu-bg,
 *   --comfy-input-bg, --border-color, --input-text
 *   GraphView-hWkD98Et.css — --p-primary-color, --text-primary
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const THEME_VARS = {
  bg: ["--bg-color", "#1a1a1a"],
  fg: ["--fg-color", "#ddd"],
  menuBg: ["--comfy-menu-bg", "#353535"],
  inputBg: ["--comfy-input-bg", "#222"],
  border: ["--border-color", "#4a4a4a"],
  inputText: ["--input-text", "#ccc"],
  primary: ["--p-primary-color", "#4cc3ff"],
  textPrimary: ["--text-primary", "#e8e8e8"],
};

export function themeVar(name) {
  const [key, fallback] = THEME_VARS[name] || ["--fg-color", "#ddd"];
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(key).trim();
  return v || fallback;
}

export function rafThrottle(fn) {
  let raf = 0;
  let lastArgs = null;
  const run = () => {
    raf = 0;
    const args = lastArgs;
    lastArgs = null;
    if (args) fn(...args);
  };
  return (...args) => {
    lastArgs = args;
    if (!raf) raf = requestAnimationFrame(run);
  };
}

export function cancelRaf(handle) {
  if (handle) cancelAnimationFrame(handle);
}

/** Chain onRemoved — F-INV4: never replace, always wrap. */
export function chainOnRemoved(node, cleanup) {
  const orig = node.onRemoved;
  node.onRemoved = function (...a) {
    try { cleanup?.(); } catch (_) { /* never break graph */ }
    return orig?.apply(this, a);
  };
}

export function disposeState(node, key) {
  const st = node[key];
  if (!st) return;
  stopLoadingLoop(st);
  if (st.resizeObs) {
    try { st.resizeObs.disconnect(); } catch (_) {}
  }
  if (st.apiOff) {
    for (const off of st.apiOff) {
      try { off(); } catch (_) {}
    }
  }
  delete node[key];
}

export function setupDpiCanvas(canvas, cssW, cssH) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.floor(cssW * dpr));
  const h = Math.max(1, Math.floor(cssH * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function widgetByName(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

/**
 * Append DOM widget LAST — built-in schema widgets are already present when JS runs.
 * Never call addWidget after addDOMWidget (Comfy-Org/ComfyUI_frontend#7942).
 */
export function addDomWidgetLast(node, id, el, computeHeight) {
  const w = node.addDOMWidget(id, "div", el, { serialize: false });
  w.computeSize = (width) => [0, computeHeight(width)];
  return w;
}

export function bindExecutionLifecycle(node, st, { onStart, onAbort }) {
  const offs = [];
  const add = (type, fn) => {
    api.addEventListener(type, fn);
    offs.push(() => api.removeEventListener(type, fn));
  };

  add("executing", (e) => {
    const d = e.detail || {};
    if (d.node != null && String(d.node) === String(node.id)) {
      onStart?.();
      return;
    }
    if (d.node == null && st.state === "loading") {
      onAbort?.("Execution finished without a UI update for this node.");
    }
  });

  add("execution_error", (e) => {
    const d = e.detail || {};
    const nid = d.node_id ?? d.node;
    if (nid != null && String(nid) !== String(node.id)) return;
    onAbort?.(d.exception_message || d.exception_type || "Execution error.");
  });

  add("execution_interrupted", () => {
    if (st.state === "loading") {
      onAbort?.("Execution interrupted.");
    }
  });

  return () => { for (const off of offs) off(); };
}

export function startLoadingLoop(st, paintFn) {
  stopLoadingLoop(st);
  const tick = () => {
    if (st.state !== "loading") {
      st.raf = 0;
      return;
    }
    paintFn();
    st.raf = requestAnimationFrame(tick);
  };
  st.raf = requestAnimationFrame(tick);
}

export function stopLoadingLoop(st) {
  cancelRaf(st.raf);
  st.raf = 0;
}

export function firstUiString(output, key) {
  const v = output?.[key];
  if (Array.isArray(v)) return v[0];
  return v;
}

export function imageUrlFromUi(ui, key = "images") {
  return maskFromOutput(ui, key);
}

export function maskFromOutput(output, key) {
  const v = output?.[key];
  const meta = Array.isArray(v) ? v[0] : v;
  if (!meta) return null;
  if (meta.filename) {
    const params = new URLSearchParams({
      filename: meta.filename,
      subfolder: meta.subfolder || "",
      type: meta.type || "temp",
      rand: String(Date.now()),
    });
    return api.apiURL(`/view?${params}`);
  }
  return null;
}

export function parseJsonSafe(text) {
  if (text == null) return null;
  const s = String(text).trim();
  if (!s) return null;
  try { return JSON.parse(s); } catch (_) { return null; }
}

export function drawPlaceholder(ctx, w, h, message, state = "empty") {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = themeVar("menuBg");
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = themeVar("border");
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.fillStyle = state === "error" ? "#f87171" : themeVar("inputText");
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const lines = String(message || "").split("\n");
  const lh = 16;
  const y0 = h / 2 - ((lines.length - 1) * lh) / 2;
  lines.forEach((ln, i) => ctx.fillText(ln, w / 2, y0 + i * lh));
}

export function drawLoadingSpinner(ctx, w, h) {
  drawPlaceholder(ctx, w, h, "Running…", "loading");
  const t = Date.now() / 1000;
  const cx = w / 2;
  const cy = h / 2 - 20;
  const r = 10;
  ctx.strokeStyle = themeVar("primary");
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, t * 6, t * 6 + Math.PI * 1.2);
  ctx.stroke();
}

export { app, api };
