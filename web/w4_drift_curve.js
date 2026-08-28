/**
 * W4 — H3 Drift QC curve + pass/fail badge (read-only).
 * Reads structured mmx_drift JSON from UI payload.
 */
import {
  addDomWidgetLast,
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  drawLoadingSpinner,
  drawPlaceholder,
  firstUiString,
  parseJsonSafe,
  rafThrottle,
  setupDpiCanvas,
  startLoadingLoop,
  stopLoadingLoop,
  themeVar,
} from "./shared.js";

const NODE = "MiniMaxH3_DriftQC";
const ST_KEY = "_mmxW4";
const MIN_H = 130;

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:100%;position:relative;";
  const badge = document.createElement("div");
  badge.style.cssText = "position:absolute;top:6px;right:8px;padding:2px 8px;border-radius:4px;font:600 11px sans-serif;z-index:2;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;display:block;";
  box.append(badge, canvas);

  const st = {
    box,
    canvas,
    badge,
    state: "empty",
    message: "Run the node to see exterior drift.",
    series: null,
    passed: null,
    raf: 0,
    apiOff: [],
  };
  node[ST_KEY] = st;

  addDomWidgetLast(node, "mmx_drift_curve", box, (width) => {
    const w = Math.max(220, width || node.size?.[0] || 300);
    return Math.max(MIN_H, Math.round(w * 0.35));
  });

  const paint = () => paintFrame(node);
  const render = rafThrottle(paint);
  st.resizeObs = new ResizeObserver(render);
  st.resizeObs.observe(box);
  st.apiOff.push(bindExecutionLifecycle(node, st, {
    onStart: () => {
      st.state = "loading";
      startLoadingLoop(st, paint);
    },
    onAbort: (msg) => {
      stopLoadingLoop(st);
      if (st.state === "loading") {
        st.state = "error";
        st.message = msg || "Execution failed.";
        render();
      }
    },
  }));

  chainOnRemoved(node, () => disposeState(node, ST_KEY));
  render();
  return st;
}

function setBadge(st, passed) {
  if (passed === true) {
    st.badge.textContent = "PASS";
    st.badge.style.background = "rgba(34,197,94,0.25)";
    st.badge.style.color = "#86efac";
    st.badge.style.border = "1px solid rgba(34,197,94,0.5)";
  } else if (passed === false) {
    st.badge.textContent = "FAIL";
    st.badge.style.background = "rgba(239,68,68,0.25)";
    st.badge.style.color = "#fca5a5";
    st.badge.style.border = "1px solid rgba(239,68,68,0.5)";
  } else {
    st.badge.textContent = "—";
    st.badge.style.background = "rgba(100,100,100,0.2)";
    st.badge.style.color = themeVar("inputText");
    st.badge.style.border = `1px solid ${themeVar("border")}`;
  }
}

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const rect = st.box.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const ctx = setupDpiCanvas(st.canvas, w, h);
  setBadge(st, st.passed);

  if (st.state === "loading") {
    drawLoadingSpinner(ctx, w, h);
    return;
  }
  if (st.state === "error" || st.state === "empty") {
    drawPlaceholder(ctx, w, h, st.message, st.state === "error" ? "error" : "empty");
    return;
  }

  const series = st.series || [];
  if (!series.length) {
    drawPlaceholder(ctx, w, h, "No drift samples.", "error");
    return;
  }

  const pad = { l: 32, r: 10, t: 20, b: 22 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  ctx.fillStyle = themeVar("menuBg");
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = themeVar("border");
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  const maxY = Math.max(...series, 0.001);
  const xAt = (i) => pad.l + (i / Math.max(series.length - 1, 1)) * pw;
  const yAt = (v) => pad.t + ph - (v / maxY) * ph;

  ctx.beginPath();
  series.forEach((v, i) => {
    const x = xAt(i);
    if (i === 0) ctx.moveTo(x, yAt(v));
    else ctx.lineTo(x, yAt(v));
  });
  ctx.strokeStyle = themeVar("primary");
  ctx.lineWidth = 1.5;
  ctx.stroke();

  series.forEach((v, i) => {
    ctx.fillStyle = themeVar("primary");
    ctx.beginPath();
    ctx.arc(xAt(i), yAt(v), 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = themeVar("inputText");
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("frame", pad.l + pw / 2, h - 4);
  ctx.save();
  ctx.translate(10, pad.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("drift px", 0, 0);
  ctx.restore();
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  stopLoadingLoop(st);
  const raw = firstUiString(output, "mmx_drift");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.drift_px)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_drift UI payload." : "No mmx_drift in UI output.";
    st.passed = null;
    paintFrame(node);
    return;
  }
  st.series = data.drift_px.map(Number);
  st.passed = Boolean(data.passed);
  st.state = "success";
  st.message = "";
  paintFrame(node);
}

app.registerExtension({
  name: "MiniMaxH3.W4DriftCurve",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const _created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = _created?.apply(this, arguments);
      buildDom(this);
      return r;
    };
    const _executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = _executed?.apply(this, arguments);
      try { handleExecuted(this, output || {}); } catch (_) { /* never break graph */ }
      return r;
    };
  },
});
