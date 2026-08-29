/**
 * W3 — H3 Sigma Inspector live plot.
 * Reads sigma_json on execute; recomputes curve client-side when shift widgets change.
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
  widgetByName,
} from "./shared.js";

const NODE = "MiniMaxH3_SigmaInspector";
const ST_KEY = "_mmxW3";
const MIN_H = 160;

function timeShiftSigma(sigma, fromShift, toShift) {
  const s = Number(sigma);
  const base = s / (fromShift + s * (1.0 - fromShift));
  return (toShift * base) / (1.0 + (toShift - 1.0) * base);
}

function dSigmaDt(t, shift) {
  const d = 1.0 + (shift - 1.0) * t;
  return shift / (d * d);
}

function sigmaToT(sigma, shift) {
  const denom = shift + sigma * (1.0 - shift);
  return Math.abs(denom) > 1e-12 ? sigma / denom : 0.0;
}

function dsigmaADsigmaVAnalytic(sigmaV, shiftVideo, shiftAudio) {
  const t = sigmaToT(sigmaV, shiftVideo);
  const dv = dSigmaDt(t, shiftVideo);
  const da = dSigmaDt(t, shiftAudio);
  return Math.abs(dv) > 1e-12 ? da / dv : 0;
}

function recomputeSteps(sigmasV, shiftVideo, shiftAudio) {
  const sv = Number(shiftVideo);
  const sa = Number(shiftAudio);
  const steps = [];
  for (let i = 0; i < sigmasV.length; i++) {
    const sigV = Number(sigmasV[i]);
    const sigA = timeShiftSigma(sigV, sv, sa);
    const dsAnalytic = dsigmaADsigmaVAnalytic(sigV, sv, sa);
    let dsFd = 0;
    if (i + 1 < sigmasV.length) {
      const nxtV = Number(sigmasV[i + 1]);
      const nxtA = timeShiftSigma(nxtV, sv, sa);
      const dv = nxtV - sigV;
      dsFd = Math.abs(dv) > 1e-12 ? (nxtA - sigA) / dv : 0;
    }
    steps.push({
      step: i,
      sigma_v: sigV,
      sigma_a: sigA,
      dsigma_a_dsigma_v: dsAnalytic,
      dsigma_a_dsigma_v_fd: dsFd,
      carry_factor: Math.abs(sigA) > 1e-12 ? sigV / sigA : 0,
    });
  }
  return steps;
}

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:100%;position:relative;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;display:block;";
  box.appendChild(canvas);

  const st = {
    box,
    canvas,
    state: "empty",
    message: "Run the node to see the sigma curve.",
    sigmasV: null,
    steps: null,
    shiftVideo: 12,
    shiftAudio: 3,
    raf: 0,
    apiOff: [],
  };
  node[ST_KEY] = st;

  addDomWidgetLast(node, "mmx_sigma_plot", box, (width) => {
    const w = Math.max(240, width || node.size?.[0] || 320);
    return Math.max(MIN_H, Math.round(w * 0.45));
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

  const hookWidget = (name) => {
    const w = widgetByName(node, name);
    if (!w) return;
    const orig = w.callback;
    w.callback = function (v) {
      const r = orig?.apply(this, arguments);
      if (name === "shift_video") st.shiftVideo = Number(v);
      if (name === "shift_audio") st.shiftAudio = Number(v);
      if (st.sigmasV) {
        st.steps = recomputeSteps(st.sigmasV, st.shiftVideo, st.shiftAudio);
        st.state = "success";
        render();
      }
      return r;
    };
  };
  hookWidget("shift_video");
  hookWidget("shift_audio");

  chainOnRemoved(node, () => disposeState(node, ST_KEY));
  render();
  return st;
}

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const rect = st.box.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const ctx = setupDpiCanvas(st.canvas, w, h);

  if (st.state === "loading") {
    drawLoadingSpinner(ctx, w, h);
    return;
  }
  if (st.state === "error" || st.state === "empty") {
    drawPlaceholder(ctx, w, h, st.message, st.state === "error" ? "error" : "empty");
    return;
  }

  const steps = st.steps || [];
  if (!steps.length) {
    drawPlaceholder(ctx, w, h, "No sigma steps to plot.", "error");
    return;
  }

  const pad = { l: 36, r: 12, t: 14, b: 24 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  ctx.fillStyle = themeVar("menuBg");
  ctx.fillRect(0, 0, w, h);

  const maxV = Math.max(...steps.map((s) => s.sigma_v), 0.001);
  const maxA = Math.max(...steps.map((s) => s.sigma_a), 0.001);
  const maxDs = Math.max(...steps.map((s) => Math.abs(s.dsigma_a_dsigma_v)), 0.001);

  const xAt = (i) => pad.l + (i / Math.max(steps.length - 1, 1)) * pw;
  const yV = (v) => pad.t + ph - (v / maxV) * ph;
  const yA = (v) => pad.t + ph - (v / maxA) * ph;
  const yDs = (v) => pad.t + ph / 2 - (v / maxDs) * (ph / 2);

  ctx.strokeStyle = themeVar("border");
  ctx.strokeRect(pad.l, pad.t, pw, ph);

  ctx.lineWidth = 1.5;
  ctx.beginPath();
  steps.forEach((s, i) => {
    const x = xAt(i);
    if (i === 0) ctx.moveTo(x, yV(s.sigma_v));
    else ctx.lineTo(x, yV(s.sigma_v));
  });
  ctx.strokeStyle = themeVar("primary");
  ctx.stroke();

  ctx.beginPath();
  steps.forEach((s, i) => {
    const x = xAt(i);
    if (i === 0) ctx.moveTo(x, yA(s.sigma_a));
    else ctx.lineTo(x, yA(s.sigma_a));
  });
  ctx.strokeStyle = "#fbbf24";
  ctx.stroke();

  ctx.beginPath();
  steps.forEach((s, i) => {
    const x = xAt(i);
    if (i === 0) ctx.moveTo(x, yDs(s.dsigma_a_dsigma_v));
    else ctx.lineTo(x, yDs(s.dsigma_a_dsigma_v));
  });
  ctx.strokeStyle = "#a78bfa";
  ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = themeVar("inputText");
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("σv", pad.l + 4, pad.t + 10);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("σa", pad.l + 24, pad.t + 10);
  ctx.fillStyle = "#a78bfa";
  ctx.fillText("dσa/dσv (analytic)", pad.l + 44, pad.t + 10);
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  stopLoadingLoop(st);
  const raw = firstUiString(output, "mmx_sigma");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.sigmas_v)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_sigma UI payload." : "No mmx_sigma in UI output.";
    paintFrame(node);
    return;
  }
  st.sigmasV = data.sigmas_v;
  st.shiftVideo = Number(widgetByName(node, "shift_video")?.value ?? data.shift_video ?? 12);
  st.shiftAudio = Number(widgetByName(node, "shift_audio")?.value ?? data.shift_audio ?? 3);
  st.steps = recomputeSteps(st.sigmasV, st.shiftVideo, st.shiftAudio);
  st.state = "success";
  paintFrame(node);
}

app.registerExtension({
  name: "MiniMaxH3.W3SigmaPlot",
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
