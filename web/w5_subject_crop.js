/**
 * W5 — Subject Crop plan inspector (interactive).
 *
 * WHY THIS EXISTS: SubjectCrop decides a constant-size box and a piecewise
 * path for it. Whether that plan is good is a VISUAL question -- does the box
 * actually sit still, does it clip the subject, how often does it jump -- and
 * none of that is answerable from a text report. Socket values never reach the
 * browser (only NodeOutput.ui does), so the node serialises its plan into
 * `mmx_crop_plan` and this widget draws it.
 *
 * Interactive, not a static plot:
 *   - drag / arrow-key the frame scrubber and watch the box move
 *   - the timeline marks every frame where the box JUMPED; click a mark to go
 *     there, because the jumps are the only frames worth inspecting
 *   - Play steps through frames so a "still" plan can be judged as motion
 *   - the box is drawn to scale inside the frame, with the crop size readout,
 *     so an over-tight plan is obvious before you run the rest of the graph
 */
import {
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  firstUiString,
  parseJsonSafe,
} from "./shared.js";
import {
  mountPanel,
  button,
  emptyState,
  installZoomRepaint,
  canvasBackingScale,
} from "./c2c_ui/index.js";

const NODES = new Set(["MiniMaxH3_SubjectCrop", "MiniMaxH3_SubjectCropAdvanced"]);
const ST_KEY = "_mmxW5";
const MIN_H = 260;
const NODE_MIN_W = 380;
const BAR_H = 26;
const TIMELINE_H = 18;

function resolveCu(root, token, fallback) {
  try {
    const prop = token.startsWith("--") ? token : `--${token}`;
    const v = getComputedStyle(root).getPropertyValue(prop).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}

/** Shared between legend chips and canvas — same resolved colours, matched opacity. */
function planPalette(root) {
  const fg = resolveCu(root, "--cu-ink", "#e8e6f7");
  const accent = resolveCu(root, "--cu-accent", "#b494ff");
  return {
    frame: fg,
    allBoxes: accent,
    currentBox: accent,
    jump: accent,
    playhead: fg,
    allBoxesAlpha: 0.12,
    frameAlpha: 0.35,
  };
}

function syncLegendColours(st, pal) {
  if (!st.legendDots) return;
  st.legendDots.current.style.background = pal.currentBox;
  st.legendDots.current.style.opacity = "1";
  st.legendDots.all.style.background = pal.allBoxes;
  st.legendDots.all.style.opacity = String(Math.max(0.35, pal.allBoxesAlpha));
  st.legendDots.jump.style.background = pal.jump;
  st.legendDots.jump.style.opacity = "1";
}

function setupCanvas(canvas, cssW, cssH) {
  const scale = canvasBackingScale(cssW, cssH);
  const bw = Math.max(1, Math.round(cssW * scale));
  const bh = Math.max(1, Math.round(cssH * scale));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return ctx;
}

function setPlaying(st, on) {
  st.playing = on;
  st.playBtn.querySelector("span:last-child").textContent = on ? "Pause" : "Play";
  if (st.timer) {
    clearInterval(st.timer);
    st.timer = 0;
  }
  if (!on || !st.plan) return;
  st.timer = setInterval(() => {
    const n = st.plan.boxes.length;
    if (st.frame >= n - 1) {
      setPlaying(st, false);
      return;
    }
    st.frame += 1;
    st.slider.value = String(st.frame);
    st.paint();
  }, 1000 / 12);
}

function updateHeader(st) {
  if (!st.plan?.boxes?.length) {
    st.summary.textContent = "Run the node to see its crop plan";
    st.pill.textContent = "—";
    return;
  }
  const plan = st.plan;
  const n = plan.boxes.length;
  const b = plan.boxes[st.frame];
  st.pill.textContent = "OK";
  st.pill.classList.add("c2c-ui-chart__pill--ok");
  st.summary.textContent =
    `${plan.mode} · box ${b.w}x${b.h} · out ${plan.out_w}x${plan.out_h} · ` +
    `${plan.jumps.length} jump${plan.jumps.length === 1 ? "" : "s"} / ${n}f`;
}

function draw(st) {
  const cssW = Math.max(1, st.canvasWrap.clientWidth || 300);
  const canvasH = Math.max(80, (st.canvasWrap.clientHeight || MIN_H) - TIMELINE_H);
  const ctx = setupCanvas(st.canvas, cssW, canvasH);

  if (!st.plan || !st.plan.boxes || !st.plan.boxes.length) {
    st.overlay.hidden = false;
    st.overlay.innerHTML = "";
    st.overlay.appendChild(emptyState({
      title: "Crop plan",
      hint: "Run the node to see its crop plan",
    }));
    ctx.clearRect(0, 0, cssW, canvasH);
    st.readout.textContent = "";
    updateHeader(st);
    return;
  }

  st.overlay.hidden = true;
  const plan = st.plan;
  const n = plan.boxes.length;
  const f = Math.max(0, Math.min(n - 1, st.frame));
  const b = plan.boxes[f];
  const plotH = canvasH - 6;

  const pal = planPalette(st.root);
  const { frame: fg, currentBox: accent } = pal;
  syncLegendColours(st, pal);

  const fw = plan.frame_w || 1;
  const fh = plan.frame_h || 1;
  const scale = Math.min((cssW - 16) / fw, (plotH - 16) / fh);
  const ox = (cssW - fw * scale) / 2;
  const oy = (plotH - fh * scale) / 2;

  ctx.clearRect(0, 0, cssW, canvasH);

  ctx.strokeStyle = fg;
  ctx.globalAlpha = pal.frameAlpha;
  ctx.lineWidth = 1;
  ctx.strokeRect(ox, oy, fw * scale, fh * scale);
  ctx.globalAlpha = 1;

  ctx.strokeStyle = pal.allBoxes;
  ctx.globalAlpha = pal.allBoxesAlpha;
  for (const q of plan.boxes) {
    ctx.strokeRect(ox + q.x * scale, oy + q.y * scale, q.w * scale, q.h * scale);
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = pal.currentBox;
  ctx.lineWidth = 2;
  ctx.strokeRect(ox + b.x * scale, oy + b.y * scale, b.w * scale, b.h * scale);

  const cx = ox + (b.x + b.w / 2) * scale;
  const cy = oy + (b.y + b.h / 2) * scale;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
  ctx.stroke();
  ctx.globalAlpha = 1;

  const ty = canvasH + 4;
  const tlCtx = setupCanvas(st.timeline, cssW, TIMELINE_H);
  tlCtx.clearRect(0, 0, cssW, TIMELINE_H);
  tlCtx.globalAlpha = 0.25;
  tlCtx.fillStyle = fg;
  tlCtx.fillRect(0, 4, cssW, 2);
  tlCtx.globalAlpha = 1;
  tlCtx.fillStyle = pal.jump;
  for (const j of plan.jumps) {
    const x = (j / Math.max(1, n - 1)) * cssW;
    tlCtx.fillRect(x - 1, 0, 2, 10);
  }
  const px = (f / Math.max(1, n - 1)) * cssW;
  tlCtx.fillStyle = pal.playhead;
  tlCtx.fillRect(px - 1, 0, 2, 14);

  st.readout.textContent = `f ${f}/${n - 1}  ${b.x},${b.y}`;
  updateHeader(st);
}

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];

  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "4px";
  root.style.width = "100%";
  root.style.height = "100%";

  const header = document.createElement("div");
  header.className = "c2c-ui-chart__header";
  const pill = document.createElement("span");
  pill.className = "c2c-ui-chart__pill";
  pill.textContent = "—";
  const summary = document.createElement("span");
  summary.className = "c2c-ui-chart__header-text";
  header.appendChild(pill);
  header.appendChild(summary);

  const legend = document.createElement("div");
  legend.className = "c2c-ui-chart__chips";
  const legendDots = {};
  for (const [key, label] of [
    ["current", "current box"],
    ["all", "all boxes"],
    ["jump", "jump frame"],
  ]) {
    const chip = document.createElement("span");
    chip.className = "c2c-ui-chart__chip";
    chip.style.pointerEvents = "none";
    const dot = document.createElement("span");
    dot.className = "c2c-ui-chart__chip-dot";
    chip.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.textContent = label;
    chip.appendChild(lbl);
    legend.appendChild(chip);
    legendDots[key] = dot;
  }

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;flex:1 1 auto;min-height:80px;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:calc(100% - 18px);";
  canvas.tabIndex = 0;
  const timeline = document.createElement("canvas");
  timeline.style.cssText = `display:block;width:100%;height:${TIMELINE_H}px;cursor:pointer;`;
  const overlay = document.createElement("div");
  overlay.className = "c2c-ui-chart__overlay";
  overlay.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;";
  canvasWrap.appendChild(canvas);
  canvasWrap.appendChild(overlay);
  canvasWrap.appendChild(timeline);

  const bar = document.createElement("div");
  bar.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:6px;flex-wrap:wrap;";

  const playBtn = button("Play", {
    onClick: () => setPlaying(st, !st.playing),
  });
  playBtn.title =
    "Step through the plan. A plan that claims 'stillness' should look still here.";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "0";
  slider.step = "1";
  slider.value = "0";
  slider.title = "Scrub the planned frames. Jump frames are ticked on the timeline.";
  slider.style.cssText = "flex:1 1 auto;min-width:40px;";

  const readout = document.createElement("span");
  readout.className = "c2c-ui-status";
  readout.style.fontVariantNumeric = "tabular-nums";

  bar.append(playBtn, slider, readout);
  root.append(header, legend, canvasWrap, bar);

  const st = {
    root, canvasWrap, canvas, timeline, overlay, bar,
    playBtn, slider, readout, pill, summary, legendDots,
    plan: null, frame: 0, playing: false, timer: 0,
    zoomOff: null, resizeObs: null,
  };
  node[ST_KEY] = st;

  st.paint = () => draw(st);
  syncLegendColours(st, planPalette(root));

  const panelWidget = mountPanel(node, "mmx_crop_plan_view", root, { minHeight: MIN_H });
  panelWidget.computeSize = (w) => [w, node._mmxW5Collapsed ? 0 : MIN_H + 8];
  panelWidget.onPanelResize = () => st.paint();
  st.zoomOff = installZoomRepaint(node, () => st.paint(), "_c2cW5Zoom");

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => st.paint());
    st.resizeObs.observe(canvasWrap);
  }

  slider.addEventListener("input", () => {
    st.frame = Number(slider.value) || 0;
    st.paint();
  });
  canvas.addEventListener("keydown", (e) => {
    if (!st.plan) return;
    const n = st.plan.boxes.length;
    if (e.key === "ArrowRight") st.frame = Math.min(n - 1, st.frame + 1);
    else if (e.key === "ArrowLeft") st.frame = Math.max(0, st.frame - 1);
    else return;
    e.preventDefault();
    slider.value = String(st.frame);
    st.paint();
  });
  timeline.addEventListener("click", (e) => {
    if (!st.plan) return;
    const r = timeline.getBoundingClientRect();
    const n = st.plan.boxes.length;
    const f = Math.round(((e.clientX - r.left) / Math.max(1, r.width)) * (n - 1));
    st.frame = Math.max(0, Math.min(n - 1, f));
    slider.value = String(st.frame);
    st.paint();
  });

  return st;
}

function applyUi(node, st, output) {
  const raw = firstUiString(output, "mmx_crop_plan");
  const plan = raw ? parseJsonSafe(raw) : null;
  if (!plan || !plan.boxes) return;
  setPlaying(st, false);
  st.plan = plan;
  st.frame = 0;
  st.slider.min = "0";
  st.slider.max = String(Math.max(0, plan.boxes.length - 1));
  st.slider.value = "0";
  st.paint();
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "MiniMaxH3.W5.SubjectCropPlan",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData.name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      const st = buildDom(this);
      bindExecutionLifecycle(this, st, {
        onStart: () => { st.plan = null; setPlaying(st, false); st.paint(); },
        onAbort: () => { setPlaying(st, false); },
      });
      chainOnRemoved(this, () => {
        setPlaying(st, false);
        try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
        try { st.resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
        disposeState(this, ST_KEY);
      });
      st.paint();
      return r;
    };

    const onExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = onExec?.apply(this, arguments);
      const st = this[ST_KEY];
      if (st) { try { applyUi(this, st, output); } catch (e) { console.error("[MMX W5]", e); } }
      return r;
    };
  },
});
