/**
 * W6 — H3 Context Windows timeline (interactive).
 *
 * WHY: the whole point of this node is a TILING, and a tiling is a picture.
 * "6 windows, stride 68, overlap 22" tells you nothing about whether the
 * overlaps are where you want them, whether the tail is a stub, or how much
 * material is being rendered twice. So the plan is drawn: each window is a bar,
 * overlaps are shaded, and the cost of the overlap is stated in frames.
 *
 * Interactive: hover or click a window to pin its numbers (start, end, latent
 * rows, RoPE offset), arrow-key between windows, and toggle a "cost" view that
 * shades how many frames are generated twice.
 *
 * Socket values never reach the browser -- only NodeOutput.ui does -- so this
 * reads `mmx_context_plan` from the execution payload.
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

const NODE = "MiniMaxH3_ContextWindows";
const ST_KEY = "_mmxW6";
const MIN_H = 120;
const NODE_MIN_W = 380;
const ROW_H = 16;
const TOP = 8;
const CHROME_H = 108;
const EMPTY_H = 72;

function resolveCu(root, token, fallback) {
  try {
    const prop = token.startsWith("--") ? token : `--${token}`;
    const v = getComputedStyle(root).getPropertyValue(prop).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
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

function updateHeader(st) {
  if (!st.plan?.windows?.length) {
    st.line1.textContent = "Run the node to see the window plan";
    st.line2.textContent = "";
    st.pill.textContent = "—";
    return;
  }
  const p = st.plan;
  const wins = p.windows;
  const total = Math.max(1, p.total_frames);
  const dup = wins.reduce((a, x) => a + (x.overlap_prev || 0), 0);
  st.pill.textContent = "OK";
  st.pill.classList.add("c2c-ui-chart__pill--ok");
  st.line1.textContent =
    `${p.window_count} passes · win ${p.window_frames}f · stride ${p.stride_frames}f ` +
    `· overlap ${p.overlap_frames}f · ${(total / p.fps).toFixed(1)}s`;
  st.line2.textContent =
    `${dup} frames rendered twice (${((dup / total) * 100).toFixed(0)}% extra)`;
}

function plotHeight(st) {
  if (!st.plan?.windows?.length) return EMPTY_H;
  return TOP + st.plan.windows.length * ROW_H + 8;
}

function draw(st) {
  const cssW = Math.max(1, st.canvasWrap.clientWidth || 320);
  const cssH = plotHeight(st);
  st.canvas.style.height = `${cssH}px`;
  st.canvasWrap.style.height = `${cssH}px`;
  const ctx = setupCanvas(st.canvas, cssW, cssH);

  if (!st.plan || !st.plan.windows || !st.plan.windows.length) {
    st.overlay.hidden = false;
    st.overlay.innerHTML = "";
    st.overlay.appendChild(emptyState({
      title: "Context windows",
      hint: "Run the node to see the window plan",
    }));
    ctx.clearRect(0, 0, cssW, cssH);
    st.info.textContent = "";
    updateHeader(st);
    st.node?.setDirtyCanvas?.(true, true);
    return;
  }

  st.overlay.hidden = true;
  const p = st.plan;
  const wins = p.windows;
  const total = Math.max(1, p.total_frames);
  const x0 = 6;
  const x1 = cssW - 6;
  const span = x1 - x0;
  const fx = (f) => x0 + (f / total) * span;

  const fg = resolveCu(st.root, "--cu-ink", "#e8e6f7");
  const accent = resolveCu(st.root, "--cu-accent", "#b494ff");
  const warn = resolveCu(st.root, "--cu-warn", "#f3d288");
  const danger = resolveCu(st.root, "--cu-danger", "#f27a92");

  ctx.clearRect(0, 0, cssW, cssH);
  updateHeader(st);

  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.25;
  ctx.beginPath();
  ctx.moveTo(x0, TOP - 6);
  ctx.lineTo(x1, TOP - 6);
  ctx.stroke();
  ctx.globalAlpha = 1;

  for (let i = 0; i < wins.length; i++) {
    const win = wins[i];
    const y = TOP + i * ROW_H;
    if (y > cssH - 4) break;
    const bx = fx(win.start);
    const bw = Math.max(2, fx(win.end) - fx(win.start));

    ctx.globalAlpha = i === st.sel ? 0.85 : 0.4;
    ctx.fillStyle = accent;
    ctx.fillRect(bx, y + 2, bw, ROW_H - 5);

    if (st.showCost && win.overlap_prev > 0) {
      const ox = fx(win.start);
      const ow = Math.max(1, fx(win.start + win.overlap_prev) - ox);
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = warn;
      ctx.fillRect(ox, y + 2, ow, ROW_H - 5);
    }

    if (win.short_tail) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = danger;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx + 0.5, y + 2.5, bw - 1, ROW_H - 6);
    }
    ctx.globalAlpha = 1;
  }

  const s = wins[Math.min(st.sel, wins.length - 1)];
  st.info.textContent =
    `#${s.index}  ${s.start}-${s.end} (${s.frames}f)  ·  ${s.latent_rows} rows  ` +
    `·  rope ${s.rope_offset.toFixed(2)}` + (s.short_tail ? "  ·  SHORT TAIL" : "");
  if (st._lastPlotH !== cssH) {
    st._lastPlotH = cssH;
    st.node?.setDirtyCanvas?.(true, true);
  }
}

function build(node) {
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
  const headerBody = document.createElement("div");
  headerBody.style.display = "flex";
  headerBody.style.flexDirection = "column";
  headerBody.style.gap = "2px";
  headerBody.style.flex = "1";
  const line1 = document.createElement("span");
  line1.className = "c2c-ui-chart__header-text";
  const line2 = document.createElement("span");
  line2.className = "c2c-ui-chart__header-text";
  line2.style.opacity = "0.75";
  headerBody.appendChild(line1);
  headerBody.appendChild(line2);
  header.appendChild(pill);
  header.appendChild(headerBody);

  const legend = document.createElement("div");
  legend.className = "c2c-ui-chart__chips";
  for (const [label, colour] of [
    ["window", "var(--cu-accent)"],
    ["overlap cost", "var(--cu-warn)"],
    ["short tail", "var(--cu-danger)"],
  ]) {
    const chip = document.createElement("span");
    chip.className = "c2c-ui-chart__chip";
    chip.style.pointerEvents = "none";
    const dot = document.createElement("span");
    dot.className = "c2c-ui-chart__chip-dot";
    dot.style.background = colour;
    chip.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.textContent = label;
    chip.appendChild(lbl);
    legend.appendChild(chip);
  }

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;flex:0 0 auto;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%;";
  canvas.tabIndex = 0;
  const overlay = document.createElement("div");
  overlay.className = "c2c-ui-chart__overlay";
  overlay.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;";
  canvasWrap.appendChild(canvas);
  canvasWrap.appendChild(overlay);

  const bar = document.createElement("div");
  bar.style.cssText = "flex:0 0 auto;display:flex;align-items:flex-start;gap:6px;flex-wrap:wrap;";
  const costBtn = button("Show cost", {
    onClick: () => {
      st.showCost = !st.showCost;
      costBtn.querySelector("span:last-child").textContent =
        st.showCost ? "Hide cost" : "Show cost";
      st.paint();
    },
  });
  costBtn.title =
    "Shade the frames that get generated TWICE because they sit in an overlap.";
  const info = document.createElement("span");
  info.className = "c2c-ui-status";
  info.style.flex = "1 1 200px";
  bar.append(costBtn, info);

  root.append(header, legend, canvasWrap, bar);

  const st = {
    root, canvasWrap, canvas, overlay, bar, info, pill, line1, line2,
    plan: null, sel: 0, showCost: false, node,
    zoomOff: null, resizeObs: null, _lastPlotH: 0,
  };
  node[ST_KEY] = st;
  st.paint = () => draw(st);

  const panelWidget = mountPanel(node, "mmx_context_view", root, { minHeight: MIN_H });
  panelWidget.computeSize = (w) => [w, CHROME_H + plotHeight(st) + 8];
  panelWidget.onPanelResize = () => st.paint();
  st.zoomOff = installZoomRepaint(node, () => st.paint(), "_c2cW6Zoom");

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => st.paint());
    st.resizeObs.observe(canvasWrap);
  }

  const pick = (clientX, clientY) => {
    if (!st.plan) return -1;
    const r = canvas.getBoundingClientRect();
    const y = clientY - r.top;
    const idx = Math.floor((y - TOP) / ROW_H);
    return (idx >= 0 && idx < st.plan.windows.length) ? idx : -1;
  };
  canvas.addEventListener("mousemove", (e) => {
    const i = pick(e.clientX, e.clientY);
    if (i >= 0 && i !== st.sel) { st.sel = i; st.paint(); }
  });
  canvas.addEventListener("click", (e) => {
    const i = pick(e.clientX, e.clientY);
    if (i >= 0) { st.sel = i; st.paint(); canvas.focus(); }
  });
  canvas.addEventListener("keydown", (e) => {
    if (!st.plan) return;
    const n = st.plan.windows.length;
    if (e.key === "ArrowDown") st.sel = Math.min(n - 1, st.sel + 1);
    else if (e.key === "ArrowUp") st.sel = Math.max(0, st.sel - 1);
    else return;
    e.preventDefault();
    st.paint();
  });

  return st;
}

app.registerExtension({
  name: "MiniMaxH3.W6.ContextWindows",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      const st = build(this);
      bindExecutionLifecycle(this, st, {
        onStart: () => { st.plan = null; st.paint(); },
        onAbort: () => {},
      });
      chainOnRemoved(this, () => {
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
      if (st) {
        const raw = firstUiString(output, "mmx_context_plan");
        const plan = raw ? parseJsonSafe(raw) : null;
        if (plan && plan.windows) { st.plan = plan; st.sel = 0; st.paint(); }
      }
      return r;
    };
  },
});
