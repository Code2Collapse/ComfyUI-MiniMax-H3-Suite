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
  addDomWidgetLast,
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  drawPlaceholder,
  firstUiString,
  parseJsonSafe,
  rafThrottle,
  setupDpiCanvas,
  themeVar,
} from "./shared.js";

const NODE = "MiniMaxH3_ContextWindows";
const ST_KEY = "_mmxW6";
const MIN_H = 190;
const ROW_H = 16;
const TOP = 34;

function build(node) {
  if (node[ST_KEY]) return node[ST_KEY];

  const box = document.createElement("div");
  box.style.cssText =
    "width:100%;height:100%;display:flex;flex-direction:column;gap:3px;overflow:hidden;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;flex:1 1 auto;min-height:0;";
  canvas.tabIndex = 0;

  const bar = document.createElement("div");
  bar.style.cssText =
    "flex:0 0 auto;display:flex;align-items:center;gap:6px;font:11px system-ui,sans-serif;";

  const costBtn = document.createElement("button");
  costBtn.textContent = "Show cost";
  costBtn.title =
    "Shade the frames that get generated TWICE because they sit in an overlap. " +
    "That is the render time the seam quality is costing you.";
  costBtn.style.cssText =
    "font:11px system-ui,sans-serif;padding:2px 8px;cursor:pointer;border-radius:4px;";

  const info = document.createElement("span");
  info.style.cssText = "flex:1 1 auto;opacity:.85;white-space:nowrap;overflow:hidden;";

  bar.append(costBtn, info);
  box.append(canvas, bar);

  const st = {
    box, canvas, bar, info, costBtn,
    ctx: canvas.getContext("2d"),
    plan: null, sel: 0, showCost: false,
  };
  node[ST_KEY] = st;
  st.paint = rafThrottle(() => draw(st));

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
  costBtn.addEventListener("click", () => {
    st.showCost = !st.showCost;
    costBtn.textContent = st.showCost ? "Hide cost" : "Show cost";
    st.paint();
  });

  return st;
}

function draw(st) {
  const { canvas, ctx } = st;
  const w = canvas.clientWidth || 320;
  const h = canvas.clientHeight || MIN_H;
  setupDpiCanvas(canvas, w, h);
  ctx.clearRect(0, 0, w, h);

  const fg = themeVar("--input-text") || "#ddd";
  const accent = themeVar("--c2c-accentVivid") || "#4ea1ff";

  if (!st.plan || !st.plan.windows || !st.plan.windows.length) {
    drawPlaceholder(ctx, w, h, "Run the node to see the window plan", "empty");
    return;
  }

  const p = st.plan;
  const wins = p.windows;
  const total = Math.max(1, p.total_frames);
  const x0 = 6, x1 = w - 6;
  const span = x1 - x0;
  const fx = (f) => x0 + (f / total) * span;

  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "top";

  // header
  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.9;
  const dup = wins.reduce((a, x) => a + (x.overlap_prev || 0), 0);
  ctx.fillText(
    `${p.window_count} passes  ·  win ${p.window_frames}f  ·  stride ${p.stride_frames}f  ` +
    `·  overlap ${p.overlap_frames}f  ·  ${(total / p.fps).toFixed(1)}s`, 6, 4);
  ctx.globalAlpha = 0.55;
  ctx.fillText(`${dup} frames rendered twice (${((dup / total) * 100).toFixed(0)}% extra)`,
               6, 18);
  ctx.globalAlpha = 1;

  // full-length rule
  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.25;
  ctx.beginPath();
  ctx.moveTo(x0, TOP - 6); ctx.lineTo(x1, TOP - 6); ctx.stroke();
  ctx.globalAlpha = 1;

  for (let i = 0; i < wins.length; i++) {
    const win = wins[i];
    const y = TOP + i * ROW_H;
    if (y > h - 4) break;
    const bx = fx(win.start), bw = Math.max(2, fx(win.end) - fx(win.start));

    // the window itself
    ctx.globalAlpha = i === st.sel ? 0.85 : 0.4;
    ctx.fillStyle = accent;
    ctx.fillRect(bx, y + 2, bw, ROW_H - 5);

    // the overlap with the previous window — the double-rendered part
    if (st.showCost && win.overlap_prev > 0) {
      const ox = fx(win.start), ow = Math.max(1, fx(win.start + win.overlap_prev) - ox);
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = "#e0a33a";
      ctx.fillRect(ox, y + 2, ow, ROW_H - 5);
    }

    // a short tail is a real gotcha — mark it
    if (win.short_tail) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#e06c6c";
      ctx.lineWidth = 1;
      ctx.strokeRect(bx + 0.5, y + 2.5, bw - 1, ROW_H - 6);
    }
    ctx.globalAlpha = 1;
  }

  const s = wins[Math.min(st.sel, wins.length - 1)];
  st.info.textContent =
    `#${s.index}  ${s.start}-${s.end} (${s.frames}f)  ·  ${s.latent_rows} rows  ` +
    `·  rope ${s.rope_offset.toFixed(2)}` + (s.short_tail ? "  ·  SHORT TAIL" : "");
}

app.registerExtension({
  name: "MiniMaxH3.W6.ContextWindows",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const st = build(this);
      addDomWidgetLast(this, "mmx_context_view", st.box, () => MIN_H);
      bindExecutionLifecycle(this, st, {
        onStart: () => { st.plan = null; st.paint(); },
        onAbort: () => {},
      });
      chainOnRemoved(this, () => disposeState(this, ST_KEY));
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
