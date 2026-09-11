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

const NODES = new Set(["MiniMaxH3_SubjectCrop", "MiniMaxH3_SubjectCropAdvanced"]);
const ST_KEY = "_mmxW5";
const MIN_H = 230;
const BAR_H = 26;

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];

  const box = document.createElement("div");
  box.style.cssText =
    "width:100%;height:100%;position:relative;overflow:hidden;display:flex;flex-direction:column;gap:4px;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;flex:1 1 auto;min-height:0;";
  canvas.tabIndex = 0;

  const bar = document.createElement("div");
  bar.style.cssText =
    "flex:0 0 auto;display:flex;align-items:center;gap:6px;font:11px system-ui,sans-serif;";

  const play = document.createElement("button");
  play.textContent = "Play";
  play.title = "Step through the plan. A plan that claims 'stillness' should " +
               "look still here; if the box twitches every frame, raise the " +
               "movement cost or use the action mode.";
  play.style.cssText =
    "font:11px system-ui,sans-serif;padding:2px 8px;cursor:pointer;border-radius:4px;";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "0";
  slider.step = "1";
  slider.value = "0";
  slider.title = "Scrub the planned frames. Jump frames are ticked on the timeline above.";
  slider.style.cssText = "flex:1 1 auto;min-width:40px;";

  const readout = document.createElement("span");
  readout.style.cssText = "opacity:.85;white-space:nowrap;min-width:118px;text-align:right;";

  bar.append(play, slider, readout);
  box.append(canvas, bar);

  const st = {
    box, canvas, bar, play, slider, readout,
    ctx: canvas.getContext("2d"),
    plan: null, frame: 0, playing: false, timer: 0, raf: 0,
  };
  node[ST_KEY] = st;

  const paint = rafThrottle(() => draw(st));
  st.paint = paint;

  slider.addEventListener("input", () => {
    st.frame = Number(slider.value) || 0;
    paint();
  });
  canvas.addEventListener("keydown", (e) => {
    if (!st.plan) return;
    const n = st.plan.boxes.length;
    if (e.key === "ArrowRight") { st.frame = Math.min(n - 1, st.frame + 1); }
    else if (e.key === "ArrowLeft") { st.frame = Math.max(0, st.frame - 1); }
    else return;
    e.preventDefault();
    slider.value = String(st.frame);
    paint();
  });
  // Click a jump tick to jump there — the jumps are the frames worth seeing.
  canvas.addEventListener("click", (e) => {
    if (!st.plan) return;
    const r = canvas.getBoundingClientRect();
    if (e.clientY - r.top < r.height - BAR_H) return;
    const n = st.plan.boxes.length;
    const f = Math.round(((e.clientX - r.left) / Math.max(1, r.width)) * (n - 1));
    st.frame = Math.max(0, Math.min(n - 1, f));
    slider.value = String(st.frame);
    paint();
  });
  play.addEventListener("click", () => setPlaying(st, !st.playing));

  return st;
}

function setPlaying(st, on) {
  st.playing = on;
  st.play.textContent = on ? "Pause" : "Play";
  if (st.timer) { clearInterval(st.timer); st.timer = 0; }
  if (!on || !st.plan) return;
  st.timer = setInterval(() => {
    const n = st.plan.boxes.length;
    st.frame = (st.frame + 1) % n;
    st.slider.value = String(st.frame);
    st.paint();
  }, 1000 / 12);
}

function draw(st) {
  const { canvas, ctx } = st;
  const cssW = canvas.clientWidth || 300;
  const cssH = canvas.clientHeight || MIN_H;
  setupDpiCanvas(canvas, cssW, cssH);
  ctx.clearRect(0, 0, cssW, cssH);

  const fg = themeVar("--input-text") || "#ddd";
  const accent = themeVar("--c2c-accentVivid") || "#4ea1ff";

  if (!st.plan || !st.plan.boxes || !st.plan.boxes.length) {
    drawPlaceholder(ctx, cssW, cssH, "Run the node to see its crop plan", "empty");
    return;
  }

  const plan = st.plan;
  const n = plan.boxes.length;
  const f = Math.max(0, Math.min(n - 1, st.frame));
  const b = plan.boxes[f];

  const plotH = cssH - BAR_H - 6;

  // ---- frame rectangle, aspect-fit -----------------------------------------
  const fw = plan.frame_w || 1, fh = plan.frame_h || 1;
  const scale = Math.min((cssW - 16) / fw, (plotH - 16) / fh);
  const ox = (cssW - fw * scale) / 2;
  const oy = (plotH - fh * scale) / 2;

  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  ctx.strokeRect(ox, oy, fw * scale, fh * scale);
  ctx.globalAlpha = 1;

  // ---- every box faintly, so the SPREAD of the plan is visible -------------
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.12;
  for (const q of plan.boxes) {
    ctx.strokeRect(ox + q.x * scale, oy + q.y * scale, q.w * scale, q.h * scale);
  }
  ctx.globalAlpha = 1;

  // ---- the current box, solid ---------------------------------------------
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.strokeRect(ox + b.x * scale, oy + b.y * scale, b.w * scale, b.h * scale);

  // crosshair at box centre — makes drift obvious while scrubbing
  const cx = ox + (b.x + b.w / 2) * scale;
  const cy = oy + (b.y + b.h / 2) * scale;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // ---- header text ---------------------------------------------------------
  ctx.fillStyle = fg;
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.globalAlpha = 0.85;
  ctx.fillText(
    `${plan.mode}  ·  box ${b.w}x${b.h}  ·  out ${plan.out_w}x${plan.out_h}  ·  ` +
    `${plan.jumps.length} jump${plan.jumps.length === 1 ? "" : "s"} / ${n}f`,
    6, 4);
  ctx.globalAlpha = 1;

  // ---- timeline with jump ticks -------------------------------------------
  const ty = cssH - BAR_H + 6;
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = fg;
  ctx.fillRect(0, ty, cssW, 2);
  ctx.globalAlpha = 1;

  ctx.fillStyle = accent;
  for (const j of plan.jumps) {
    const x = (j / Math.max(1, n - 1)) * cssW;
    ctx.fillRect(x - 1, ty - 4, 2, 10);
  }
  // playhead
  const px = (f / Math.max(1, n - 1)) * cssW;
  ctx.fillStyle = fg;
  ctx.fillRect(px - 1, ty - 7, 2, 16);

  st.readout.textContent = `f ${f}/${n - 1}  ${b.x},${b.y}`;
}

function applyUi(node, st, output) {
  const raw = firstUiString(output, "mmx_crop_plan");
  const plan = raw ? parseJsonSafe(raw) : null;
  if (!plan || !plan.boxes) return;
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
      const st = buildDom(this);
      addDomWidgetLast(this, "mmx_crop_plan_view", st.box,
                       () => (this._mmxW5Collapsed ? 0 : MIN_H));
      bindExecutionLifecycle(this, st, {
        onStart: () => { st.plan = null; st.paint(); },
        onAbort: () => { setPlaying(st, false); },
      });
      chainOnRemoved(this, () => {
        setPlaying(st, false);
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
