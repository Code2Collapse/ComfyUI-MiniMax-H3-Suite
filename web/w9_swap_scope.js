/**
 * W9 — H3 Swap Control: what is driving, and what is deliberately not.
 *
 * WHY: which landmarks drive a swap is the node's whole behaviour, and in a
 * list of dropdowns it is invisible. So it is drawn: each driving group lights
 * in the same colour the renderer actually uses, and changing swap_scope or
 * drive_jaw relights it.
 *
 * The jaw is the group worth seeing. It carries head POSE and the CHIN DROP
 * that lets the mouth open, as well as the dupe's skull SHAPE - so it is
 * driven by default, and turning drive_jaw off greys it out with the cost
 * spelled out in the caption rather than left to be discovered in a render.
 *
 * The mouth is the other one. With a locked audio track, the audio can shape
 * the lips itself - but only if the control is not already drawing the dupe's
 * mouth over the top of it. drive_mouth off greys the lips and hands them to
 * the audio; that is the DUB setting, and with no audio locked it leaves the
 * lips with nothing driving them at all, which the caption says outright.
 *
 * The face here is a SCHEMATIC, not a mean face shape. Drawing a real 68-point
 * mean face would imply a precision this diagram does not have; the groups are
 * at anatomically sensible places and that is all it claims.
 *
 * Colours mirror GROUP_COLOUR in mmx_utils/swap_control.py and are pinned by
 * tests/test_js_python_parity.py.
 *
 * Plain ES module, no Vue, chained onRemoved.
 */
import {
  app,
  chainOnRemoved,
  disposeState,
  widgetByName,
} from "./shared.js";
import {
  mountPanel,
  statusLine,
  installZoomRepaint,
  canvasBackingScale,
} from "./c2c_ui/index.js";

const NODE_ID = "MiniMaxH3_SwapControl";
const STATE = "_mmxSwapScope";
const PANEL_MIN = 220;
const NODE_MIN_W = 380;
const PANEL_H = 148;

/** Mirrors GROUP_COLOUR in mmx_utils/swap_control.py. */
const GROUP_COLOUR = {
  jaw: "#8cbfff",
  pupils: "#ffffff",
  body: "#0099ff",
  feet: "#00d9bf",
  left_hand: "#ffb31a",
  right_hand: "#f2731a",
  brows: "#b373ff",
  nose: "#4de673",
  eyes: "#fff259",
  mouth: "#ff4d73",
};

/** Mirrors SWAP_SCOPES[*]["groups"] in mmx_utils/swap_regions.py. */
const SCOPE_GROUPS = {
  lips: ["mouth", "jaw"],
  face: ["jaw", "brows", "nose", "eyes", "pupils", "mouth"],
  head: ["jaw", "brows", "nose", "eyes", "pupils", "mouth"],
  body: ["body", "feet", "left_hand", "right_hand"],
  person: ["body", "feet", "left_hand", "right_hand",
           "jaw", "brows", "nose", "eyes", "pupils", "mouth"],
};

const SCOPE_NOTE = {
  lips: "Mouth and jaw drive it — the chin has to drop for the mouth to open — but only the mouth is regenerated, so a lip-sync never reshapes the chin.",
  face: "The whole face drives it: jaw for head pose, pupils for gaze, mouth for lip movement. The region is masked too, so the skull can still move toward your reference.",
  head: "As face, with hair and skull inside the mask as well.",
  body: "Body, feet and hands. The face is left to the plate.",
  person: "Everything the dupe does — body, hands, face, gaze.",
};

const LEGEND_GROUPS = [
  "jaw", "brows", "nose", "eyes", "pupils", "mouth",
  "body", "feet", "left_hand", "right_hand",
];

function resolveCu(root, token, fallback) {
  try {
    const prop = token.startsWith("--") ? token : `--${token}`;
    const v = getComputedStyle(root).getPropertyValue(prop).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}

/** Schematic face in 0..1 box coordinates. Not a mean face — see the header. */
function facePaths() {
  return {
    jaw: [[0.16, 0.34], [0.17, 0.56], [0.26, 0.76], [0.42, 0.88],
          [0.58, 0.88], [0.74, 0.76], [0.83, 0.56], [0.84, 0.34]],
    brows: [[[0.24, 0.30], [0.32, 0.25], [0.42, 0.27]],
            [[0.58, 0.27], [0.68, 0.25], [0.76, 0.30]]],
    nose: [[0.50, 0.36], [0.50, 0.55], [0.43, 0.60], [0.50, 0.62], [0.57, 0.60]],
    eyes: [[0.26, 0.40], [0.44, 0.40]],
    mouth: [0.50, 0.73],
  };
}

function syncLegend(st, active) {
  for (const g of LEGEND_GROUPS) {
    const chip = st.legendChips[g];
    if (!chip) continue;
    const on = active.has(g);
    chip.style.opacity = on ? "1" : "0.22";
    chip.querySelector(".c2c-ui-chart__chip-dot").style.opacity = on ? "1" : "0.4";
  }
}

function paint(st, node) {
  const scope = String(widgetByName(node, "swap_scope")?.value ?? "face");
  const driveJaw = widgetByName(node, "drive_jaw")?.value !== false;
  const driveMouth = widgetByName(node, "drive_mouth")?.value !== false;
  const active = new Set(SCOPE_GROUPS[scope] || SCOPE_GROUPS.face);
  if (!driveJaw) active.delete("jaw");
  if (!driveMouth) active.delete("mouth");
  const facey = ["brows", "nose", "eyes", "mouth"].some((g) => active.has(g));
  const jawOn2 = active.has("jaw");

  st.pill.textContent = scope;
  syncLegend(st, active);

  const cssW = Math.max(160, st.canvasWrap.clientWidth || 360);
  const cssH = PANEL_H;
  const scale = canvasBackingScale(cssW, cssH);
  const bw = Math.max(1, Math.round(cssW * scale));
  const bh = Math.max(1, Math.round(cssH * scale));
  const canvas = st.canvas;
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  const ink = resolveCu(st.root, "--cu-ink", "#e8e6f7");
  const dim = resolveCu(st.root, "--cu-ink-dim", "#6f6d9b");
  const w = cssW;
  const h = cssH;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(0, 0, w, h);

  const size = Math.min(h - 8, w * 0.55);
  const bx = 12;
  const by = (h - size) / 2;
  const P = (p) => [bx + p[0] * size, by + p[1] * size];
  const F = facePaths();

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const jawOn = active.has("jaw");
  ctx.save();
  if (!jawOn) ctx.setLineDash([3, 3]);
  ctx.strokeStyle = jawOn ? GROUP_COLOUR.jaw : dim;
  if (!jawOn) ctx.globalAlpha = 0.42;
  ctx.lineWidth = jawOn ? 2.2 : 1.5;
  ctx.beginPath();
  F.jaw.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.restore();

  const stroke = (g, wdt = 2.2) => {
    const on = active.has(g);
    ctx.globalAlpha = on ? 1 : 0.20;
    ctx.strokeStyle = on ? GROUP_COLOUR[g] : dim;
    ctx.lineWidth = on ? wdt : 1.2;
  };

  stroke("brows");
  for (const arc of F.brows) {
    ctx.beginPath();
    arc.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }

  stroke("nose");
  ctx.beginPath();
  F.nose.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();

  stroke("eyes");
  for (const e of F.eyes) {
    const [x, y] = P(e);
    ctx.beginPath();
    ctx.ellipse(x + size * 0.04, y, size * 0.075, size * 0.036, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  for (const e of F.eyes) {
    const [x, y] = P(e);
    ctx.globalAlpha = active.has("pupils") ? 1 : 0.22;
    ctx.fillStyle = active.has("pupils") ? GROUP_COLOUR.pupils : dim;
    ctx.beginPath();
    ctx.arc(x + size * 0.04, y, Math.max(1.5, size * 0.014), 0, Math.PI * 2);
    ctx.fill();
  }

  const mouthInScope = (SCOPE_GROUPS[scope] || []).includes("mouth");
  const mouthToAudio = mouthInScope && !driveMouth;
  ctx.save();
  if (mouthToAudio) ctx.setLineDash([3, 3]);
  stroke("mouth", 2.6);
  if (mouthToAudio) {
    ctx.strokeStyle = "rgba(255,77,115,0.45)";
    ctx.lineWidth = 1.6;
  }
  const [mx, my] = P(F.mouth);
  ctx.beginPath();
  ctx.ellipse(mx, my, size * 0.135, size * 0.055, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(mx - size * 0.135, my);
  ctx.lineTo(mx + size * 0.135, my);
  ctx.stroke();
  ctx.restore();

  const jl = jawOn ? "jaw — head pose + chin drop" : "jaw — not driven";
  const jawNote = !facey
    ? " No face group is driven in this scope."
    : jawOn2
      ? " A strong jaw control pulls face width toward the dupe — lower the ControlNet strength if the head starts taking their shape."
      : " drive_jaw is off: the reference actor's skull is unopposed, but a profile will read as a front-on face and the mouth cannot open as far.";
  const mouthNote = mouthToAudio
    ? " drive_mouth is off: the lips are still masked, so they can change, but nothing is drawing them — lock an audio track or they have nothing to follow."
    : "";
  st.caption.setText(`${jl}. ${SCOPE_NOTE[scope] || ""}${jawNote}${mouthNote}`, "default");
}

function build(node) {
  disposeState(node, STATE);

  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "4px";
  root.style.width = "100%";

  const header = document.createElement("div");
  header.className = "c2c-ui-chart__header";
  const pill = document.createElement("span");
  pill.className = "c2c-ui-chart__pill";
  pill.textContent = "face";
  const summary = document.createElement("span");
  summary.className = "c2c-ui-chart__header-text";
  summary.textContent = "Groups lit in the colour the renderer uses";
  header.appendChild(pill);
  header.appendChild(summary);

  const legend = document.createElement("div");
  legend.className = "c2c-ui-chart__chips";
  legend.style.flexWrap = "wrap";
  const legendChips = {};
  for (const g of LEGEND_GROUPS) {
    const chip = document.createElement("span");
    chip.className = "c2c-ui-chart__chip";
    chip.style.pointerEvents = "none";
    const dot = document.createElement("span");
    dot.className = "c2c-ui-chart__chip-dot";
    dot.style.background = GROUP_COLOUR[g];
    chip.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.textContent = g.replace("_", " ");
    chip.appendChild(lbl);
    legend.appendChild(chip);
    legendChips[g] = chip;
  }

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;min-height:160px;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:160px;border-radius:4px;";
  canvasWrap.appendChild(canvas);

  const caption = statusLine();
  root.append(header, legend, canvasWrap, caption.el);

  const st = {
    root, canvasWrap, canvas, pill, caption, legendChips,
    zoomOff: null, resizeObs: null,
  };
  node[STATE] = st;

  st.paint = () => paint(st, node);

  const panelWidget = mountPanel(node, "mmx_swap_scope", root, { minHeight: PANEL_MIN });
  panelWidget.onPanelResize = () => st.paint();
  st.zoomOff = installZoomRepaint(node, () => st.paint(), "_c2cW9Zoom");

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => st.paint());
    st.resizeObs.observe(canvasWrap);
  }

  for (const name of ["swap_scope", "drive_jaw", "drive_mouth"]) {
    const wdg = widgetByName(node, name);
    if (!wdg) continue;
    const prev = wdg.callback;
    wdg.callback = function (...a) {
      const r = prev?.apply(this, a);
      st.paint();
      return r;
    };
  }

  chainOnRemoved(node, () => {
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { st.resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
    disposeState(node, STATE);
  });
  st.paint();
}

app.registerExtension({
  name: "MiniMaxSuite.SwapScope",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      try { build(this); } catch (_e) { /* a broken diagram must not kill the node */ }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...a) {
      const r = onConfigure?.apply(this, a);
      this[STATE]?.paint?.();
      return r;
    };
  },
});
