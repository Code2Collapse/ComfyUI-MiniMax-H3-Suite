/**
 * W7 — H3 Mask-Aware ControlNet gate profile.
 *
 * WHY: this node's whole job is deciding how much control reaches a row, as a
 * function of how much that row is being generated. That is a CURVE, and the
 * two widgets that define it read as numbers with no shape:
 *
 *   preserved_strength   what a fully-preserved row still receives
 *   boundary_softness    how wide the ramp between the two states is
 *
 * "0.0 and 1.0" does not tell a compositor whether their inpaint boundary will
 * show a seam. So the curve is drawn, with the preserved end and the generated
 * end labelled, and the node says in words which of the three regimes it is in.
 *
 * The one that matters most is the failure state: preserved_strength = 1.0
 * silently restores ComfyUI's behaviour, which is the bug this node exists to
 * fix. Somebody who set it to 1.0 while experimenting needs to be told, not
 * left wondering why the fix stopped working. That case is drawn in red and
 * named.
 *
 * Runtime row counts are NOT shown here. They depend on the mask actually
 * sampled with, which happens long after this node returns, and a number that
 * confident would be a guess. They go to the log instead.
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
  lineChart,
  mountPanel,
  statusLine,
  installZoomRepaint,
} from "./c2c_ui/index.js";

const NODE_ID = "MiniMaxH3_MaskAwareControlNet";
const STATE = "_mmxMaskGate";
const PANEL_MIN = 280;
const NODE_MIN_W = 380;

/** The gate the node actually applies, for one row's mask value in 0..1. */
function gateAt(maskValue, preservedStrength, softness) {
  let m = maskValue;
  if (softness > 0) {
    const w = Math.min(0.9, softness / 8);
    const lo = 0.5 - w / 2;
    const t = Math.min(1, Math.max(0, (m - lo) / Math.max(w, 1e-6)));
    m = t * t * (3 - 2 * t);
  } else {
    m = m >= 0.5 ? 1 : 0;
  }
  return preservedStrength + (1 - preservedStrength) * m;
}

/** Which of the 49 control channels are carrying real data. */
function channels(node) {
  const w = (n) => node.widgets?.find((x) => x.name === n);
  const inpaint = !!w("inpaint")?.value;
  const hasControl = node.inputs?.some(
    (i) => i.name === "control_video" && i.link != null);
  const hasMask = node.inputs?.some((i) => i.name === "mask" && i.link != null);
  return { inpaint, hasControl, hasMask: inpaint && hasMask };
}

function regime(preserved, softness) {
  if (preserved >= 0.999) {
    return {
      tone: "error",
      title: "the fix is switched off",
      body:
        "preserved_strength is 1.0, so the mask is ignored and the control " +
        "pushes every row - including the ones the sampler is holding still. " +
        "That is ComfyUI's own behaviour, contradiction included. Lower it.",
    };
  }
  if (preserved <= 0.001) {
    return {
      tone: "ok",
      title: "control is kept out of preserved rows",
      body:
        softness > 0
          ? `Rows the mask preserves receive nothing; the ramp spans about ` +
            `${softness.toFixed(1)} latent patches so the edge does not show.`
          : "Rows the mask preserves receive nothing. The edge is hard - if a " +
            "seam shows where the control stops, raise boundary_softness.",
    };
  }
  return {
    tone: "warn",
    title: "partial hold",
    body:
      `Preserved rows keep ${Math.round(preserved * 100)}% of the control. ` +
      "The control still informs them, but no longer overrides the mask.",
  };
}

function read(node, name, fallback) {
  const w = widgetByName(node, name);
  return w ? Number(w.value) : fallback;
}

function buildChannelChips() {
  const row = document.createElement("div");
  row.className = "c2c-ui-chart__chips";
  row.style.flexWrap = "wrap";
  const specs = [
    { id: "control", label: "control (24)", colour: "#6fa8d1" },
    { id: "vis", label: "vis (1)", colour: "#6fb36f" },
    { id: "masked", label: "masked (24)", colour: "#c89a4a" },
  ];
  const chips = {};
  for (const s of specs) {
    const chip = document.createElement("span");
    chip.className = "c2c-ui-chart__chip";
    chip.style.cursor = "default";
    chip.style.pointerEvents = "none";
    const dot = document.createElement("span");
    dot.className = "c2c-ui-chart__chip-dot";
    dot.style.background = s.colour;
    chip.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.textContent = s.label;
    chip.appendChild(lbl);
    row.appendChild(chip);
    chips[s.id] = chip;
  }
  return { row, chips };
}

function updateChannelChips(st, ch) {
  const on = { control: ch.hasControl, vis: true, masked: ch.hasMask };
  for (const [id, chip] of Object.entries(st.channelChips)) {
    chip.style.opacity = on[id] ? "1" : "0.18";
  }
}

function rebuild(st, node) {
  const preserved = read(node, "preserved_strength", 0);
  const softness = read(node, "boundary_softness", 1);
  const strength = read(node, "strength", 1);
  const N = 64;
  const xs = [];
  const gate = [];
  for (let i = 0; i < N; i++) {
    const m = i / (N - 1);
    xs.push(m);
    gate.push(gateAt(m, preserved, softness) * Math.min(1, strength));
  }
  st.chart.setData(xs, { gate });

  const thresholds = [
    { axis: "y", value: preserved, label: `preserved ${preserved.toFixed(2)}` },
    { axis: "y", value: 1, label: "full control" },
  ];
  if (softness > 0) {
    const w = Math.min(0.9, softness / 8);
    thresholds.push(
      { axis: "x", value: 0.5 - w / 2, label: "ramp start" },
      { axis: "x", value: 0.5 + w / 2, label: "ramp end" },
    );
  }
  st.chart.setThresholds(thresholds);
  st.chart.setMarkers([]);

  const info = regime(preserved, softness);
  st.regimeLine.setText(`${info.title}. ${info.body}`, info.tone);

  const ch = channels(node);
  updateChannelChips(st, ch);
  const mode = ch.inpaint
    ? "inpaint — all 49 channels carry data"
    : "structural only — visibility filled with ONES, so nothing reads as a hole";
  st.modeLine.setText(mode, "default");

  st.chart.setState("ready");
}

function build(node) {
  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "4px";
  root.style.width = "100%";
  root.style.height = "100%";

  const chart = lineChart({
    minHeight: 150,
    xLabel: "mask fraction (preserved → generated)",
    yLabel: "control strength",
    xFormat: (v) => v.toFixed(2),
    yFormat: (v) => v.toFixed(2),
    series: [
      { id: "gate", label: "gate", color: "--cu-series-1", axis: "y" },
    ],
  });

  const { row: channelRow, chips: channelChips } = buildChannelChips();
  const regimeLine = statusLine();
  const modeLine = statusLine();

  root.appendChild(chart.el);
  root.appendChild(channelRow);
  root.appendChild(regimeLine.el);
  root.appendChild(modeLine.el);

  const st = {
    dead: false,
    root,
    chart,
    channelChips,
    regimeLine,
    modeLine,
    zoomOff: null,
  };
  node[STATE] = st;

  st.zoomOff = installZoomRepaint(node, () => chart.redraw(), "_c2cW7Zoom");
  mountPanel(node, "mmx_mask_gate", root, { minHeight: PANEL_MIN });
  chart.el.querySelector(".c2c-ui-chart__plot-wrap").style.minHeight = "150px";

  const repaint = () => {
    if (st.dead) return;
    rebuild(st, node);
  };

  for (const w of node.widgets || []) {
    const prev = w.callback;
    w.callback = function (...args) {
      const r = prev?.apply(this, args);
      repaint();
      return r;
    };
  }

  chainOnRemoved(node, () => {
    st.dead = true;
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { chart.destroy(); } catch (_e) { /* ignore */ }
    disposeState(node, STATE);
  });

  rebuild(st, node);
  return st;
}

app.registerExtension({
  name: "MiniMaxSuite.MaskAwareControlGate",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      try {
        build(this);
      } catch (_e) {
        /* a broken plot must never take the node down */
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      if (!this[STATE]?.dead) rebuild(this[STATE], this);
      return r;
    };
  },
});
