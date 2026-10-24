/**
 * W12 — H3 Tone Compensate: the measured seam drift, per channel, across the overlap frames (C2C L9.39).
 *
 * The node has always serialised its per-frame drift into the UI payload ("mmx_tone_drift") for a plot, and nothing
 * read it: the only trace of the measurement was the report text. The curve answers the question the node exists
 * for - was the seam a tone problem at all? - at a glance: three flat lines near zero mean "no", a slope or an
 * offset in one channel says which colour jumped and how it moved over the overlap.
 */
import {
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  firstUiString,
  parseJsonSafe,
} from "./shared.js";
import { lineChart, mountPanel, installZoomRepaint } from "./c2c_ui/index.js";

const NODE = "MiniMaxH3_ToneCompensate";
const ST_KEY = "_mmxW12";
const PANEL_MIN = 230;
const NODE_MIN_W = 380;
const NONE = 1e-4;            // the node's own "essentially no drift" line

function setHeader(st, kind, text) {
  st.pill.textContent = kind === "ok" ? "NO DRIFT" : kind === "warn" ? "DRIFT" : "—";
  st.pill.classList.remove("c2c-ui-chart__pill--ok", "c2c-ui-chart__pill--danger");
  if (kind === "ok") st.pill.classList.add("c2c-ui-chart__pill--ok");
  else if (kind === "warn") st.pill.classList.add("c2c-ui-chart__pill--danger");
  st.summary.textContent = text || "";
}

function applyPayload(st, data) {
  const rows = (data.drift_per_frame || []).map((r) => (Array.isArray(r) ? r.map(Number) : [Number(r)]));
  const xs = rows.map((_, i) => i);
  const col = (c) => rows.map((r) => (r.length > c ? r[c] : r[0]));
  st.chart.setData(xs, { r: col(0), g: col(1), b: col(2) });
  st.chart.setThresholds([{ axis: "y", value: 0, label: "0", kind: "default" }]);
  let peakI = -1, peak = 0;
  rows.forEach((r, i) => { const m = Math.max(...r.map(Math.abs)); if (m > peak) { peak = m; peakI = i; } });
  st.chart.setMarkers(peakI >= 0 ? [{ x: peakI, seriesId: "r", label: `max |drift| ${peak.toFixed(4)} · f${peakI}` }] : []);
  st.chart.setState("ready");
  const ch = (data.drift_per_channel || []).map((v) => (v >= 0 ? "+" : "") + Number(v).toFixed(4)).join(" ");
  let text = `${data.mode}: mean drift R G B ${ch} over ${data.overlap_used} overlap frame(s)`;
  if (data.overlap_clamped) {
    text += ` - overlap ${data.overlap_requested} was clamped to ${data.overlap_used}: check it is the keyframe count`;
  }
  setHeader(st, Number(data.drift_abs_max) < NONE ? "ok" : "warn", text);
}

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];
  const root = document.createElement("div");
  Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%" });
  const header = document.createElement("div");
  header.className = "c2c-ui-chart__header";
  const pill = document.createElement("span");
  pill.className = "c2c-ui-chart__pill";
  pill.textContent = "—";
  const summary = document.createElement("span");
  summary.className = "c2c-ui-chart__header-text";
  header.append(pill, summary);
  root.appendChild(header);

  const chart = lineChart({
    minHeight: 130,
    xLabel: "overlap frame",
    yLabel: "drift (target - source)",
    xFormat: (v) => String(Math.round(v)),
    yFormat: (v) => v.toFixed(3),
    series: [
      { id: "r", label: "R", color: "--cu-series-4", axis: "y", points: true },
      { id: "g", label: "G", color: "--cu-series-2", axis: "y", points: true },
      { id: "b", label: "B", color: "--cu-series-5", axis: "y", points: true },
    ],
    empty: { title: "Seam drift", hint: "Run the node to see how far the target's colour moved from the source." },
  });
  chart.setState("empty");
  root.appendChild(chart.el);

  const st = { root, chart, pill, summary, state: "empty", apiOff: [], zoomOff: null, panelWidget: null };
  node[ST_KEY] = st;
  st.panelWidget = mountPanel(node, "mmx_tone_drift", root, { minHeight: PANEL_MIN });
  st.panelWidget.onPanelResize = () => chart.redraw();
  st.zoomOff = installZoomRepaint(node, () => chart.redraw(), "_c2cW12Zoom");
  st.apiOff.push(bindExecutionLifecycle(node, st, {
    onStart: () => { st.state = "loading"; chart.setState("loading"); },
    onAbort: (msg) => {
      if (st.state === "loading") {
        st.state = "error";
        chart.setState("error", msg || "Execution failed.");
        setHeader(st, null, msg || "Execution failed.");
      }
    },
  }));
  chainOnRemoved(node, () => {
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { chart.destroy(); } catch (_e) { /* ignore */ }
    disposeState(node, ST_KEY);
  });
  return st;
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  const data = parseJsonSafe(firstUiString(output, "mmx_tone_drift"));
  if (!data || !Array.isArray(data.drift_per_frame)) {
    st.state = "error";
    st.chart.setState("error", "No drift measurement in this run's output.");
    setHeader(st, null, "No drift measurement in this run's output.");
    return;
  }
  st.state = "ready";
  applyPayload(st, data);
}

app.registerExtension({
  name: "MiniMaxH3.W12ToneDrift",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      buildDom(this);
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = executed?.apply(this, arguments);
      try { handleExecuted(this, output || {}); } catch (_) { /* never break the graph */ }
      return r;
    };
  },
});
