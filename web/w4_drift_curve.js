/**
 * W4 — H3 Drift QC curve + pass/fail badge (read-only).
 * Reads structured mmx_drift JSON from UI payload.
 */
import {
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  firstUiString,
  parseJsonSafe,
  widgetByName,
} from "./shared.js";
import {
  lineChart,
  mountPanel,
  installZoomRepaint,
} from "./c2c_ui/index.js";

const NODE = "MiniMaxH3_DriftQC";
const ST_KEY = "_mmxW4";
const PANEL_MIN = 230;
const NODE_MIN_W = 380;

function buildHeaderText(series, threshold, peakPx, peakFrame, passed) {
  const over = series.filter((v) => v > threshold).length;
  const peakStr = Number(peakPx).toFixed(1);
  const thrStr = Number(threshold).toFixed(1);
  const frameStr = peakFrame >= 0 ? peakFrame : 0;
  if (!passed) {
    const overNote = over > 0 ? ` (${over} frame${over === 1 ? "" : "s"} over)` : "";
    return `Peak ${peakStr} px at frame ${frameStr} is above ${thrStr} px${overNote}`;
  }
  return `Peak ${peakStr} px at frame ${frameStr} stays under ${thrStr} px`;
}

function setHeader(st, passed, text) {
  const pill = st.pill;
  pill.textContent = passed === true ? "PASS" : passed === false ? "FAIL" : "—";
  pill.classList.remove("c2c-ui-chart__pill--ok", "c2c-ui-chart__pill--danger");
  if (passed === true) pill.classList.add("c2c-ui-chart__pill--ok");
  else if (passed === false) pill.classList.add("c2c-ui-chart__pill--danger");
  st.summary.textContent = text || "";
}

function applyPayload(st, data, node) {
  const series = (data.drift_px || []).map(Number);
  const threshold = Number(
    data.threshold_px ?? widgetByName(node, "threshold_px")?.value ?? 2.0,
  );
  const peakPx = data.peak_px != null
    ? Number(data.peak_px)
    : (series.length ? Math.max(...series) : 0);
  const peakFrame = data.peak_frame != null
    ? Number(data.peak_frame)
    : (series.length ? series.indexOf(Math.max(...series)) : -1);
  const passed = Boolean(data.passed);

  st.series = series;
  st.threshold = threshold;
  st.passed = passed;

  const xs = series.map((_, i) => i);
  st.chart.setThresholds([{
    axis: "y",
    value: threshold,
    label: `${threshold.toFixed(1)} px`,
    kind: "danger",
  }]);
  st.chart.setData(xs, { drift: series });
  if (peakFrame >= 0 && series.length) {
    st.chart.setMarkers([{
      x: peakFrame,
      seriesId: "drift",
      label: `peak ${peakPx.toFixed(1)} px · f${peakFrame}`,
    }]);
  } else {
    st.chart.setMarkers([]);
  }
  st.chart.setState("ready");
  setHeader(st, passed, buildHeaderText(series, threshold, peakPx, peakFrame, passed));
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
  root.appendChild(header);

  const chart = lineChart({
    minHeight: 130,
    xLabel: "frame",
    yLabel: "drift px",
    xFormat: (v) => String(Math.round(v)),
    series: [
      { id: "drift", label: "drift", color: "--cu-series-2", axis: "y", points: true },
    ],
    empty: {
      title: "Drift QC",
      hint: "Run the node to see exterior drift.",
    },
  });
  chart.setState("empty");
  root.appendChild(chart.el);

  const st = {
    root,
    chart,
    header,
    pill,
    summary,
    state: "empty",
    message: "Run the node to see exterior drift.",
    series: null,
    threshold: 2,
    passed: null,
    apiOff: [],
    zoomOff: null,
    panelWidget: null,
  };
  node[ST_KEY] = st;

  st.panelWidget = mountPanel(node, "mmx_drift_curve", root, { minHeight: PANEL_MIN });
  st.panelWidget.onPanelResize = () => chart.redraw();
  st.zoomOff = installZoomRepaint(node, () => chart.redraw(), "_c2cW4Zoom");

  st.apiOff.push(bindExecutionLifecycle(node, st, {
    onStart: () => {
      st.state = "loading";
      chart.setState("loading");
    },
    onAbort: (msg) => {
      if (st.state === "loading") {
        st.state = "error";
        st.message = msg || "Execution failed.";
        chart.setState("error", st.message);
        setHeader(st, null, st.message);
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
  const raw = firstUiString(output, "mmx_drift");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.drift_px)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_drift UI payload." : "No mmx_drift in UI output.";
    st.passed = null;
    st.chart.setState("error", st.message);
    setHeader(st, null, st.message);
    return;
  }
  st.state = "ready";
  applyPayload(st, data, node);
}

app.registerExtension({
  name: "MiniMaxH3.W4DriftCurve",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const _created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = _created?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
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
