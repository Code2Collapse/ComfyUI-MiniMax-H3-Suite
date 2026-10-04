/**
 * W3 — H3 Sigma Inspector live plot.
 * Reads sigma_json on execute; recomputes curve client-side when shift widgets change.
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
  statusLine,
  installZoomRepaint,
} from "./c2c_ui/index.js";

const NODE = "MiniMaxH3_SigmaInspector";
const ST_KEY = "_mmxW3";
const PANEL_MIN = 250;
const NODE_MIN_W = 380;

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

function stepsToChartData(steps) {
  const xs = steps.map((s) => s.step);
  return {
    xs,
    values: {
      video: steps.map((s) => s.sigma_v),
      audio: steps.map((s) => s.sigma_a),
      ratio: steps.map((s) => s.dsigma_a_dsigma_v),
      carry: steps.map((s) => s.carry_factor),
    },
  };
}

function updateShiftStatus(st, shiftVideo, shiftAudio) {
  const tv = Number(st.trainedVideo ?? 12);
  const ta = Number(st.trainedAudio ?? 3);
  const sv = Number(shiftVideo);
  const sa = Number(shiftAudio);
  if (Math.abs(sv - tv) < 1e-4 && Math.abs(sa - ta) < 1e-4) {
    st.status.setText(`Trained shifts ${tv} / ${ta}`, "ok");
  } else {
    st.status.setText(
      `Shifts ${sv} / ${sa} differ from the trained ${tv} / ${ta} — audio and video may drift apart`,
      "warn",
    );
  }
}

function pushSteps(st) {
  if (!st.steps?.length) return;
  const { xs, values } = stepsToChartData(st.steps);
  st.chart.setData(xs, values);
  updateShiftStatus(st, st.shiftVideo, st.shiftAudio);
}

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];

  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "4px";
  root.style.width = "100%";
  root.style.height = "100%";

  const chart = lineChart({
    minHeight: 150,
    xLabel: "step",
    yLabel: "σ",
    y2Label: "dσa/dσv",
    xFormat: (v) => String(Math.round(v)),
    series: [
      { id: "video", label: "σ video", color: "--cu-series-1", axis: "y" },
      { id: "audio", label: "σ audio", color: "--cu-series-3", axis: "y" },
      { id: "ratio", label: "dσa/dσv", color: "--cu-series-4", axis: "y2", dashed: true },
      { id: "carry", label: "carry σv/σa", color: "--cu-series-2", plot: false, readout: true },
    ],
    empty: {
      title: "Sigma Inspector",
      hint: "Run the node to see the sigma curve.",
    },
  });
  chart.setState("empty");

  const status = statusLine();
  root.appendChild(chart.el);
  root.appendChild(status.el);

  const st = {
    root,
    chart,
    status,
    state: "empty",
    message: "Run the node to see the sigma curve.",
    sigmasV: null,
    steps: null,
    shiftVideo: 12,
    shiftAudio: 3,
    trainedVideo: 12,
    trainedAudio: 3,
    apiOff: [],
    zoomOff: null,
    panelWidget: null,
  };
  node[ST_KEY] = st;

  st.panelWidget = mountPanel(node, "mmx_sigma_plot", root, { minHeight: PANEL_MIN });
  st.panelWidget.onPanelResize = () => chart.redraw();
  st.zoomOff = installZoomRepaint(node, () => chart.redraw(), "_c2cW3Zoom");

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
        st.state = "ready";
        chart.setState("ready");
        pushSteps(st);
      } else {
        updateShiftStatus(st, st.shiftVideo, st.shiftAudio);
      }
      return r;
    };
  };
  hookWidget("shift_video");
  hookWidget("shift_audio");

  chainOnRemoved(node, () => {
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { chart.destroy(); } catch (_e) { /* ignore */ }
    disposeState(node, ST_KEY);
  });

  return st;
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  const raw = firstUiString(output, "mmx_sigma");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.sigmas_v)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_sigma UI payload." : "No mmx_sigma in UI output.";
    st.chart.setState("error", st.message);
    return;
  }
  st.sigmasV = data.sigmas_v;
  st.trainedVideo = Number(data.trained_shift_video ?? 12);
  st.trainedAudio = Number(data.trained_shift_audio ?? 3);
  st.shiftVideo = Number(widgetByName(node, "shift_video")?.value ?? data.shift_video ?? 12);
  st.shiftAudio = Number(widgetByName(node, "shift_audio")?.value ?? data.shift_audio ?? 3);
  st.steps = recomputeSteps(st.sigmasV, st.shiftVideo, st.shiftAudio);
  st.state = "ready";
  st.chart.setState("ready");
  pushSteps(st);
}

app.registerExtension({
  name: "MiniMaxH3.W3SigmaPlot",
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
