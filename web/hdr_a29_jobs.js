/**
 * A29 HDR jobs panel — background jobs of the HDR dataset + training nodes (Fetch Poly Haven, Fetch Hugging Face,
 * HDRI -> Clips, Make Pairs, Train IC-LoRA). Shows the job list, progress, a log tail, the training loss curve
 * (c2c_ui lineChart) and a Cancel button. Routes: GET /mmx/hdr/jobs, GET /mmx/hdr/jobs/{id}/log,
 * POST /mmx/hdr/jobs/{id}/cancel (mmx_utils/hdr_a29_jobs.py).
 *
 * Polling: one request every 1.5 s, and only while the shown job is queued / running; it stops by itself when the
 * job ends and never walks the graph. A node polls once when created (to find a job still running from earlier).
 */
import {
  app,
  api,
  chainOnRemoved,
  disposeState,
  firstUiString,
  parseJsonSafe,
} from "./shared.js";
import { lineChart, mountPanel, installZoomRepaint, button } from "./c2c_ui/index.js";

const NODES = {
  MiniMaxH3_HDRFetchPolyHaven: "fetch",
  MiniMaxH3_HDRFetchHuggingFace: "fetch",
  MiniMaxH3_HDRHDRIToClips: "clips",
  MiniMaxH3_HDRMakePairs: "pairs",
  MiniMaxH3_HDRTrainICLoRA: "train",
};
const ST_KEY = "_mmxHdrJobs";
const POLL_MS = 1500;
const ACTIVE = new Set(["queued", "running"]);
const PANEL_MIN = 120;
const PANEL_MIN_TRAIN = 300;
const NODE_MIN_W = 420;

async function getJson(path, opts) {
  const r = await api.fetchApi(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function fmtProgress(job) {
  if (!job) return "";
  const pct = job.total > 0 ? Math.round((100 * job.done) / job.total) : null;
  const part = job.total > 0 ? `${job.done}/${job.total}${pct != null ? ` (${pct} %)` : ""}` : "";
  return [part, job.message].filter(Boolean).join(" · ");
}

function setPill(st, status) {
  const pill = st.pill;
  pill.textContent = status || "no job";
  pill.classList.remove("c2c-ui-chart__pill--ok", "c2c-ui-chart__pill--danger");
  if (status === "done") pill.classList.add("c2c-ui-chart__pill--ok");
  else if (status === "error" || status === "cancelled") pill.classList.add("c2c-ui-chart__pill--danger");
}

function render(st, data) {
  const job = data?.job || null;
  st.job = job;
  setPill(st, job?.status);
  st.title.textContent = job ? job.title : "Run the node to start a job.";
  st.progressText.textContent = job ? (job.error || fmtProgress(job)) : "";
  const frac = job && job.total > 0 ? Math.min(1, job.done / job.total) : 0;
  st.bar.style.width = `${(frac * 100).toFixed(1)}%`;
  st.cancel.disabled = !(job && ACTIVE.has(job.status));
  const lines = data?.lines || [];
  const atBottom = st.log.scrollTop + st.log.clientHeight >= st.log.scrollHeight - 4;
  st.log.textContent = lines.slice(-200).join("\n");
  if (atBottom) st.log.scrollTop = st.log.scrollHeight;
  if (st.chart) {
    const loss = data?.loss || [];
    if (loss.length) {
      st.chart.setData(loss.map((p) => p[0]), { loss: loss.map((p) => p[1]) });
      st.chart.setState("ready");
    } else {
      st.chart.setState("empty");
    }
  }
}

function stopPolling(st) {
  if (st.timer) clearTimeout(st.timer);
  st.timer = 0;
}

async function refresh(st) {
  stopPolling(st);
  if (!st.jobId || st.disposed) return;
  try {
    const data = await getJson(`/mmx/hdr/jobs/${encodeURIComponent(st.jobId)}/log?tail=200`);
    render(st, data);
    if (data?.job && ACTIVE.has(data.job.status) && !st.disposed) {
      st.timer = setTimeout(() => refresh(st), POLL_MS);
    }
  } catch (e) {
    // 404 = the server restarted and forgot the job (the files on disk are kept)
    st.progressText.textContent = `Job ${st.jobId} is not known to the server any more (${e.message}).`;
    setPill(st, "");
  }
}

async function loadJobList(st) {
  try {
    const data = await getJson("/mmx/hdr/jobs");
    const mine = (data.jobs || []).filter((j) => j.kind === st.kind);
    const sel = st.select;
    sel.textContent = "";
    for (const j of mine) {
      const o = document.createElement("option");
      o.value = j.id;
      o.textContent = `${j.status} · ${j.title}`;
      sel.appendChild(o);
    }
    sel.hidden = mine.length < 2;
    if (!st.jobId && mine.length) st.jobId = mine.find((j) => ACTIVE.has(j.status))?.id || mine[0].id;
    if (st.jobId) sel.value = st.jobId;
  } catch (_e) { /* server without the routes: the panel stays empty */ }
}

function buildDom(node, kind) {
  if (node[ST_KEY]) return node[ST_KEY];
  const root = document.createElement("div");
  Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%" });

  const header = document.createElement("div");
  header.className = "c2c-ui-chart__header";
  const pill = document.createElement("span");
  pill.className = "c2c-ui-chart__pill";
  const title = document.createElement("span");
  title.className = "c2c-ui-chart__header-text";
  header.append(pill, title);

  const select = document.createElement("select");
  select.className = "c2c-ui-select";
  select.hidden = true;

  const track = document.createElement("div");
  Object.assign(track.style, { height: "6px", borderRadius: "3px", background: "var(--cu-edge)",
    overflow: "hidden" });
  const bar = document.createElement("div");
  Object.assign(bar.style, { height: "100%", width: "0%", background: "var(--cu-accent)",
    transition: "width 0.3s" });
  track.appendChild(bar);

  const progressText = document.createElement("div");
  progressText.className = "c2c-ui-chart__header-text";

  let chart = null;
  if (kind === "train") {
    chart = lineChart({
      minHeight: 140,
      xLabel: "step",
      yLabel: "loss",
      xFormat: (v) => String(Math.round(v)),
      series: [{ id: "loss", label: "loss", color: "--cu-series-1", axis: "y" }],
      empty: { title: "Loss", hint: "The curve appears once training reports its first step." },
    });
    chart.setState("empty");
  }

  const log = document.createElement("pre");
  Object.assign(log.style, { margin: "0", maxHeight: "120px", minHeight: "48px", overflow: "auto",
    fontSize: "10px", lineHeight: "1.3", color: "var(--cu-ink-soft)", whiteSpace: "pre-wrap", wordBreak: "break-all",
    background: "var(--cu-sunken)", padding: "4px", borderRadius: "4px" });

  const st = { node, kind, root, pill, title, select, bar, progressText, log, chart, jobId: null, job: null,
    timer: 0, disposed: false, cancel: null, zoomOff: null };

  st.cancel = button("Cancel job", {
    danger: true,
    disabled: true,
    onClick: async () => {
      if (!st.jobId) return;
      st.cancel.disabled = true;
      try {
        await getJson(`/mmx/hdr/jobs/${encodeURIComponent(st.jobId)}/cancel`, { method: "POST" });
      } catch (_e) { /* refresh shows the state */ }
      refresh(st);
    },
  });

  select.addEventListener("change", () => {
    st.jobId = select.value || null;
    node.properties = node.properties || {};
    node.properties.mmx_hdr_job = st.jobId;
    refresh(st);
  });

  root.append(header, select, track, progressText);
  if (chart) root.appendChild(chart.el);
  root.append(log, st.cancel);
  node[ST_KEY] = st;

  const w = mountPanel(node, "mmx_hdr_jobs", root, { minHeight: kind === "train" ? PANEL_MIN_TRAIN : PANEL_MIN });
  if (chart) {
    w.onPanelResize = () => chart.redraw();
    st.zoomOff = installZoomRepaint(node, () => chart.redraw(), "_mmxHdrJobsZoom");
  }

  chainOnRemoved(node, () => {
    st.disposed = true;
    stopPolling(st);
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { chart?.destroy(); } catch (_e) { /* ignore */ }
    disposeState(node, ST_KEY);
  });
  return st;
}

app.registerExtension({
  name: "MiniMaxH3.HDRJobs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const kind = NODES[nodeData.name];
    if (!kind) return;
    const _created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = _created?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      const st = buildDom(this, kind);
      setPill(st, "");
      // one look at the server: a job still running from before (e.g. a reloaded page)
      setTimeout(async () => {
        st.jobId = st.jobId || this.properties?.mmx_hdr_job || null;
        await loadJobList(st);
        refresh(st);
      }, 0);
      return r;
    };
    const _executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = _executed?.apply(this, arguments);
      try {
        const data = parseJsonSafe(firstUiString(output || {}, "mmx_hdr_job"));
        if (data?.job_id) {
          const st = buildDom(this, kind);
          st.jobId = data.job_id;
          this.properties = this.properties || {};
          this.properties.mmx_hdr_job = data.job_id;
          loadJobList(st).then(() => refresh(st));
        }
      } catch (_e) { /* never break the graph */ }
      return r;
    };
  },
});
