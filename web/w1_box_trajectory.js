/**
 * W1 — H3 Track + Crop box trajectory overlay.
 * Reads UI payload: images (PreviewImage) + mmx_boxes JSON.
 */
import {
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  firstUiString,
  imageUrlFromUi,
  parseJsonSafe,
} from "./shared.js";
import {
  mountPanel,
  emptyState,
  installZoomRepaint,
  canvasBackingScale,
} from "./c2c_ui/index.js";

const NODE = "MiniMaxH3_TrackCrop";
const ST_KEY = "_mmxW1";
const PANEL_MIN = 300;
const NODE_MIN_W = 380;
const CHROME_H = 56;
// The preview is letterboxed ("contain"), so its box need not match the
// image: a 480x720 portrait at full node width would be ~560 px tall and push
// past the node. Height follows the aspect, clamped to this range.
const STAGE_MIN = 180;
const STAGE_MAX = 340;
function stageHeightFor(st, w) {
  const h = Math.round(Math.max(1, w) / (st.imgAspect || 16 / 9));
  return Math.max(STAGE_MIN, Math.min(STAGE_MAX, h));
}
function syncStage(st, node) {
  const w = st.stage.clientWidth || node.size?.[0] || 380;
  const h = stageHeightFor(st, w);
  if (st.stage.style.height !== `${h}px`) st.stage.style.height = `${h}px`;
  // grow the node to fit (never shrink it under the user)
  const need = node.computeSize?.();
  // classic only: Nodes 2.0 sizes to content and treats size[1] as a minimum
  if (!globalThis.LiteGraph?.vueNodesMode && need && node.size && node.size[1] < need[1]) node.setSize?.([node.size[0], need[1]]);
}

function resolveCu(root, token, fallback) {
  try {
    const prop = token.startsWith("--") ? token : `--${token}`;
    const v = getComputedStyle(root).getPropertyValue(prop).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}

function setHeader(st, kind, text) {
  st.pill.textContent = kind === "ok" ? "OK" : kind === "error" ? "ERR" : "—";
  st.pill.classList.remove("c2c-ui-chart__pill--ok", "c2c-ui-chart__pill--danger");
  if (kind === "ok") st.pill.classList.add("c2c-ui-chart__pill--ok");
  else if (kind === "error") st.pill.classList.add("c2c-ui-chart__pill--danger");
  st.summary.textContent = text || "";
}

function showOverlay(st, mode, message) {
  const overlay = st.overlay;
  overlay.innerHTML = "";
  overlay.hidden = mode === "hidden";
  if (mode === "loading") {
    const spin = document.createElement("div");
    spin.className = "c2c-ui-chart__spinner";
    spin.setAttribute("role", "status");
    spin.setAttribute("aria-label", "Loading");
    overlay.appendChild(spin);
  } else if (mode === "empty" || mode === "error") {
    overlay.appendChild(emptyState({
      title: "Box trajectory",
      hint: message || "Run the node to see the crop trajectory.",
    }));
  }
}

function setupCanvas(st, cssW, cssH) {
  const scale = canvasBackingScale(cssW, cssH);
  const bw = Math.max(1, Math.round(cssW * scale));
  const bh = Math.max(1, Math.round(cssH * scale));
  if (st.canvas.width !== bw || st.canvas.height !== bh) {
    st.canvas.width = bw;
    st.canvas.height = bh;
  }
  st.canvas.style.width = `${cssW}px`;
  st.canvas.style.height = `${cssH}px`;
  const ctx = st.canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return ctx;
}

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const cssW = Math.max(1, st.stage.clientWidth || 1);
  const cssH = Math.max(1, st.stage.clientHeight || 1);
  const ctx = setupCanvas(st, cssW, cssH);

  if (st.state === "loading") {
    showOverlay(st, "loading");
    ctx.clearRect(0, 0, cssW, cssH);
    return;
  }
  if (st.state === "error") {
    showOverlay(st, "error", st.message);
    ctx.clearRect(0, 0, cssW, cssH);
    return;
  }
  if (st.state === "empty") {
    showOverlay(st, "empty", st.message);
    ctx.clearRect(0, 0, cssW, cssH);
    return;
  }

  showOverlay(st, "hidden");

  const img = st._img || (st._img = new Image());
  const drawBoxes = () => {
    ctx.clearRect(0, 0, cssW, cssH);
    if (img.complete && img.naturalWidth > 0) {
      const scale = Math.min(cssW / img.naturalWidth, cssH / img.naturalHeight);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      const ox = (cssW - dw) / 2;
      const oy = (cssH - dh) / 2;
      ctx.drawImage(img, ox, oy, dw, dh);
      const data = st.boxes;
      if (data?.boxes?.length && data?.src_size?.length === 2) {
        const [sw, sh] = data.src_size;
        const sx = dw / sw;
        const sy = dh / sh;
        ctx.strokeStyle = resolveCu(st.root, "--cu-accent", "#b494ff");
        ctx.lineWidth = 2;
        for (const b of data.boxes) {
          if (!Array.isArray(b) || b.length < 4) continue;
          const [x, y, bw, bh] = b;
          ctx.strokeRect(ox + x * sx, oy + y * sy, bw * sx, bh * sy);
        }
      }
      const n = data?.boxes?.length || 0;
      setHeader(st, "ok", n ? `${n} box${n === 1 ? "" : "es"} on preview` : "Preview loaded");
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        const aspect = img.naturalWidth / img.naturalHeight;
        if (st.imgAspect !== aspect) {
          st.imgAspect = aspect;
          syncStage(st, node);
        }
        node.setDirtyCanvas?.(true, true);
      }
    } else {
      showOverlay(st, "empty", "Preview loading…");
    }
  };

  if (st.previewUrl && img.src !== st.previewUrl) {
    img.onload = drawBoxes;
    img.onerror = () => {
      st.state = "error";
      st.message = "Could not load preview image.";
      setHeader(st, "error", st.message);
      paintFrame(node);
    };
    img.src = st.previewUrl;
  } else {
    drawBoxes();
  }
}

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];

  const root = document.createElement("div");
  root.className = "c2c-ui";
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
  summary.textContent = "Run the node to see the crop trajectory.";
  header.appendChild(pill);
  header.appendChild(summary);

  const legend = document.createElement("div");
  legend.className = "c2c-ui-chart__chips";
  const chip = document.createElement("span");
  chip.className = "c2c-ui-chart__chip";
  chip.style.pointerEvents = "none";
  const dot = document.createElement("span");
  dot.className = "c2c-ui-chart__chip-dot";
  dot.style.background = "var(--cu-accent, #b494ff)";
  chip.appendChild(dot);
  const chipLbl = document.createElement("span");
  chipLbl.textContent = "crop box";
  chip.appendChild(chipLbl);
  legend.appendChild(chip);

  const stage = document.createElement("div");
  stage.className = "c2c-ui-stage__viewport";
  stage.style.position = "relative";
  stage.style.flex = "0 0 auto";
  stage.style.width = "100%";
  stage.style.height = `${STAGE_MIN}px`;
  stage.style.minHeight = "0";
  stage.style.background = "var(--cu-sunken, #0c0d23)";

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.width = "100%";
  canvas.style.height = "100%";

  const overlay = document.createElement("div");
  overlay.className = "c2c-ui-chart__overlay";
  overlay.style.position = "absolute";
  overlay.style.inset = "0";
  overlay.style.display = "flex";
  overlay.style.alignItems = "center";
  overlay.style.justifyContent = "center";

  stage.appendChild(canvas);
  stage.appendChild(overlay);

  root.appendChild(header);
  root.appendChild(legend);
  root.appendChild(stage);

  const st = {
    root,
    stage,
    canvas,
    overlay,
    pill,
    summary,
    state: "empty",
    message: "Run the node to see the crop trajectory.",
    boxes: null,
    previewUrl: "",
    imgAspect: 16 / 9,
    apiOff: [],
    zoomOff: null,
    resizeObs: null,
  };
  node[ST_KEY] = st;

  const panelWidget = mountPanel(node, "mmx_box_trajectory", root, { minHeight: PANEL_MIN });
  panelWidget.computeSize = (w) => {
    return [w, CHROME_H + stageHeightFor(st, w) + 8];
  };
  panelWidget.onPanelResize = () => { syncStage(st, node); paintFrame(node); };
  st.zoomOff = installZoomRepaint(node, () => paintFrame(node), "_c2cW1Zoom");

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => paintFrame(node));
    st.resizeObs.observe(stage);
  }

  st.apiOff.push(bindExecutionLifecycle(node, st, {
    onStart: () => {
      st.state = "loading";
      setHeader(st, null, "Running…");
      paintFrame(node);
    },
    onAbort: (msg) => {
      if (st.state === "loading") {
        st.state = "error";
        st.message = msg || "Execution failed.";
        setHeader(st, "error", st.message);
        paintFrame(node);
      }
    },
  }));

  chainOnRemoved(node, () => {
    try { st.zoomOff?.(); } catch (_e) { /* ignore */ }
    try { st.resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
    disposeState(node, ST_KEY);
  });

  showOverlay(st, "empty", st.message);
  paintFrame(node);
  return st;
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  const raw = firstUiString(output, "mmx_boxes");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.boxes)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_boxes UI payload." : "No mmx_boxes in UI output.";
    setHeader(st, "error", st.message);
    paintFrame(node);
    return;
  }
  st.boxes = data;
  st.previewUrl = imageUrlFromUi(output, "images") || "";
  if (!st.previewUrl) {
    st.state = "error";
    st.message = "No preview image in UI output.";
    setHeader(st, "error", st.message);
    paintFrame(node);
    return;
  }
  st.state = "success";
  st.message = "";
  paintFrame(node);
}

app.registerExtension({
  name: "MiniMaxH3.W1BoxTrajectory",
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
