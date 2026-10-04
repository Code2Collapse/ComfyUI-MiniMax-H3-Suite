/**
 * W2 — H3 Mask Prep token grid preview (pixel mask vs latent token mask).
 * Read-only — reads mmx_pixel_mask + mmx_token_preview from UI payload.
 */
import {
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  imageUrlFromUi,
} from "./shared.js";
import {
  mountPanel,
  emptyState,
  installZoomRepaint,
  canvasBackingScale,
} from "./c2c_ui/index.js";

const NODE = "MiniMaxH3_MaskPrep";
const ST_KEY = "_mmxW2";
const PANEL_MIN = 140;
const NODE_MIN_W = 380;

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

function setupCanvas(canvas, root, cssW, cssH) {
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

function showPaneOverlay(overlay, mode, message) {
  overlay.innerHTML = "";
  overlay.hidden = mode === "hidden";
  if (mode === "loading") {
    const spin = document.createElement("div");
    spin.className = "c2c-ui-chart__spinner";
    spin.setAttribute("role", "status");
    overlay.appendChild(spin);
  } else if (mode === "empty" || mode === "error") {
    overlay.appendChild(emptyState({
      title: "Mask preview",
      hint: message || "—",
    }));
  }
}

function drawMaskCanvas(canvas, overlay, root, url, cssW, cssH) {
  const ctx = setupCanvas(canvas, root, cssW, cssH);
  if (!url) {
    showPaneOverlay(overlay, "empty", "No image");
    ctx.clearRect(0, 0, cssW, cssH);
    return;
  }
  showPaneOverlay(overlay, "hidden");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const cctx = setupCanvas(canvas, root, cssW, cssH);
    cctx.fillStyle = resolveCu(root, "--cu-sunken", "#0c0d23");
    cctx.fillRect(0, 0, cssW, cssH);
    const scale = Math.min(cssW / img.width, cssH / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    cctx.drawImage(img, (cssW - dw) / 2, (cssH - dh) / 2, dw, dh);
  };
  img.onerror = () => {
    showPaneOverlay(overlay, "error", "Could not load mask image.");
    drawPlaceholderClear(ctx, cssW, cssH);
  };
  img.src = url;
}

function drawPlaceholderClear(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
}

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const cssW = Math.max(1, st.row.clientWidth || 1);
  const cssH = Math.max(1, st.row.clientHeight || 1);
  const half = Math.floor((cssW - 4) / 2);

  if (st.state === "loading") {
    showPaneOverlay(st.leftOverlay, "loading");
    showPaneOverlay(st.rightOverlay, "loading");
    setupCanvas(st.left, st.root, half, cssH);
    setupCanvas(st.right, st.root, half, cssH);
    return;
  }
  if (st.state === "error" || st.state === "empty") {
    showPaneOverlay(st.leftOverlay, st.state, st.message);
    showPaneOverlay(st.rightOverlay, st.state, "");
    setupCanvas(st.left, st.root, half, cssH);
    setupCanvas(st.right, st.root, half, cssH);
    return;
  }

  drawMaskCanvas(st.left, st.leftOverlay, st.root, st.leftUrl, half, cssH);
  drawMaskCanvas(st.right, st.rightOverlay, st.root, st.rightUrl, half, cssH);
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
  summary.textContent = "Run the node to see pixel vs token masks.";
  header.appendChild(pill);
  header.appendChild(summary);

  const legend = document.createElement("div");
  legend.className = "c2c-ui-chart__chips";
  for (const label of ["pixel mask", "token preview (2×2 patch grid)"]) {
    const chip = document.createElement("span");
    chip.className = "c2c-ui-chart__chip";
    chip.style.pointerEvents = "none";
    const dot = document.createElement("span");
    dot.className = "c2c-ui-chart__chip-dot";
    dot.style.background = "var(--cu-accent, #b494ff)";
    chip.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.textContent = label;
    chip.appendChild(lbl);
    legend.appendChild(chip);
  }

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:4px;flex:1 1 auto;min-height:80px;position:relative;";

  const mkPane = () => {
    const pane = document.createElement("div");
    pane.style.cssText = "flex:1;min-width:0;position:relative;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%;";
    const overlay = document.createElement("div");
    overlay.className = "c2c-ui-chart__overlay";
    overlay.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;";
    pane.appendChild(canvas);
    pane.appendChild(overlay);
    return { pane, canvas, overlay };
  };

  const leftPane = mkPane();
  const rightPane = mkPane();
  row.appendChild(leftPane.pane);
  row.appendChild(rightPane.pane);

  root.appendChild(header);
  root.appendChild(legend);
  root.appendChild(row);

  const st = {
    root,
    row,
    left: leftPane.canvas,
    right: rightPane.canvas,
    leftOverlay: leftPane.overlay,
    rightOverlay: rightPane.overlay,
    pill,
    summary,
    state: "empty",
    message: "Run the node to see pixel vs token masks.",
    leftUrl: "",
    rightUrl: "",
    apiOff: [],
    zoomOff: null,
    resizeObs: null,
  };
  node[ST_KEY] = st;

  const panelWidget = mountPanel(node, "mmx_token_grid", root, { minHeight: PANEL_MIN });
  panelWidget.onPanelResize = () => paintFrame(node);
  st.zoomOff = installZoomRepaint(node, () => paintFrame(node), "_c2cW2Zoom");

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => paintFrame(node));
    st.resizeObs.observe(row);
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

  paintFrame(node);
  return st;
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  const pixel = imageUrlFromUi(output, "mmx_pixel_mask");
  const token = imageUrlFromUi(output, "mmx_token_preview");
  if (!pixel || !token) {
    st.state = "error";
    st.message = "Missing mmx_pixel_mask or mmx_token_preview in UI output.";
    setHeader(st, "error", st.message);
    paintFrame(node);
    return;
  }
  st.leftUrl = pixel;
  st.rightUrl = token;
  st.state = "success";
  st.message = "";
  setHeader(st, "ok", "Pixel mask and token preview loaded");
  paintFrame(node);
}

app.registerExtension({
  name: "MiniMaxH3.W2TokenGrid",
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
