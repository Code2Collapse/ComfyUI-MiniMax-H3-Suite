/**
 * W2 — H3 Mask Prep token grid preview (pixel mask vs latent token mask).
 * Read-only — reads mmx_pixel_mask + mmx_token_preview from UI payload.
 */
import {
  addDomWidgetLast,
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  drawLoadingSpinner,
  drawPlaceholder,
  imageUrlFromUi,
  rafThrottle,
  setupDpiCanvas,
  startLoadingLoop,
  stopLoadingLoop,
  themeVar,
} from "./shared.js";

const NODE = "MiniMaxH3_MaskPrep";
const ST_KEY = "_mmxW2";
const MIN_H = 120;

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:100%;display:flex;gap:4px;padding:2px;box-sizing:border-box;";
  const left = document.createElement("canvas");
  const right = document.createElement("canvas");
  left.style.cssText = right.style.cssText = "flex:1;height:100%;min-width:0;";
  const label = document.createElement("div");
  label.style.cssText = "position:absolute;bottom:4px;left:0;right:0;text-align:center;font:10px sans-serif;color:var(--input-text,#aaa);pointer-events:none;";
  label.textContent = "pixel mask  |  token preview (2×2 patch grid)";
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:relative;width:100%;height:100%;";
  wrap.append(left, right, label);
  box.appendChild(wrap);

  const st = {
    box: wrap,
    left,
    right,
    state: "empty",
    message: "Run the node to see pixel vs token masks.",
    leftUrl: "",
    rightUrl: "",
    raf: 0,
    apiOff: [],
  };
  node[ST_KEY] = st;

  addDomWidgetLast(node, "mmx_token_grid", box, (width) => {
    const w = Math.max(220, width || node.size?.[0] || 300);
    return Math.max(MIN_H, Math.round(w * 0.22));
  });

  const paint = () => paintFrame(node);
  const render = rafThrottle(paint);
  st.resizeObs = new ResizeObserver(render);
  st.resizeObs.observe(wrap);
  st.apiOff.push(bindExecutionLifecycle(node, st, {
    onStart: () => {
      st.state = "loading";
      startLoadingLoop(st, paint);
    },
    onAbort: (msg) => {
      stopLoadingLoop(st);
      if (st.state === "loading") {
        st.state = "error";
        st.message = msg || "Execution failed.";
        render();
      }
    },
  }));

  chainOnRemoved(node, () => disposeState(node, ST_KEY));
  render();
  return st;
}

function drawMaskCanvas(canvas, url, halfW, h) {
  const ctx = setupDpiCanvas(canvas, halfW, h);
  if (!url) {
    drawPlaceholder(ctx, halfW, h, "—");
    return;
  }
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const cctx = setupDpiCanvas(canvas, halfW, h);
    cctx.fillStyle = themeVar("menuBg");
    cctx.fillRect(0, 0, halfW, h);
    const scale = Math.min(halfW / img.width, h / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    cctx.drawImage(img, (halfW - dw) / 2, (h - dh) / 2, dw, dh);
  };
  img.onerror = () => drawPlaceholder(ctx, halfW, h, "load failed", "error");
  img.src = url;
}

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const rect = st.box.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const half = Math.floor((w - 4) / 2);

  if (st.state === "loading") {
    drawLoadingSpinner(setupDpiCanvas(st.left, half, h), half, h);
    drawLoadingSpinner(setupDpiCanvas(st.right, half, h), half, h);
    return;
  }
  if (st.state === "error" || st.state === "empty") {
    drawPlaceholder(setupDpiCanvas(st.left, half, h), half, h, st.message, st.state === "error" ? "error" : "empty");
    drawPlaceholder(setupDpiCanvas(st.right, half, h), half, h, "", st.state === "error" ? "error" : "empty");
    return;
  }

  drawMaskCanvas(st.left, st.leftUrl, half, h);
  drawMaskCanvas(st.right, st.rightUrl, half, h);
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  stopLoadingLoop(st);
  const pixel = imageUrlFromUi(output, "mmx_pixel_mask");
  const token = imageUrlFromUi(output, "mmx_token_preview");
  if (!pixel || !token) {
    st.state = "error";
    st.message = "Missing mmx_pixel_mask or mmx_token_preview in UI output.";
    paintFrame(node);
    return;
  }
  st.leftUrl = pixel;
  st.rightUrl = token;
  st.state = "success";
  st.message = "";
  paintFrame(node);
}

app.registerExtension({
  name: "MiniMaxH3.W2TokenGrid",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const _created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = _created?.apply(this, arguments);
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
