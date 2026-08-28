/**
 * W1 — H3 Track + Crop box trajectory overlay.
 * Reads UI payload: images (PreviewImage) + mmx_boxes JSON.
 */
import {
  addDomWidgetLast,
  app,
  bindExecutionLifecycle,
  chainOnRemoved,
  disposeState,
  drawLoadingSpinner,
  drawPlaceholder,
  firstUiString,
  imageUrlFromUi,
  parseJsonSafe,
  rafThrottle,
  setupDpiCanvas,
  startLoadingLoop,
  stopLoadingLoop,
  themeVar,
} from "./shared.js";

const NODE = "MiniMaxH3_TrackCrop";
const ST_KEY = "_mmxW1";
const MIN_H = 140;

function buildDom(node) {
  if (node[ST_KEY]) return node[ST_KEY];
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:100%;position:relative;overflow:hidden;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%;";
  box.appendChild(canvas);

  const st = {
    box,
    canvas,
    state: "empty",
    message: "Run the node to see the crop trajectory.",
    boxes: null,
    previewUrl: "",
    raf: 0,
    apiOff: [],
  };
  node[ST_KEY] = st;

  addDomWidgetLast(node, "mmx_box_trajectory", box, (width) => {
    const w = Math.max(200, width || node.size?.[0] || 300);
    return Math.max(MIN_H, Math.round(w * 9 / 16));
  });

  const paint = () => paintFrame(node);
  const render = rafThrottle(paint);
  st.resizeObs = new ResizeObserver(render);
  st.resizeObs.observe(box);
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

function paintFrame(node) {
  const st = node[ST_KEY];
  if (!st) return;
  const rect = st.box.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const ctx = setupDpiCanvas(st.canvas, w, h);

  if (st.state === "loading") {
    drawLoadingSpinner(ctx, w, h);
    return;
  }
  if (st.state === "error") {
    drawPlaceholder(ctx, w, h, st.message, "error");
    return;
  }
  if (st.state === "empty") {
    drawPlaceholder(ctx, w, h, st.message);
    return;
  }

  const img = st._img || (st._img = new Image());
  const drawBoxes = () => {
    ctx.clearRect(0, 0, w, h);
    if (img.complete && img.naturalWidth > 0) {
      const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      const ox = (w - dw) / 2;
      const oy = (h - dh) / 2;
      ctx.drawImage(img, ox, oy, dw, dh);
      const data = st.boxes;
      if (data?.boxes?.length && data?.src_size?.length === 2) {
        const [sw, sh] = data.src_size;
        const sx = dw / sw;
        const sy = dh / sh;
        ctx.strokeStyle = themeVar("primary");
        ctx.lineWidth = 2;
        for (const b of data.boxes) {
          if (!Array.isArray(b) || b.length < 4) continue;
          const [x, y, bw, bh] = b;
          ctx.strokeRect(ox + x * sx, oy + y * sy, bw * sx, bh * sy);
        }
      }
    } else {
      drawPlaceholder(ctx, w, h, "Preview loading…");
    }
  };

  if (st.previewUrl && img.src !== st.previewUrl) {
    img.onload = drawBoxes;
    img.onerror = () => {
      st.state = "error";
      st.message = "Could not load preview image.";
      drawBoxes();
    };
    img.src = st.previewUrl;
  } else {
    drawBoxes();
  }
}

function handleExecuted(node, output) {
  const st = buildDom(node);
  stopLoadingLoop(st);
  const raw = firstUiString(output, "mmx_boxes");
  const data = parseJsonSafe(raw);
  if (!data || !Array.isArray(data.boxes)) {
    st.state = "error";
    st.message = raw ? "Malformed mmx_boxes UI payload." : "No mmx_boxes in UI output.";
    paintFrame(node);
    return;
  }
  st.boxes = data;
  st.previewUrl = imageUrlFromUi(output, "images") || "";
  if (!st.previewUrl) {
    st.state = "error";
    st.message = "No preview image in UI output.";
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
