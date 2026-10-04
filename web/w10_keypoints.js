/**
 * W10 — H3 Pose Puppeteer: drag the keypoints instead of typing them.
 *
 * WHY. The node takes `keypoints_json`, which is
 *
 *     {"frames":[{"keypoints":[{"x":..,"y":..,"c":..}, ...]}, ...]}
 *
 * — a skeleton per frame. It shipped as a multiline text box. Nobody types a
 * skeleton, so the field was effectively unusable and the node fell back to
 * "keypoints from pose image only" every time, which is the node's WEAKER
 * path: the whole point of the override is correcting a joint the detector
 * put in the wrong place.
 *
 * WHAT IT DELIBERATELY IS NOT. It is not a pose DETECTOR and it does not
 * invent a skeleton. The keypoints come from a detector upstream; this edits
 * them. An editor that could also generate would be a second, worse detector
 * to maintain, and the one upstream is already better than anything drawn by
 * hand.
 *
 * THE BACKDROP IS THE POINT. Keypoint coordinates mean nothing without the
 * picture they sit on — a joint 15px off is invisible as a number and obvious
 * on a frame. The driving_pose image is drawn underneath whenever the node has
 * executed and has one.
 *
 * CONFIDENCE IS SHOWN, NOT HIDDEN. The backend keeps a per-point `c`, and a
 * low-confidence point is exactly the one worth moving. They are drawn hollow
 * so the eye goes to them, and dragging one sets it to 1.0 — because a point
 * you placed by hand IS certain, and leaving it at 0.2 tells everything
 * downstream to distrust the correction you just made.
 *
 * Plain ES module, no Vue, chained onRemoved.
 */
import {
  app,
  chainOnRemoved,
  disposeState,
  parseJsonSafe,
  widgetByName,
} from "./shared.js";
import {
  mountPanel,
  stage,
  button,
  section,
  sliderRow,
  openEditor,
  canvasBackingScale,
} from "./c2c_ui/index.js";

const NODE_ID = "MiniMaxH3_PosePuppeteer";
const STATE = "_mmxKeypoints";
const PANEL_MIN = 200;
const NODE_MIN_W = 380;
const PREVIEW_H = 140;
const HIT_PX = 9;

const LIMBS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function emptyDoc() {
  return { frames: [{ keypoints: [] }] };
}

function readDoc(text) {
  const raw = parseJsonSafe(text);
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.frames)) return emptyDoc();
  const frames = raw.frames.map((f) => {
    const kps = Array.isArray(f?.keypoints) ? f.keypoints : [];
    return {
      keypoints: kps.map((k) => {
        if (Array.isArray(k)) {
          return { x: +k[0] || 0, y: +k[1] || 0, c: k.length > 2 ? +k[2] : 1 };
        }
        return { x: +k?.x || 0, y: +k?.y || 0, c: k?.c === undefined ? 1 : +k.c };
      }),
    };
  });
  return { frames: frames.length ? frames : [{ keypoints: [] }] };
}

function writeDoc(doc) {
  return JSON.stringify({
    frames: doc.frames.map((f) => ({
      keypoints: f.keypoints.map((k) => ({
        x: +k.x.toFixed(2), y: +k.y.toFixed(2), c: +(+k.c).toFixed(3),
      })),
    })),
  });
}

function cloneDoc(doc) {
  return readDoc(writeDoc(doc));
}

function backdrop(node) {
  const img = node.imgs?.[0];
  if (img?.width) return img;
  return null;
}

/** Keypoints are PIXELS on the driving_pose image (the renderer rounds them
 *  straight to pixel positions). Without that image, the drawing area is the
 *  keypoints' own extent over ALL frames (so paging frames does not rescale),
 *  plus a margin - the rule the editor had before the c2c_ui move. */
function extentOf(doc) {
  let mx = 1;
  let my = 1;
  for (const f of doc.frames || []) {
    for (const k of f?.keypoints || []) { mx = Math.max(mx, k.x); my = Math.max(my, k.y); }
  }
  return { w: mx * 1.08, h: my * 1.08 };
}

function view(doc, frame, node, w, h) {
  const img = backdrop(node);
  let srcW = img?.width || 0;
  let srcH = img?.height || 0;
  if (!srcW || !srcH) {
    const e = extentOf(doc);
    srcW = e.w;
    srcH = e.h;
  }
  const s = Math.min(w / srcW, h / srcH);
  return { s, ox: (w - srcW * s) / 2, oy: (h - srcH * s) / 2, srcW, srcH, img };
}

function drawSkeleton(ctx, doc, frame, node, w, h, sel) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(0, 0, w, h);

  const f = doc.frames[frame];
  const pts = f?.keypoints || [];
  if (!pts.length) return { empty: true, count: 0, low: 0 };

  const v = view(doc, frame, node, w, h);
  if (v.img) {
    try { ctx.drawImage(v.img, v.ox, v.oy, v.srcW * v.s, v.srcH * v.s); }
    catch (_e) { /* not-yet-decoded image */ }
  } else {
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 1;
    ctx.strokeRect(v.ox, v.oy, v.srcW * v.s, v.srcH * v.s);
  }

  const X = (k) => v.ox + k.x * v.s;
  const Y = (k) => v.oy + k.y * v.s;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(120,190,255,0.65)";
  for (const [a, b] of LIMBS) {
    if (a >= pts.length || b >= pts.length) continue;
    if (pts[a].c <= 0.05 || pts[b].c <= 0.05) continue;
    ctx.beginPath();
    ctx.moveTo(X(pts[a]), Y(pts[a]));
    ctx.lineTo(X(pts[b]), Y(pts[b]));
    ctx.stroke();
  }

  pts.forEach((k, i) => {
    const x = X(k);
    const y = Y(k);
    const sure = k.c >= 0.5;
    const isSel = i === sel;
    ctx.beginPath();
    ctx.arc(x, y, isSel ? 6 : 4.2, 0, Math.PI * 2);
    if (sure) {
      ctx.fillStyle = isSel ? "#ffd479" : "#7ee787";
      ctx.fill();
    } else {
      ctx.strokeStyle = isSel ? "#ffd479" : "#ff8f6b";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    if (isSel) {
      ctx.strokeStyle = "rgba(255,212,121,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  const low = pts.filter((k) => k.c < 0.5).length;
  return { empty: false, count: pts.length, low, v };
}

function setupCanvas(canvas, cssW, cssH) {
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

function frameReadout(doc, frame, sel) {
  const n = doc.frames.length;
  const pts = doc.frames[frame]?.keypoints || [];
  if (!pts.length) return "—";
  const base = `frame ${frame + 1} / ${n} · ${pts.length} point${pts.length === 1 ? "" : "s"}`;
  if (sel >= 0 && pts[sel]) {
    return `${base} · point ${sel + 1} · c=${(+pts[sel].c).toFixed(2)}`;
  }
  const low = pts.filter((k) => k.c < 0.5).length;
  if (low) return `${base} · ${low} low-confidence (hollow)`;
  return base;
}

function hitTest(doc, frame, node, canvas, ev) {
  const r = canvas.getBoundingClientRect();
  const cssW = r.width;
  const cssH = r.height;
  const v = view(doc, frame, node, cssW, cssH);
  const px = ev.clientX - r.left;
  const py = ev.clientY - r.top;
  const pts = doc.frames[frame]?.keypoints || [];
  let best = -1;
  let bestD = HIT_PX * HIT_PX;
  pts.forEach((k, i) => {
    const dx = v.ox + k.x * v.s - px;
    const dy = v.oy + k.y * v.s - py;
    const d = dx * dx + dy * dy;
    if (d <= bestD) { bestD = d; best = i; }
  });
  return { idx: best, px, py, v, cssW, cssH };
}

function openKeypointEditor(node, st) {
  if (st.editorOpen) return;
  st.editorOpen = true;

  const wdg = widgetByName(node, "keypoints_json");
  let workDoc = cloneDoc(st.previewDoc);
  const originalDoc = cloneDoc(st.previewDoc);
  let frame = st.frame;
  let sel = -1;
  let drag = false;
  let dragSnapshot = null;
  const undoStack = [];
  const redoStack = [];
  let shell = null;
  let fieldUndoPending = false;

  const pushUndo = (snap) => {
    undoStack.push(snap);
    redoStack.length = 0;
    shell?.setDirty(true);
  };

  const centre = document.createElement("div");
  centre.style.cssText = "width:100%;height:100%;display:flex;align-items:center;justify-content:center;";
  const editCanvas = document.createElement("canvas");
  editCanvas.style.cssText = "display:block;max-width:100%;max-height:100%;cursor:crosshair;touch-action:none;";
  centre.appendChild(editCanvas);

  const frameLabel = document.createElement("div");
  frameLabel.className = "c2c-ui-status";
  frameLabel.style.cssText =
    "font-variant-numeric:tabular-nums;width:100%;white-space:normal;overflow-wrap:anywhere;";

  const leftBody = document.createElement("div");
  leftBody.style.display = "flex";
  leftBody.style.flexDirection = "column";
  leftBody.style.gap = "8px";

  const rightBody = document.createElement("div");
  rightBody.style.display = "flex";
  rightBody.style.flexDirection = "column";
  rightBody.style.gap = "8px";

  let xRow;
  let yRow;
  let cRow;
  let pointHint;
  let pointFields;

  function syncFields() {
    const k = sel >= 0 ? workDoc.frames[frame]?.keypoints?.[sel] : null;
    // style.display, not [hidden]: these rows carry an inline display:flex
    if (pointHint) pointHint.style.display = k ? "none" : "";
    if (pointFields) pointFields.style.display = k ? "flex" : "none";
    const img = backdrop(node);
    const ext = extentOf(workDoc);
    const maxX = img?.width || Math.ceil(ext.w * 1.5);
    const maxY = img?.height || Math.ceil(ext.h * 1.5);
    const pixels = !!img || ext.w > 2 || ext.h > 2;
    if (xRow) {
      xRow.querySelector("input[type=range]").disabled = !k;
      xRow.querySelector("input[type=number]").disabled = !k;
      xRow.querySelector("input[type=range]").max = String(maxX);
      xRow.querySelector("input[type=number]").max = String(maxX);
      xRow.querySelector("input[type=range]").step = pixels ? "0.5" : "0.001";
      xRow.querySelector("input[type=number]").step = pixels ? "0.5" : "0.001";
      if (k) {
        xRow.querySelector("input[type=range]").value = String(k.x);
        xRow.querySelector("input[type=number]").value = String(k.x);
      }
    }
    if (yRow) {
      yRow.querySelector("input[type=range]").disabled = !k;
      yRow.querySelector("input[type=number]").disabled = !k;
      yRow.querySelector("input[type=range]").max = String(maxY);
      yRow.querySelector("input[type=number]").max = String(maxY);
      yRow.querySelector("input[type=range]").step = pixels ? "0.5" : "0.001";
      yRow.querySelector("input[type=number]").step = pixels ? "0.5" : "0.001";
      if (k) {
        yRow.querySelector("input[type=range]").value = String(k.y);
        yRow.querySelector("input[type=number]").value = String(k.y);
      }
    }
    if (cRow) {
      cRow.querySelector("input[type=range]").disabled = !k;
      cRow.querySelector("input[type=number]").disabled = !k;
      if (k) {
        cRow.querySelector("input[type=range]").value = String(k.c);
        cRow.querySelector("input[type=number]").value = String(k.c);
      }
    }
  }

  function onField(field, val) {
    const k = sel >= 0 ? workDoc.frames[frame]?.keypoints?.[sel] : null;
    if (!k) return;
    if (!fieldUndoPending) {
      pushUndo(writeDoc(workDoc));
      fieldUndoPending = true;
    }
    k[field] = val;
    if (field !== "c") k.c = 1;
    repaint();
    syncFields();
  }

  function repaint() {
    const cssW = Math.max(320, centre.clientWidth || 640);
    const cssH = Math.max(240, centre.clientHeight || 480);
    const ctx = setupCanvas(editCanvas, cssW, cssH);
    const info = drawSkeleton(ctx, workDoc, frame, node, cssW, cssH, sel);
    frameLabel.textContent = frameReadout(workDoc, frame, sel);
    if (info.empty) {
      frameLabel.textContent = "No keypoints — run with a driving pose or wire keypoints in";
    }
    syncFields();
  }

  function step(d) {
    const n = workDoc.frames.length;
    if (!n) return;
    frame = (frame + d + n) % n;
    sel = -1;
    repaint();
  }

  pointHint = document.createElement("p");
  pointHint.className = "c2c-ui-status";
  pointHint.textContent = "Click a point to edit it";
  pointFields = document.createElement("div");
  pointFields.style.display = "flex";
  pointFields.style.flexDirection = "column";
  pointFields.style.gap = "8px";
  pointFields.style.display = "none";
  xRow = sliderRow("x", { min: 0, max: 1, step: 0.001, value: 0, onChange: (v) => onField("x", v) });
  yRow = sliderRow("y", { min: 0, max: 1, step: 0.001, value: 0, onChange: (v) => onField("y", v) });
  cRow = sliderRow("c", { min: 0, max: 1, step: 0.01, value: 1, onChange: (v) => onField("c", v) });
  pointFields.append(xRow, yRow, cRow);
  rightBody.appendChild(section("Selected point", pointHint, pointFields));

  const navRow = document.createElement("div");
  navRow.style.display = "flex";
  navRow.style.gap = "6px";
  navRow.style.alignItems = "center";
  const prevBtn = button("‹", { onClick: () => step(-1) });
  const nextBtn = button("›", { onClick: () => step(1) });
  navRow.append(prevBtn, nextBtn);
  leftBody.appendChild(navRow);
  leftBody.appendChild(frameLabel);

  const copyPrev = button("Copy from previous frame", {
    onClick: () => {
      if (frame <= 0) return;
      pushUndo(writeDoc(workDoc));
      workDoc.frames[frame].keypoints = workDoc.frames[frame - 1].keypoints.map((k) => ({ ...k }));
      sel = -1;
      syncFields();
      repaint();
    },
  });
  const resetFrame = button("Reset frame", {
    onClick: () => {
      pushUndo(writeDoc(workDoc));
      workDoc.frames[frame].keypoints = cloneDoc({
        frames: [{ keypoints: originalDoc.frames[frame]?.keypoints || [] }],
      }).frames[0].keypoints.map((k) => ({ ...k }));
      sel = -1;
      syncFields();
      repaint();
    },
  });
  leftBody.append(copyPrev, resetFrame);

  const onDown = (ev) => {
    fieldUndoPending = false;
    const { idx } = hitTest(workDoc, frame, node, editCanvas, ev);
    sel = idx;
    drag = idx >= 0;
    if (drag) {
      dragSnapshot = writeDoc(workDoc);
      editCanvas.setPointerCapture?.(ev.pointerId);
    }
    repaint();
    ev.preventDefault();
  };

  const onMove = (ev) => {
    if (!drag || sel < 0) return;
    const { v, cssW, cssH } = hitTest(workDoc, frame, node, editCanvas, ev);
    const k = workDoc.frames[frame].keypoints[sel];
    if (!k) return;
    const r = editCanvas.getBoundingClientRect();
    k.x = (ev.clientX - r.left - v.ox) / v.s;
    k.y = (ev.clientY - r.top - v.oy) / v.s;
    k.c = 1;
    repaint();
    ev.preventDefault();
  };

  const onUp = () => {
    if (!drag) return;
    drag = false;
    if (dragSnapshot) {
      pushUndo(dragSnapshot);
      dragSnapshot = null;
    }
    repaint();
  };

  editCanvas.addEventListener("pointerdown", onDown);
  editCanvas.addEventListener("pointermove", onMove);
  editCanvas.addEventListener("pointerup", onUp);
  editCanvas.addEventListener("pointercancel", onUp);
  editCanvas.addEventListener("lostpointercapture", onUp);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);

  let resizeObs = null;
  if (typeof ResizeObserver !== "undefined") {
    resizeObs = new ResizeObserver(() => repaint());
    resizeObs.observe(centre);
  }

  shell = openEditor({
    title: "Keypoint editor",
    left: section("Frames", leftBody),
    right: rightBody,
    centre,
    hints: [
      "Drag a point to move it",
      "Hollow points are low confidence",
      "Ctrl+Z undo",
    ],
    onUndo: () => {
      if (!undoStack.length) return;
      redoStack.push(writeDoc(workDoc));
      workDoc = readDoc(undoStack.pop());
      sel = -1;
      repaint();
      syncFields();   // the right panel must drop the undone point's values
    },
    onRedo: () => {
      if (!redoStack.length) return;
      undoStack.push(writeDoc(workDoc));
      workDoc = readDoc(redoStack.pop());
      sel = -1;
      repaint();
      syncFields();
    },
    onSave: () => {
      if (!wdg) return false;
      wdg.value = writeDoc(workDoc);
      wdg.callback?.(wdg.value);
      node.graph?.setDirtyCanvas?.(true, false);
      st.previewDoc = cloneDoc(workDoc);
      st.originalDoc = cloneDoc(workDoc);
      st.frame = frame;
      paintPreview(st, node);
      return true;
    },
    onClose: () => {
      st.editorOpen = false;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      editCanvas.removeEventListener("pointerdown", onDown);
      editCanvas.removeEventListener("pointermove", onMove);
      editCanvas.removeEventListener("pointerup", onUp);
      editCanvas.removeEventListener("pointercancel", onUp);
      editCanvas.removeEventListener("lostpointercapture", onUp);
      try { resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
      shell = null;
    },
  });

  requestAnimationFrame(() => repaint());
}

function paintPreview(st, node) {
  const canvas = st.previewCanvas;
  if (!canvas) return;
  const cssW = Math.max(1, canvas.parentElement?.clientWidth || node.size?.[0] - 40 || 300);
  const cssH = PREVIEW_H;
  const ctx = setupCanvas(canvas, cssW, cssH);
  const info = drawSkeleton(ctx, st.previewDoc, st.frame, node, cssW, cssH, -1);
  if (info.empty) {
    st.stageApi.setEmpty({
      title: "Pose Puppeteer",
      hint: "Run once with a driving pose, or wire keypoints in — this edits a skeleton, it does not invent one.",
    });
  } else {
    st.stageApi.setCanvas(canvas);
    st.stageApi.setFooter(frameReadout(st.previewDoc, st.frame, -1));
  }
}

function loadFromWidget(st, node) {
  const w = widgetByName(node, "keypoints_json");
  st.previewDoc = readDoc(w?.value);
  st.originalDoc = cloneDoc(st.previewDoc);
  st.frame = Math.min(st.frame, Math.max(0, st.previewDoc.frames.length - 1));
}

/** The editor authors keypoints_json, so its raw text box only clutters the
 *  node (same rule as Paint's canvas_data). The widget stays: it still
 *  serialises with the workflow and its input socket still takes a link. */
function hideRawJson(node) {
  const w = widgetByName(node, "keypoints_json");
  if (!w) return;
  if (!w.options) w.options = {};
  w.options.hidden = true;
  w.hidden = true;
  w.computeSize = () => [0, -4];
  if (w.element) w.element.style.display = "none";
}

function build(node) {
  disposeState(node, STATE);
  hideRawJson(node);

  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "8px";
  root.style.width = "100%";

  const previewCanvas = document.createElement("canvas");
  previewCanvas.style.display = "block";

  const stageApi = stage({
    aspect: 16 / 9,
    empty: {
      title: "Pose Puppeteer",
      hint: "Open the editor to adjust keypoints on the driving pose",
    },
  });

  const st = {
    previewCanvas,
    stageApi,
    previewDoc: emptyDoc(),
    originalDoc: emptyDoc(),
    frame: 0,
    editorOpen: false,
    resizeObs: null,
  };
  node[STATE] = st;

  const openBtn = button("Open keypoint editor", {
    primary: true,
    block: true,
    onClick: () => openKeypointEditor(node, st),
  });

  root.appendChild(openBtn);
  root.appendChild(stageApi.el);

  stageApi.el.querySelector(".c2c-ui-stage__viewport").style.cursor = "pointer";
  stageApi.el.querySelector(".c2c-ui-stage__viewport").addEventListener("click", () => {
    openKeypointEditor(node, st);
  });

  mountPanel(node, "mmx_keypoints", root, { minHeight: PANEL_MIN });

  loadFromWidget(st, node);

  if (typeof ResizeObserver !== "undefined") {
    st.resizeObs = new ResizeObserver(() => paintPreview(st, node));
    st.resizeObs.observe(stageApi.el);
  }

  chainOnRemoved(node, () => {
    try { st.resizeObs?.disconnect(); } catch (_e) { /* ignore */ }
    disposeState(node, STATE);
  });

  paintPreview(st, node);
}

app.registerExtension({
  name: "MiniMaxSuite.PoseKeypoints",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
      try { build(this); } catch (_e) { /* a broken editor must not kill the node */ }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...a) {
      const r = onConfigure?.apply(this, a);
      const st = this[STATE];
      if (st) {
        st.frame = 0;
        loadFromWidget(st, this);
        paintPreview(st, this);
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (...a) {
      const r = onExecuted?.apply(this, a);
      const st = this[STATE];
      if (st) paintPreview(st, this);
      return r;
    };
  },
});
