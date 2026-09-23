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
 * Plain ES module, no Vue, rAF-throttled, chained onRemoved.
 */
import {
  addDomWidgetLast,
  app,
  chainOnRemoved,
  disposeState,
  drawPlaceholder,
  parseJsonSafe,
  rafThrottle,
  setupDpiCanvas,
  themeVar,
  widgetByName,
} from "./shared.js";

const NODE_ID = "MiniMaxH3_PosePuppeteer";
const STATE = "_mmxKeypoints";
const PANEL_H = 300;
const HIT_PX = 9;

/** COCO-ish limb pairs. Drawn only when both ends exist, so a 5-point hand or
 *  a 133-point wholebody both render without a per-layout branch. */
const LIMBS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function emptyDoc() {
  return { frames: [{ keypoints: [] }] };
}

/** The document, however it was stored. Arrays [x,y,c] are accepted because
 *  the backend accepts them, and a file that loads in one and not the other
 *  would be a trap. */
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

function build(node) {
  disposeState(node, STATE);

  const wrap = document.createElement("div");
  wrap.style.cssText = "width:100%;box-sizing:border-box;padding:2px 2px 0;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "width:100%;display:block;border-radius:4px;cursor:crosshair;touch-action:none;";

  const bar = document.createElement("div");
  bar.style.cssText =
    "display:flex;gap:6px;align-items:center;padding:5px 2px 0;" +
    "font:11px system-ui,sans-serif;color:#9aa4b2;flex-wrap:wrap;";

  const mkBtn = (label, title) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.style.cssText =
      "font:11px system-ui,sans-serif;background:#23272e;color:#c9d1d9;" +
      "border:1px solid #3b424d;border-radius:4px;padding:3px 7px;cursor:pointer;";
    return b;
  };
  const prev = mkBtn("‹", "Previous frame");
  const next = mkBtn("›", "Next frame");
  const label = document.createElement("span");
  label.style.cssText = "font:11px ui-monospace,monospace;color:#c9d1d9;min-width:88px;";
  const reset = mkBtn("Reset point", "Put the selected point back where the detector had it");
  const note = document.createElement("span");
  note.style.cssText = "flex:1;text-align:right;opacity:.8;";
  bar.append(prev, next, label, reset, note);
  wrap.append(canvas, bar);

  const st = {
    canvas, wrap, frame: 0, sel: -1, drag: false,
    doc: emptyDoc(), original: emptyDoc(), img: null,
  };
  node[STATE] = st;

  const W = () => widgetByName(node, "keypoints_json");

  function load() {
    const w = W();
    st.doc = readDoc(w?.value);
    // A pristine copy so "reset point" means the DETECTOR's position, not
    // wherever this session last left it.
    st.original = readDoc(w?.value);
    st.frame = Math.min(st.frame, st.doc.frames.length - 1);
    if (st.frame < 0) st.frame = 0;
  }

  function save() {
    const w = W();
    if (!w) return;
    w.value = writeDoc(st.doc);
    w.callback?.(w.value);
    node.graph?.setDirtyCanvas?.(true, false);
  }

  /** The node's own executed output, if it has one, as a backdrop. */
  function backdrop() {
    const img = node.imgs?.[0];
    if (img?.width) return img;
    return null;
  }

  /** Keypoints are in SOURCE pixels; the canvas is whatever width the node
   *  happens to be. One scale for both axes, letterboxed, so a correction
   *  made here lands where it looks like it lands. */
  function view(w, h) {
    const f = st.doc.frames[st.frame];
    const img = backdrop();
    let srcW = img?.width || 0, srcH = img?.height || 0;
    if (!srcW || !srcH) {
      let mx = 1, my = 1;
      for (const k of f?.keypoints || []) { mx = Math.max(mx, k.x); my = Math.max(my, k.y); }
      srcW = mx * 1.08; srcH = my * 1.08;
    }
    const s = Math.min(w / srcW, h / srcH);
    return { s, ox: (w - srcW * s) / 2, oy: (h - srcH * s) / 2, srcW, srcH, img };
  }

  const paint = () => {
    const cssW = Math.max(200, (node.size?.[0] || 380) - 24);
    const ctx = setupDpiCanvas(canvas, cssW, PANEL_H);
    const w = cssW, h = PANEL_H;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(0, 0, w, h);

    const f = st.doc.frames[st.frame];
    const pts = f?.keypoints || [];
    if (!pts.length) {
      drawPlaceholder(ctx, w, h,
        "No keypoints yet. Run the node once with a driving pose, or wire "
        + "keypoints in — this edits a skeleton, it does not invent one.",
        "empty");
      label.textContent = "—";
      note.textContent = "";
      return;
    }

    const v = view(w, h);
    if (v.img) {
      try { ctx.drawImage(v.img, v.ox, v.oy, v.srcW * v.s, v.srcH * v.s); }
      catch (_e) { /* a not-yet-decoded image must not stop the skeleton */ }
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
      const x = X(k), y = Y(k);
      const sure = k.c >= 0.5;
      const isSel = i === st.sel;
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 6 : 4.2, 0, Math.PI * 2);
      if (sure) {
        ctx.fillStyle = isSel ? "#ffd479" : "#7ee787";
        ctx.fill();
      } else {
        // Hollow on purpose: a low-confidence point is the one worth moving,
        // so the eye should land on it.
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

    label.textContent = `frame ${st.frame + 1}/${st.doc.frames.length}`;
    const low = pts.filter((k) => k.c < 0.5).length;
    note.textContent = st.sel >= 0
      ? `point ${st.sel + 1} · c=${(+pts[st.sel].c).toFixed(2)}`
      : (low ? `${low} low-confidence point${low === 1 ? "" : "s"} (hollow)` : `${pts.length} points`);
  };

  st.paint = rafThrottle(paint);

  function hit(ev) {
    const r = canvas.getBoundingClientRect();
    const v = view(r.width, PANEL_H);
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const pts = st.doc.frames[st.frame]?.keypoints || [];
    let best = -1, bestD = HIT_PX * HIT_PX;
    pts.forEach((k, i) => {
      const dx = v.ox + k.x * v.s - px, dy = v.oy + k.y * v.s - py;
      const d = dx * dx + dy * dy;
      if (d <= bestD) { bestD = d; best = i; }
    });
    return { idx: best, px, py, v };
  }

  canvas.addEventListener("pointerdown", (ev) => {
    const { idx } = hit(ev);
    st.sel = idx;
    st.drag = idx >= 0;
    if (st.drag) canvas.setPointerCapture?.(ev.pointerId);
    st.paint();
    ev.preventDefault();
  });

  const move = (ev) => {
    if (!st.drag || st.sel < 0) return;
    const r = canvas.getBoundingClientRect();
    if (!r.width) return;
    const v = view(r.width, PANEL_H);
    const k = st.doc.frames[st.frame].keypoints[st.sel];
    if (!k) return;
    k.x = (ev.clientX - r.left - v.ox) / v.s;
    k.y = (ev.clientY - r.top - v.oy) / v.s;
    // A point you placed by hand IS certain. Leaving it at 0.2 tells
    // everything downstream to distrust the correction just made.
    k.c = 1;
    st.paint();
    ev.preventDefault();
  };
  const up = () => {
    if (!st.drag) return;
    st.drag = false;
    save();                       // one undo step per drag, not one per pixel
    st.paint();
  };
  // On the window: a fast drag leaves the canvas and the point would stick to
  // the cursor after the button came up.
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
  st._detach = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    window.removeEventListener("pointercancel", up);
  };

  const step = (d) => {
    const n = st.doc.frames.length;
    if (!n) return;
    st.frame = (st.frame + d + n) % n;
    st.sel = -1;
    st.paint();
  };
  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  reset.addEventListener("click", () => {
    if (st.sel < 0) return;
    const o = st.original.frames[st.frame]?.keypoints?.[st.sel];
    const k = st.doc.frames[st.frame]?.keypoints?.[st.sel];
    if (!o || !k) return;
    k.x = o.x; k.y = o.y; k.c = o.c;
    save();
    st.paint();
  });

  addDomWidgetLast(node, "mmx_keypoints", wrap, () => PANEL_H + 30);
  chainOnRemoved(node, () => {
    try { st._detach?.(); } catch (_e) { /* removal must not throw */ }
    disposeState(node, STATE);
  });

  load();
  paint();
}

app.registerExtension({
  name: "MiniMaxSuite.PoseKeypoints",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { build(this); } catch (_e) { /* a broken editor must not kill the node */ }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...a) {
      const r = onConfigure?.apply(this, a);
      const st = this[STATE];
      if (st) { st.frame = 0; st.sel = -1; }
      this[STATE]?.paint?.();
      return r;
    };

    // After a run the node has a driving-pose image; redraw so the skeleton
    // lands on the picture rather than on nothing.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (...a) {
      const r = onExecuted?.apply(this, a);
      this[STATE]?.paint?.();
      return r;
    };
  },
});
