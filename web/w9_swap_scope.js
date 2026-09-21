/**
 * W9 — H3 Swap Control: what is driving, and what is deliberately not.
 *
 * WHY: which landmarks drive a swap is the node's whole behaviour, and in a
 * list of dropdowns it is invisible. So it is drawn: each driving group lights
 * in the same colour the renderer actually uses, and changing swap_scope or
 * drive_jaw relights it.
 *
 * The jaw is the group worth seeing. It carries head POSE and the CHIN DROP
 * that lets the mouth open, as well as the dupe's skull SHAPE - so it is
 * driven by default, and turning drive_jaw off greys it out with the cost
 * spelled out in the caption rather than left to be discovered in a render.
 *
 * The face here is a SCHEMATIC, not a mean face shape. Drawing a real 68-point
 * mean face would imply a precision this diagram does not have; the groups are
 * at anatomically sensible places and that is all it claims.
 *
 * Colours mirror GROUP_COLOUR in mmx_utils/swap_control.py and are pinned by
 * tests/test_js_python_parity.py.
 *
 * Plain ES module, no Vue, rAF-throttled, chained onRemoved.
 */
import {
  addDomWidgetLast,
  app,
  chainOnRemoved,
  disposeState,
  rafThrottle,
  setupDpiCanvas,
  themeVar,
  widgetByName,
} from "./shared.js";

const NODE_ID = "MiniMaxH3_SwapControl";
const STATE = "_mmxSwapScope";
const PANEL_H = 196;

/** Mirrors GROUP_COLOUR in mmx_utils/swap_control.py. */
const GROUP_COLOUR = {
  jaw: "#8cbfff",
  pupils: "#ffffff",
  body: "#0099ff",
  feet: "#00d9bf",
  left_hand: "#ffb31a",
  right_hand: "#f2731a",
  brows: "#b373ff",
  nose: "#4de673",
  eyes: "#fff259",
  mouth: "#ff4d73",
};

/** Mirrors SWAP_SCOPES[*]["groups"] in mmx_utils/swap_regions.py. */
const SCOPE_GROUPS = {
  lips: ["mouth", "jaw"],
  face: ["jaw", "brows", "nose", "eyes", "pupils", "mouth"],
  head: ["jaw", "brows", "nose", "eyes", "pupils", "mouth"],
  body: ["body", "feet", "left_hand", "right_hand"],
  person: ["body", "feet", "left_hand", "right_hand",
           "jaw", "brows", "nose", "eyes", "pupils", "mouth"],
};

const SCOPE_NOTE = {
  lips: "Mouth and jaw drive it — the chin has to drop for the mouth to open — but only the mouth is regenerated, so a lip-sync never reshapes the chin.",
  face: "The whole face drives it: jaw for head pose, pupils for gaze, mouth for lip movement. The region is masked too, so the skull can still move toward your reference.",
  head: "As face, with hair and skull inside the mask as well.",
  body: "Body, feet and hands. The face is left to the plate.",
  person: "Everything the dupe does — body, hands, face, gaze.",
};

/** Schematic face in 0..1 box coordinates. Not a mean face — see the header. */
function facePaths() {
  return {
    jaw: [[0.16, 0.34], [0.17, 0.56], [0.26, 0.76], [0.42, 0.88],
          [0.58, 0.88], [0.74, 0.76], [0.83, 0.56], [0.84, 0.34]],
    brows: [[[0.24, 0.30], [0.32, 0.25], [0.42, 0.27]],
            [[0.58, 0.27], [0.68, 0.25], [0.76, 0.30]]],
    nose: [[0.50, 0.36], [0.50, 0.55], [0.43, 0.60], [0.50, 0.62], [0.57, 0.60]],
    eyes: [[0.26, 0.40], [0.44, 0.40]],
    mouth: [0.50, 0.73],
  };
}

function build(node) {
  disposeState(node, STATE);

  const wrap = document.createElement("div");
  wrap.style.cssText = "width:100%;box-sizing:border-box;padding:2px 2px 0;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;display:block;border-radius:4px;";
  const caption = document.createElement("div");
  caption.style.cssText =
    "font:10px system-ui,sans-serif;opacity:.78;padding:4px 2px 0;line-height:1.4;";
  wrap.append(canvas, caption);

  const st = { canvas, caption, capH: 30 };
  node[STATE] = st;

  const paint = () => {
    const scope = String(widgetByName(node, "swap_scope")?.value ?? "face");
    const driveJaw = widgetByName(node, "drive_jaw")?.value !== false;
    const active = new Set(SCOPE_GROUPS[scope] || SCOPE_GROUPS.face);
    if (!driveJaw) active.delete("jaw");
    const facey = ["brows", "nose", "eyes", "mouth"].some((g) => active.has(g));
    const jawOn2 = active.has("jaw");

    const cssW = Math.max(160, (node.size?.[0] || 360) - 24);
    const ctx = setupDpiCanvas(canvas, cssW, PANEL_H);
    const ink = themeVar("inputText") || "#ddd";
    const w = cssW, h = PANEL_H;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(0, 0, w, h);

    // face box, left; legend, right
    const size = Math.min(h - 26, w * 0.44);
    const bx = 12, by = (h - size) / 2 + 6;
    const P = (p) => [bx + p[0] * size, by + p[1] * size];
    const F = facePaths();

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // --- the jaw: lit when driven, dashed and grey when not ---
    const jawOn = active.has("jaw");
    ctx.save();
    if (!jawOn) ctx.setLineDash([3, 3]);
    ctx.strokeStyle = jawOn ? GROUP_COLOUR.jaw : "rgba(150,156,175,0.42)";
    ctx.lineWidth = jawOn ? 2.2 : 1.5;
    ctx.beginPath();
    F.jaw.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    ctx.restore();

    const stroke = (g, wdt = 2.2) => {
      ctx.strokeStyle = active.has(g) ? GROUP_COLOUR[g] : "rgba(150,156,175,0.20)";
      ctx.lineWidth = active.has(g) ? wdt : 1.2;
    };

    stroke("brows");
    for (const arc of F.brows) {
      ctx.beginPath();
      arc.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }

    stroke("nose");
    ctx.beginPath();
    F.nose.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();

    stroke("eyes");
    for (const e of F.eyes) {
      const [x, y] = P(e);
      ctx.beginPath();
      ctx.ellipse(x + size * 0.04, y, size * 0.075, size * 0.036, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    // pupils: the only gaze signal, so they get their own mark
    for (const e of F.eyes) {
      const [x, y] = P(e);
      ctx.fillStyle = active.has("pupils")
        ? GROUP_COLOUR.pupils : "rgba(150,156,175,0.22)";
      ctx.beginPath();
      ctx.arc(x + size * 0.04, y, Math.max(1.5, size * 0.014), 0, Math.PI * 2);
      ctx.fill();
    }

    stroke("mouth", 2.6);
    const [mx, my] = P(F.mouth);
    ctx.beginPath();
    ctx.ellipse(mx, my, size * 0.135, size * 0.055, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();                       // the lip line — an open mouth reads
    ctx.moveTo(mx - size * 0.135, my);
    ctx.lineTo(mx + size * 0.135, my);
    ctx.stroke();

    // jaw label: what it is doing right now, not a fixed claim
    ctx.font = "9px system-ui,sans-serif";
    ctx.textBaseline = "middle";
    const jl = jawOn ? "jaw — head pose + chin drop" : "jaw — not driven";
    ctx.fillStyle = jawOn ? "rgba(140,191,255,0.9)" : "rgba(150,156,175,0.75)";
    const jx = bx + size * 0.5 - ctx.measureText(jl).width / 2;
    const jy = by + size + 9;
    ctx.fillText(jl, jx, jy);
    if (!jawOn) {
      ctx.strokeStyle = "rgba(150,156,175,0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(jx - 2, jy);
      ctx.lineTo(jx + ctx.measureText(jl).width + 2, jy);
      ctx.stroke();
    }

    // --- legend ---
    const lx = bx + size + 18;
    let ly = by + 2;
    ctx.font = "10px system-ui,sans-serif";
    for (const g of ["jaw", "brows", "nose", "eyes", "pupils", "mouth",
                     "body", "feet", "left_hand", "right_hand"]) {
      const on = active.has(g);
      ctx.fillStyle = on ? GROUP_COLOUR[g] : "rgba(150,156,175,0.22)";
      ctx.fillRect(lx, ly - 3, 9, 6);
      ctx.fillStyle = ink;
      ctx.globalAlpha = on ? 0.92 : 0.34;
      ctx.fillText(g.replace("_", " "), lx + 14, ly);
      ctx.globalAlpha = 1;
      ly += 15;
      if (ly > h - 12) break;
    }

    caption.textContent =
      (SCOPE_NOTE[scope] || "") +
      (!facey ? " No face group is driven in this scope."
        : jawOn2
          ? " A strong jaw control pulls face width toward the dupe — lower the ControlNet strength if the head starts taking their shape."
          : " drive_jaw is off: the reference actor's skull is unopposed, but a profile will read as a front-on face and the mouth cannot open as far.");
    const capH = Math.max(16, caption.offsetHeight || 0) + 6;
    if (Math.abs(capH - st.capH) > 1) {
      st.capH = capH;
      node.setDirtyCanvas(true, true);
      node.setSize?.(node.computeSize?.() || node.size);
    }
  };

  st.paint = rafThrottle(paint);
  addDomWidgetLast(node, "mmx_swap_scope", wrap, () => PANEL_H + st.capH);

  for (const name of ["swap_scope", "drive_jaw"]) {
    const wdg = widgetByName(node, name);
    if (!wdg) continue;
    const prev = wdg.callback;
    wdg.callback = function (...a) {
      const r = prev?.apply(this, a);
      st.paint();
      return r;
    };
  }

  chainOnRemoved(node, () => disposeState(node, STATE));
  paint();
}

app.registerExtension({
  name: "MiniMaxSuite.SwapScope",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { build(this); } catch (_e) { /* a broken diagram must not kill the node */ }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...a) {
      const r = onConfigure?.apply(this, a);
      this[STATE]?.paint?.();
      return r;
    };
  },
});
