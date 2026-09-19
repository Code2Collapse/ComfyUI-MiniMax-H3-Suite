/**
 * W7 — H3 Mask-Aware ControlNet gate profile.
 *
 * WHY: this node's whole job is deciding how much control reaches a row, as a
 * function of how much that row is being generated. That is a CURVE, and the
 * two widgets that define it read as numbers with no shape:
 *
 *   preserved_strength   what a fully-preserved row still receives
 *   boundary_softness    how wide the ramp between the two states is
 *
 * "0.0 and 1.0" does not tell a compositor whether their inpaint boundary will
 * show a seam. So the curve is drawn, with the preserved end and the generated
 * end labelled, and the node says in words which of the three regimes it is in.
 *
 * The one that matters most is the failure state: preserved_strength = 1.0
 * silently restores ComfyUI's behaviour, which is the bug this node exists to
 * fix. Somebody who set it to 1.0 while experimenting needs to be told, not
 * left wondering why the fix stopped working. That case is drawn in red and
 * named.
 *
 * Runtime row counts are NOT shown here. They depend on the mask actually
 * sampled with, which happens long after this node returns, and a number that
 * confident would be a guess. They go to the log instead.
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

const NODE_ID = "MiniMaxH3_MaskAwareControlNet";
const STATE = "_mmxMaskGate";
const PLOT_H = 148;

/** The gate the node actually applies, for one row's mask value in 0..1. */
function gateAt(maskValue, preservedStrength, softness) {
  // soften_rows blurs the mask spatially; on this axis the effect is a ramp of
  // width ~softness patches around the boundary, so the profile is drawn as a
  // smoothstep whose width follows the setting. The ENDS are exact: a fully
  // preserved row gets preserved_strength, a fully generated row gets 1.
  let m = maskValue;
  if (softness > 0) {
    const w = Math.min(0.9, softness / 8);
    const lo = 0.5 - w / 2;
    const t = Math.min(1, Math.max(0, (m - lo) / Math.max(w, 1e-6)));
    m = t * t * (3 - 2 * t);
  } else {
    m = m >= 0.5 ? 1 : 0;
  }
  return preservedStrength + (1 - preservedStrength) * m;
}

function regime(preserved, softness) {
  if (preserved >= 0.999) {
    return {
      tone: "error",
      title: "the fix is switched off",
      body:
        "preserved_strength is 1.0, so the mask is ignored and the control " +
        "pushes every row - including the ones the sampler is holding still. " +
        "That is ComfyUI's own behaviour, contradiction included. Lower it.",
    };
  }
  if (preserved <= 0.001) {
    return {
      tone: "ok",
      title: "control is kept out of preserved rows",
      body:
        softness > 0
          ? `Rows the mask preserves receive nothing; the ramp spans about ` +
            `${softness.toFixed(1)} latent patches so the edge does not show.`
          : "Rows the mask preserves receive nothing. The edge is hard - if a " +
            "seam shows where the control stops, raise boundary_softness.",
    };
  }
  return {
    tone: "warn",
    title: "partial hold",
    body:
      `Preserved rows keep ${Math.round(preserved * 100)}% of the control. ` +
      "The control still informs them, but no longer overrides the mask.",
  };
}

function build(node) {
  const wrap = document.createElement("div");
  Object.assign(wrap.style, {
    width: "100%",
    boxSizing: "border-box",
    padding: "4px 2px 2px",
    font: "11px system-ui, sans-serif",
    color: themeVar("--input-text") || "#ddd",
  });

  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, {
    width: "100%",
    height: `${PLOT_H}px`,
    display: "block",
    borderRadius: "4px",
  });

  const caption = document.createElement("div");
  Object.assign(caption.style, {
    padding: "4px 4px 0",
    lineHeight: "1.45",
    whiteSpace: "pre-line",
  });

  wrap.append(canvas, caption);

  const st = {
    // no raf handle here: rafThrottle owns its own, and st.dead makes a late
    // frame a no-op
    dead: false,
    el: wrap,
    canvas,
    caption,
  };
  node[STATE] = st;

  const read = (name, fallback) => {
    const w = widgetByName(node, name);
    return w ? Number(w.value) : fallback;
  };

  const paint = () => {
    if (st.dead || !canvas.isConnected || !canvas.clientWidth) return;
    // setupDpiCanvas returns the context and has ALREADY applied the
    // device-pixel transform, so everything below is in CSS pixels.
    const w = canvas.clientWidth;
    const h = PLOT_H;
    const ctx = setupDpiCanvas(canvas, w, h);
    if (!ctx) return;

    const preserved = read("preserved_strength", 0);
    const softness = read("boundary_softness", 1);
    const strength = read("strength", 1);

    const bg = themeVar("--comfy-input-bg") || "#1e1e1e";
    const grid = "rgba(255,255,255,0.10)";
    const ink = themeVar("--input-text") || "#ddd";

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // The two ends of the axis are what the user is reasoning about, so they
    // get a band each rather than a bare tick.
    const pad = 22;
    const plotW = w - pad * 2;
    const plotH = h - pad - 16;

    ctx.fillStyle = "rgba(90,120,160,0.16)";
    ctx.fillRect(pad, 6, plotW * 0.18, plotH);
    ctx.fillStyle = "rgba(120,160,90,0.16)";
    ctx.fillRect(pad + plotW * 0.82, 6, plotW * 0.18, plotH);

    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = 6 + (plotH * i) / 4;
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + plotW, y);
    }
    ctx.stroke();

    const info = regime(preserved, softness);
    const curve =
      info.tone === "error" ? "#e06c6c"
      : info.tone === "warn" ? "#e0a24a"
      : "#6fb36f";

    ctx.strokeStyle = curve;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const N = Math.max(48, Math.round(plotW));
    for (let i = 0; i < N; i++) {
      const m = i / (N - 1);
      const g = gateAt(m, preserved, softness) * Math.min(1, strength);
      const x = pad + m * plotW;
      const y = 6 + plotH - Math.min(1, Math.max(0, g)) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("preserved", pad + 2, h - 4);
    const rightLabel = "generated";
    const tw = ctx.measureText(rightLabel).width;
    ctx.fillText(rightLabel, pad + plotW - tw - 2, h - 4);
    ctx.fillText("full control", pad + 2, 14);

    caption.style.color = info.tone === "error" ? "#e06c6c" : ink;
    caption.textContent = `${info.title}\n${info.body}`;
  };

  st.paint = rafThrottle(paint);

  // The curve reads three widgets, so it repaints on any of them moving.
  for (const w of node.widgets || []) {
    const prev = w.callback;
    w.callback = function (...args) {
      const r = prev?.apply(this, args);
      st.paint();
      return r;
    };
  }

  if (typeof ResizeObserver !== "undefined") {
    st.ro = new ResizeObserver(() => st.paint());
    st.ro.observe(canvas);
  }

  addDomWidgetLast(node, "mmx_mask_gate", wrap, () => PLOT_H + 42);

  chainOnRemoved(node, () => {
    st.dead = true;
    st.ro?.disconnect();
    disposeState(node, STATE);
  });

  st.paint();
  return st;
}

app.registerExtension({
  name: "MiniMaxSuite.MaskAwareControlGate",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        build(this);
      } catch (_e) {
        /* a broken plot must never take the node down */
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      this[STATE]?.paint?.();
      return r;
    };
  },
});
