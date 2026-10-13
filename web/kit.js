// kit.js — the house UI layer applied to EVERY MiniMaxH3_* node.
//
// WHY: this pack has 62 nodes and six widgets. The six that need a bespoke
// picture have one (box trajectory, token grid, sigma plot, drift curve, crop
// plan, context windows). The other 56 do not need a picture - they need the
// same three things every node needs, so those are implemented once here:
//
//   1. H3 errors in plain English. H3's failure modes are specific and
//      unhelpful raw: "No module named 'comfy.ldm.minimax'" means the ComfyUI
//      BUILD lacks H3, not that this pack is broken; a frame count that is not
//      n % 17 == 5 fails deep inside the layout with no mention of 17. See
//      h3_errors.js.
//   2. A stage badge - Spine / Sampling / Mask / Crop / Long / Face / Motion /
//      Prompt. 62 nodes across one prefix are otherwise indistinguishable.
//   3. Run status, so you can see which pass in a long chain actually ran.
//
// Sits UNDER the bespoke widgets; it adds a row and never replaces anything.
// Plain ES module, no Vue, nothing touches `window` at import time.

import { app, api, chainOnRemoved, themeVar } from "./shared.js";
import { humaniseH3Error } from "./h3_errors.js";

const PREFIX = "MiniMaxH3_";
const ST = "_mmxKit";

// Stage hues as PALETTE keys plus the literal this pack shipped, so the badges
// move with the theme instead of being eleven fixed colours that clash with
// whatever ground ends up behind them.
const STAGE_COLOR = {
  Spine:     ["blue",     "#5a8fc8"],
  Sampling:  ["peach",    "#c8894a"],
  Mask:      ["teal",     "#4fb3a5"],
  Crop:      ["green",    "#7db35a"],
  Long:      ["mauve",    "#a07cc8"],
  Face:      ["maroon",   "#c85a7d"],
  Motion:    ["yellow",   "#d1a33a"],
  Prompt:    ["overlay2", "#8a8a8a"],
  Spectrum:  ["sky",      "#5aa8c8"],
  Authoring: ["overlay2", "#8a8a8a"],
  QC:        ["red",      "#d16a6a"],
};

function stageColor(stage) {
  const entry = STAGE_COLOR[stage];
  if (!entry) return themeVar("dim");
  return themeVar("--c2c-" + entry[0]) || entry[1];
}

function stageOf(nodeData) {
  const cat = String(nodeData?.category || "");
  const tail = cat.split("/").pop() || "";
  if (STAGE_COLOR[tail]) return tail;
  for (const k of Object.keys(STAGE_COLOR)) if (cat.includes(k)) return k;
  return "";
}

function buildStrip(stage) {
  const el = document.createElement("div");
  el.style.cssText =
    "display:flex;align-items:center;gap:6px;width:100%;box-sizing:border-box;" +
    `font:10px system-ui,sans-serif;color:${themeVar("inputText")};padding:1px 2px;min-height:14px;`;

  const badge = document.createElement("span");
  if (stage) {
    badge.textContent = stage;
    // a ghost chip: the stage hue as text and edge over a faint tint of it -
    // a solid pastel pill read as a stock tag on the night body
    const hue = stageColor(stage);
    badge.style.cssText =
      "flex:0 0 auto;padding:0 6px;border-radius:4px;font-size:9px;font-weight:600;" +
      `letter-spacing:.06em;text-transform:uppercase;line-height:12px;color:${hue};` +
      `background:color-mix(in srgb, ${hue} 14%, transparent);` +
      `border:1px solid color-mix(in srgb, ${hue} 45%, transparent);`;
  }

  const status = document.createElement("span");
  status.style.cssText =
    "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.7;";

  el.append(badge, status);
  return { el, badge, status };
}

function setStatus(st, text, tone) {
  st.status.textContent = text || "";
  st.status.style.color = tone === "error" ? themeVar("danger") : "";
  st.status.style.opacity = tone === "error" ? "1" : ".7";
  st.status.title = st.fullError || "";
}

/** The saved values with one trailing placeholder when core's legacy migration would otherwise drop entries at
 *  forceInput positions (the mask below is the one frontend 1.53's migrateWidgetsValues builds); else `vals`
 *  itself. Pure, so tests/test_kit_configure_guard.py runs it against core's function. */
export function guardWidgetValues(defs, widgetNames, vals) {
  if (!defs || !Array.isArray(vals)) return vals;
  const names = new Set(widgetNames);
  const mask = Object.values(defs).filter((d) => names.has(d.name) || d.forceInput)
    .flatMap((d) => (d.control_after_generate ? [!!d.forceInput, false] : [!!d.forceInput]));
  return mask.some(Boolean) && mask.length === vals.length ? [...vals, null] : vals;
}

app.registerExtension({
  name: "MiniMaxH3.HouseKit",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData?.name || "").startsWith(PREFIX)) return;
    const stage = stageOf(nodeData);

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        if (!this[ST]) {
          const ui = buildStrip(stage);
          const st = { ui, status: ui.status, t0: 0, fullError: "" };
          this[ST] = st;
          // ComfyUI insets a DOM widget by `margin` (default 10) on every side:
          // a 15px slot gave the strip no room and its badge spilled past the
          // node's bottom edge. 2px margin in an 18px slot fits the 14px strip.
          const w = this.addDOMWidget("mmx_status", "div", ui.el, { serialize: false, margin: 2 });
          // The option alone did not keep it out of widgets_values (saved as a trailing ""); the widget's own flag
          // is what serialize() and configure() read. Which length a save ends up with also decides whether core's
          // legacy migration fires - see the configure() guard below (C2C L8.21).
          w.serialize = false;
          w.computeSize = (width) => [width, 18];
          chainOnRemoved(this, () => { delete this[ST]; });
          setStatus(st, "", "");
        }
      } catch (e) { /* a kit failure must never take the node down */ }
      return r;
    };

    // C2C L8.21: frontend 1.53's node configure() runs a legacy migration (migrateWidgetsValues) that, when the saved
    // widgets_values has exactly as many entries as "widget inputs + forceInput inputs", drops the entries at every
    // forceInput position - meant for old saves that stored values for converted widgets. Our nodes are V3 and never
    // wrote such values, so on any node with a force_input socket (Subject Uncrop's bboxes, Detail Match, ...) a
    // modern save whose length happened to match lost its first values on load (feather 16 came back ""). One
    // placeholder at the end makes the lengths differ, so the migration leaves the values alone; configure ignores
    // the extra entry. Measured on all 108 nodes: docs/evidence/L8.21/mmx_roundtrip_*.json (C2C workspace).
    const configure = nodeType.prototype.configure;
    if (typeof configure === "function") {
      nodeType.prototype.configure = function (info, ...rest) {
        try {
          const vals = guardWidgetValues(this.constructor?.nodeData?.inputs,
                                         (this.widgets ?? []).map((w) => w.name), info?.widgets_values);
          if (vals !== info?.widgets_values) info = { ...info, widgets_values: vals };
        } catch (e) { /* never block a load */ }
        return configure.call(this, info, ...rest);
      };
    }

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = onExecuted?.apply(this, arguments);
      const st = this[ST];
      if (st) {
        const ms = st.t0 ? Date.now() - st.t0 : 0;
        st.fullError = "";
        setStatus(st, ms ? `ran in ${(ms / 1000).toFixed(2)}s` : "ran", "ok");
      }
      return r;
    };
  },

  async setup() {
    api.addEventListener("executing", ({ detail }) => {
      if (detail == null) return;
      const n = app.graph?.getNodeById?.(Number(detail));
      const st = n && n[ST];
      if (st) { st.t0 = Date.now(); st.fullError = ""; setStatus(st, "running…", ""); }
    });

    api.addEventListener("execution_error", ({ detail }) => {
      const n = app.graph?.getNodeById?.(Number(detail?.node_id));
      const st = n && n[ST];
      if (!st) return;
      const raw = [detail?.exception_message, ...(detail?.traceback || [])]
        .filter(Boolean).join("\n");
      st.fullError = raw;                    // full text on hover
      setStatus(st, humaniseH3Error(raw), "error");
      n.setDirtyCanvas(true, true);
    });

    api.addEventListener("execution_interrupted", () => {
      for (const n of app.graph?._nodes || []) {
        const st = n[ST];
        if (st && st.t0) setStatus(st, "cancelled", "");
      }
    });
  },
});
