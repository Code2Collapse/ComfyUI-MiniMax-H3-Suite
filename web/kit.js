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

import { app, api, chainOnRemoved } from "./shared.js";
import { humaniseH3Error } from "./h3_errors.js";

const PREFIX = "MiniMaxH3_";
const ST = "_mmxKit";

const STAGE_COLOR = {
  Spine: "#5a8fc8", Sampling: "#c8894a", Mask: "#4fb3a5", Crop: "#7db35a",
  Long: "#a07cc8", Face: "#c85a7d", Motion: "#d1a33a", Prompt: "#8a8a8a",
  Spectrum: "#5aa8c8", Authoring: "#8a8a8a", QC: "#d16a6a",
};

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
    "font:10px system-ui,sans-serif;color:var(--input-text,#ddd);padding:1px 2px;";

  const badge = document.createElement("span");
  if (stage) {
    badge.textContent = stage;
    badge.style.cssText =
      "flex:0 0 auto;padding:0 5px;border-radius:7px;font-size:9px;" +
      "letter-spacing:.3px;color:#111;opacity:.9;background:" +
      (STAGE_COLOR[stage] || "#666") + ";";
  }

  const status = document.createElement("span");
  status.style.cssText =
    "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.7;";

  el.append(badge, status);
  return { el, badge, status };
}

function setStatus(st, text, tone) {
  st.status.textContent = text || "";
  st.status.style.color = tone === "error" ? "#e06c6c" : "";
  st.status.style.opacity = tone === "error" ? "1" : ".7";
  st.status.title = st.fullError || "";
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
          const w = this.addDOMWidget("mmx_status", "div", ui.el, { serialize: false });
          w.computeSize = (width) => [width, 15];
          chainOnRemoved(this, () => { delete this[ST]; });
          setStatus(st, "", "");
        }
      } catch (e) { /* a kit failure must never take the node down */ }
      return r;
    };

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
