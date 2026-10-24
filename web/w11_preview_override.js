/**
 * W11 — MiniMax H3 Preview Override: the animated preview ON the node (C2C L9.39).
 *
 * The node streams an animated WebP of the whole shot while it denoises ("minimax_h3_preview" events) and, by
 * default, hides ComfyUI's own single-frame preview. Nothing listened to those events, so with the default target
 * ("node") no preview appeared anywhere. This panel shows them: the animation, the step, how many frames, the
 * playback rate and what one preview cost. Nodes inside subgraphs are found by their composite id ("12:5").
 */
import { app, api, chainOnRemoved, findNodeByExecId } from "./shared.js";
import { mountPanel } from "./c2c_ui/index.js";

const NODE_ID = "MiniMaxH3_PreviewOverride";
const EVENT = "minimax_h3_preview";
const ST = Symbol("mmxPreview");
const PANEL_H = 240;

const CSS = `
.mmx-prev{display:flex;flex-direction:column;gap:4px;height:100%;box-sizing:border-box;padding:4px;
  font:11px var(--c2c-font,system-ui,sans-serif);color:var(--c2c-fg,#ddd)}
.mmx-prev [hidden]{display:none!important}
.mmx-prev .stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;
  background:var(--c2c-bg3,#111);border-radius:4px;overflow:hidden}
.mmx-prev img{max-width:100%;max-height:100%;object-fit:contain}
.mmx-prev .empty{color:var(--c2c-gray500,#888);text-align:center;padding:10px;line-height:1.4}
.mmx-prev .cap{color:var(--c2c-gray400,#999);overflow-wrap:anywhere}
`;

function ensureCss() {
  if (document.getElementById("mmx-prev-css")) return;
  const s = document.createElement("style");
  s.id = "mmx-prev-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

function build(node) {
  if (node[ST]) return node[ST];
  ensureCss();
  const el = document.createElement("div");
  el.className = "mmx-prev";
  el.innerHTML = `<div class="stage"><div class="empty">The shot appears here while it denoises.<br>
    Wire this node's model into the sampler and queue.</div><img alt="H3 preview" hidden></div>
    <div class="cap"></div>`;
  const st = { el, img: el.querySelector("img"), empty: el.querySelector(".empty"), cap: el.querySelector(".cap") };
  mountPanel(node, "mmx_preview", el, { minHeight: PANEL_H });
  node[ST] = st;
  chainOnRemoved(node, () => { st.img.removeAttribute("src"); delete node[ST]; });
  return st;
}

function show(st, d) {
  st.img.src = `data:image/webp;base64,${d.webp}`;
  st.img.hidden = false;
  st.empty.hidden = true;
  const fps = d.playback === "source fps" ? `${d.fps} fps` : `${d.fps} fps (true speed)`;
  st.cap.textContent = `step ${d.step}/${d.total_steps} · ${d.frames} frames · ${fps} · ${d.mode} · ${d.ms} ms`;
}

app.registerExtension({
  name: "MiniMaxSuite.PreviewOverridePanel",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { build(this); } catch (_e) { /* a preview panel must never take the node down */ }
      return r;
    };
  },

  async setup() {
    api.addEventListener(EVENT, ({ detail }) => {
      if (!detail?.webp) return;
      const node = findNodeByExecId(detail.node_id);
      if (!node || node.type !== NODE_ID) return;
      try { show(build(node), detail); } catch (_e) { /* ignore a malformed event */ }
    });
  },
});
