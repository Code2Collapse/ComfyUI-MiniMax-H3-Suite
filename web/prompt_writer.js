/**
 * MiniMax H3 Prompt Writer v2 - behaviour only (no panel, so not a w*.js widget).
 *
 * WHY: with the default "template (no LLM)" backend, eight of the node's widgets (URL, model, key variable,
 * timeout, retries, temperature, tokens, on-failure) do nothing. They are shown only when an LLM backend is picked,
 * so the default node is the brief and the task, not a wall of server settings (C2C L8.24).
 *
 * Plain ES module, no Vue, chained callbacks.
 */
import { app, setWidgetVisible, vueFitNodeHeight, vueSyncNodeWidgets, widgetByName } from "./shared.js";

const NODE_ID = "MiniMaxH3_PromptWriter";
const TEMPLATE = "template (no LLM)";
const LLM_ONLY = ["base_url", "model", "api_key_env", "timeout_s", "retries", "temperature", "max_tokens", "on_failure"];

function apply(node) {
  const backend = widgetByName(node, "backend");
  if (!backend) return;
  const llm = String(backend.value) !== TEMPLATE;
  let changed = false;
  for (const name of LLM_ONLY) {
    const w = widgetByName(node, name);
    if (!w) continue;
    if ((w.__mec_hidden === true) === llm) {
      setWidgetVisible(w, llm);
      changed = true;
    }
  }
  if (!changed) return;
  vueSyncNodeWidgets(node);
  const sz = node.computeSize?.();
  if (sz) node.setSize?.([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
  vueFitNodeHeight(node);
  node.setDirtyCanvas?.(true, true);
}

function hook(node) {
  if (node.__mmxPromptWriter) return;
  node.__mmxPromptWriter = true;
  const backend = widgetByName(node, "backend");
  if (backend) {
    const prev = backend.callback;
    backend.callback = function (...args) {
      const r = prev?.apply(this, args);
      try { apply(node); } catch (_e) { /* ignore */ }
      return r;
    };
  }
  apply(node);
}

app.registerExtension({
  name: "MiniMaxH3.PromptWriter",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { hook(this); } catch (_e) { /* never take the node down */ }
      return r;
    };

    // a loaded workflow sets the saved backend after creation: re-apply once its values are in
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      try { hook(this); apply(this); } catch (_e) { /* ignore */ }
      return r;
    };
  },
});
