/**
 * MiniMax H3 Prompt Preview & Edit - behaviour only (no panel, so not a w*.js widget).
 *
 * WHY: the node's job is to show the prompt that reaches H3 and let you change
 * it, but the backend prefers a wired prompt and nothing put that prompt where
 * you could read it - the box stayed empty and an edit typed into it was
 * silently ignored while the wire was connected (C2C L8.21: the Family Presets
 * templates are deterministic and meant to be edited).
 *
 * Two rules, nothing else:
 *   - after a run, while use_edited_text is off, the box is filled with the
 *     prompt that was used, so it can be read and edited from there;
 *   - typing in the box turns use_edited_text on, so the edit is what runs next.
 *     Turning it off takes the wired (regenerated) prompt again.
 *
 * Plain ES module, no Vue, chained callbacks.
 */
import { app, firstUiString, widgetByName } from "./shared.js";

const NODE_ID = "MiniMaxH3_PromptEditor";
const STATE = "_mmxPromptEditor";

function install(node) {
  if (node[STATE]) return node[STATE];
  const text = widgetByName(node, "_stored_prompt");
  const flag = widgetByName(node, "use_edited_text");
  const st = { writing: false, text, flag };
  node[STATE] = st;
  if (!text || !flag) return st;

  const markEdited = () => {
    if (st.writing || flag.value) return;
    flag.value = true;
    try { flag.callback?.(true); } catch (_e) { /* ignore */ }
    node.setDirtyCanvas?.(true, true);
  };
  const prev = text.callback;
  text.callback = function (...args) {
    const r = prev?.apply(this, args);
    markEdited();
    return r;
  };
  // Classic renderer: the textarea's own input event, in case a build calls
  // the widget callback only on blur.
  const el = text.inputEl || text.element;
  if (el?.addEventListener) el.addEventListener("input", markEdited);
  return st;
}

app.registerExtension({
  name: "MiniMaxH3.PromptEditor",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { install(this); } catch (_e) { /* never take the node down */ }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = onExecuted?.apply(this, arguments);
      try {
        const st = install(this);
        const used = firstUiString(output, "prompt");
        if (st.text && st.flag && !st.flag.value && typeof used === "string" && used) {
          st.writing = true;
          try { st.text.value = used; } finally { st.writing = false; }
          this.setDirtyCanvas?.(true, true);
        }
      } catch (_e) { /* ignore */ }
      return r;
    };
  },
});
