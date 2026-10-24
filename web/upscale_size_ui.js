/**
 * H3 Latent Upscale (Model): show only the size widgets the chosen size_by uses (C2C L9.39).
 * width / height for "width / height", scale for "scale", megapixels for "megapixels". Hidden widgets keep their
 * values and still serialise.
 */
import { app, setWidgetVisible, vueSyncNodeWidgets, widgetByName } from "./shared.js";

const NODE = "MiniMaxH3_LatentUpscaleModel";
const USES = {
  "width / height": ["width", "height"],
  scale: ["scale"],
  megapixels: ["megapixels"],
};
const ALL = ["width", "height", "scale", "megapixels"];

function apply(node) {
  const mode = widgetByName(node, "size_by")?.value ?? "width / height";
  const show = new Set(USES[mode] || USES["width / height"]);
  let changed = false;
  for (const name of ALL) {
    const w = widgetByName(node, name);
    if (!w) continue;
    const want = show.has(name);
    if (!!w.__mec_hidden === want) changed = true;
    setWidgetVisible(w, want);
  }
  if (!changed) return;
  vueSyncNodeWidgets(node);
  const sz = node.computeSize?.();
  if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "MiniMaxH3.W13UpscaleSize",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const w = widgetByName(this, "size_by");
      if (w && !w.__mmxSizeHooked) {
        w.__mmxSizeHooked = true;
        const orig = w.callback;
        w.callback = (...a) => { const rr = orig?.apply(w, a); apply(this); return rr; };
      }
      setTimeout(() => apply(this), 0);
      return r;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configure?.apply(this, arguments);
      setTimeout(() => apply(this), 0);
      return r;
    };
  },
});
