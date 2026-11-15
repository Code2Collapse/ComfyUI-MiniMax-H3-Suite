/**
 * H3 Latent Upscale (Model): show only the size widgets that apply (C2C L9.39; aspect inputs L9.245).
 * - size_by picks width + height, scale or megapixels.
 * - aspect only with scale / megapixels (a width / height box sets the ratio itself); custom_ratio only for aspect custom.
 * - anchor only for fit crop / pad; pad_fill only for fit pad.
 * Hidden widgets keep their values and still serialise.
 */
import { app, setWidgetVisible, vueFitNodeHeight, vueSyncNodeWidgets, widgetByName } from "./shared.js";

const NODE = "MiniMaxH3_LatentUpscaleModel";
const USES = {
  "width / height": ["width", "height"],
  scale: ["scale"],
  megapixels: ["megapixels"],
};
const ALL = ["width", "height", "scale", "megapixels", "aspect", "custom_ratio", "anchor", "pad_fill"];
const TRIGGERS = ["size_by", "aspect", "fit"];

export function visibleSizeWidgets(sizeBy, aspect, fit) {
  const show = new Set(USES[sizeBy] || USES["width / height"]);
  if (sizeBy === "scale" || sizeBy === "megapixels") {
    show.add("aspect");
    if (aspect === "custom") show.add("custom_ratio");
  }
  if (fit === "crop" || fit === "pad") show.add("anchor");
  if (fit === "pad") show.add("pad_fill");
  return show;
}

function apply(node) {
  const val = (n, d) => widgetByName(node, n)?.value ?? d;
  const show = visibleSizeWidgets(val("size_by", "width / height"), val("aspect", "source"), val("fit", "auto"));
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
  if (sz) node.setSize?.([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
  vueFitNodeHeight(node);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "MiniMaxH3.W13UpscaleSize",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      for (const name of TRIGGERS) {
        const w = widgetByName(this, name);
        if (!w || w.__mmxSizeHooked) continue;
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
