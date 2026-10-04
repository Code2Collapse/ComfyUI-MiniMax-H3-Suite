"""Every MiniMax widget must survive node creation, an empty run and removal.

`node --check` and the static rules cannot see a runtime error, and one in
onNodeCreated makes the node impossible to add at all: that shipped once
(w5/w6 called `.el` on a button element; found by the live probe 2026-09-28).
This loads each real module in node with a stubbed DOM and ComfyUI app, then
creates, feeds and removes each node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
WEB = PACK / "web"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NODES = [
    "MiniMaxH3_TrackCrop", "MiniMaxH3_MaskPrep", "MiniMaxH3_SigmaInspector", "MiniMaxH3_DriftQC",
    "MiniMaxH3_SubjectCrop", "MiniMaxH3_SubjectCropAdvanced", "MiniMaxH3_ContextWindows",
    "MiniMaxH3_MaskAwareControlNet", "MiniMaxH3_NegPiP", "MiniMaxH3_SwapControl", "MiniMaxH3_PosePuppeteer",
]

APP_STUB = """
export const extensions = [];
export const app = {
  registerExtension(e) { extensions.push(e); },
  canvas: { ds: { scale: 1, offset: [0, 0] }, setDirty() {}, canvas: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }) } },
  graph: { setDirtyCanvas() {}, getNodeById: () => null },
  ui: { settings: { getSettingValue: () => undefined } },
};
export default app;
"""
API_STUB = "export const api = { addEventListener() {}, removeEventListener() {}, fetchApi: async () => ({ ok: false, json: async () => ({}) }) };\n"

PRELUDE = r"""
const noop = () => {};
const ctx2d = new Proxy({}, { get: (t, k) => k === "measureText" ? () => ({ width: 10 }) :
  (k === "getImageData" ? () => ({ data: new Uint8ClampedArray(4) }) : (k in t ? t[k] : noop)), set: (t, k, v) => { t[k] = v; return true; } });
class El {
  constructor(tag) { this.tagName = String(tag).toUpperCase(); this.children = []; this.style = { setProperty: noop, removeProperty: noop };
    this.dataset = {}; this.attributes = {}; this.listeners = {}; this.className = ""; this.textContent = ""; this.hidden = false;
    this.classList = { _s: new Set(), add: (...c) => c.forEach((x) => this.classList._s.add(x)), remove: (...c) => c.forEach((x) => this.classList._s.delete(x)),
      toggle: (c, on) => { const want = on === undefined ? !this.classList._s.has(c) : on; want ? this.classList._s.add(c) : this.classList._s.delete(c); return want; },
      contains: (c) => this.classList._s.has(c) };
    this.clientWidth = 360; this.clientHeight = 200; this.width = 0; this.height = 0; this.value = ""; this.parentElement = null; }
  get firstElementChild() { return this.children[0] || null; }
  get firstChild() { return this.children[0] || null; }
  get offsetHeight() { return 20; } get offsetWidth() { return 100; } get isConnected() { return true; }
  set innerHTML(v) { this.children = []; } get innerHTML() { return ""; }
  append(...c) { for (const x of c) this.appendChild(typeof x === "string" ? new El("#text") : x); }
  prepend(...c) { this.append(...c); }
  appendChild(c) { if (c) { c.parentElement = this; this.children.push(c); } return c; }
  insertBefore(c) { return this.appendChild(c); } replaceChildren(...c) { this.children = []; this.append(...c); }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); return c; } remove() {}
  replaceWith() {} cloneNode() { return new El(this.tagName); } contains() { return false; }
  setAttribute(k, v) { this.attributes[k] = String(v); } getAttribute(k) { return this.attributes[k] ?? null; }
  removeAttribute(k) { delete this.attributes[k]; } hasAttribute(k) { return k in this.attributes; }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); } removeEventListener() {} dispatchEvent() { return true; }
  getBoundingClientRect() { return { left: 0, top: 0, right: 360, bottom: 200, width: 360, height: 200, x: 0, y: 0 }; }
  getContext() { return ctx2d; } focus() {} blur() {} click() {} setPointerCapture() {} releasePointerCapture() {}
  // ".a" / ".a.b" / "tag" selectors, matched on whole class tokens like a browser
  matches(sel) { return String(sel).split(",").some((one) => { const s = one.trim();
    const [tag, ...cls] = s.split("."); const mine = new Set([...String(this.className).split(/\s+/), ...this.classList._s]);
    return (!tag || tag.toUpperCase() === this.tagName) && cls.every((c) => mine.has(c)); }); }
  querySelectorAll(sel) { const out = []; const walk = (e) => { for (const c of e.children) { if (c.matches?.(sel)) out.push(c); walk(c); } };
    walk(this); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest() { return null; }
  toDataURL() { return "data:,"; } scrollIntoView() {}
}
globalThis.window = globalThis;
globalThis.document = { createElement: (t) => new El(t), createElementNS: (_, t) => new El(t), createTextNode: () => new El("#text"),
  head: new El("head"), body: new El("body"), getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  addEventListener: noop, removeEventListener: noop, hidden: false, activeElement: null, documentElement: new El("html") };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "", display: "block" });
globalThis.requestAnimationFrame = () => 1; globalThis.cancelAnimationFrame = noop;
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
globalThis.IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
globalThis.MutationObserver = class { observe() {} disconnect() {} };
globalThis.addEventListener = noop; globalThis.removeEventListener = noop; globalThis.devicePixelRatio = 1;
globalThis.Image = class extends El { constructor() { super("img"); } };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, vueNodesMode: false };
"""

DRIVER = r"""
import "./prelude.mjs";
import { extensions } from "../scripts/app.js";
const files = JSON.parse(process.argv[2]);
const names = JSON.parse(process.argv[3]);
const problems = [];
for (const f of files) {
  try { await import("./web/" + f); } catch (e) { problems.push(`${f}: import failed: ${e.message}`); }
}
for (const name of names) {
  const proto = {};
  const nodeType = function () {}; nodeType.prototype = proto; nodeType.comfyClass = name;
  for (const ext of extensions) {
    try { await ext.beforeRegisterNodeDef?.(nodeType, { name, input: { required: {}, optional: {} }, output: [] }, {}); }
    catch (e) { problems.push(`${name}: beforeRegisterNodeDef threw: ${e.message}`); }
  }
  const node = Object.create(proto);
  Object.assign(node, { id: 7, type: name, size: [300, 200], pos: [0, 0], widgets: [], inputs: [], outputs: [], graph: {},
    properties: {}, flags: {}, imgs: [],
    addWidget(type, n, v) { const w = { type, name: n, value: v, options: {} }; this.widgets.push(w); return w; },
    addDOMWidget(n, type, el, opts) { const w = { name: n, type, element: el, options: opts || {}, value: "" }; this.widgets.push(w); return w; },
    addCustomWidget(w) { this.widgets.push(w); return w; },
    setSize(s) { this.size = s; }, computeSize() { return [300, 200]; }, setDirtyCanvas() {}, onResize: null });
  for (const w of ["shift_video", "shift_audio", "threshold_px", "preserved_strength", "boundary_softness", "strength",
                   "keypoints_json", "swap_scope", "prompt_prefix", "terms", "total_frames", "window_frames", "overlap_frames"]) {
    node.widgets.push({ name: w, value: w === "keypoints_json" ? "" : 1, options: {}, callback: null });
  }
  try { proto.onNodeCreated?.call(node); } catch (e) { problems.push(`${name}: onNodeCreated threw: ${e.stack.split("\n").slice(0, 2).join(" ")}`); continue; }
  try { proto.onExecuted?.call(node, {}); } catch (e) { problems.push(`${name}: onExecuted({}) threw: ${e.message}`); }
  try { proto.onConfigure?.call(node, {}); } catch (e) { problems.push(`${name}: onConfigure threw: ${e.message}`); }
  try { node.onRemoved?.call(node); } catch (e) { problems.push(`${name}: onRemoved threw: ${e.message}`); }
}
process.stdout.write(JSON.stringify({ problems, extensions: extensions.length }));
"""


def test_every_widget_builds_runs_empty_and_tears_down():
    files = sorted(p.name for p in WEB.glob("w*.js"))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "pack"
        shutil.copytree(WEB, root / "web")
        (Path(td) / "scripts").mkdir()
        (Path(td) / "scripts" / "app.js").write_text(APP_STUB, encoding="utf-8")
        (Path(td) / "scripts" / "api.js").write_text(API_STUB, encoding="utf-8")
        (root / "prelude.mjs").write_text(PRELUDE, encoding="utf-8")
        # web/*.js import "../../scripts/app.js" (pack/web -> td/scripts); the
        # driver in pack/ reaches the same module as "../scripts/app.js".
        (root / "run.mjs").write_text(DRIVER, encoding="utf-8")
        p = subprocess.run([NODE, str(root / "run.mjs"), json.dumps(files), json.dumps(NODES)],
                           capture_output=True, text=True, timeout=120, cwd=root)
    assert p.returncode == 0, p.stderr[-1500:]
    out = json.loads(p.stdout)
    assert out["extensions"] >= len(files) - 1, out
    assert not out["problems"], "\n".join(out["problems"])
