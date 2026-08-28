# Stage 4 — DOM widget extensions

**Frontend pin:** `comfyui_frontend_package` **1.45.21**  
(`comfyui_frontend_package-1.45.21.dist-info`)

## Mechanism

DOM widgets use `node.addDOMWidget(..., { serialize: false })` so they render on both
Vue and legacy frontends. Canvas/`node.imgs` draws do not survive on Vue.

Reference (MIT): `third_party/ComfyUI-OCIO/web/ocio_io.js:201-202`:

```js
// a DOM widget (addDOMWidget renders on Vue and legacy frontends alike;
// node.imgs / canvas draws do not on Vue)
```

Lifecycle pattern: `ocio_io.js:324-331` — `computeSize` for height, rAF/disposal
state on the node, **`onRemoved` chained** (never replaced).

Registration: `WEB_DIRECTORY = "web"` in pack `__init__.py`. ComfyUI loads every
`*.js` file under `web/` via `nodes.py:2286-2287`.

## Upstream layout bug — design around it

[Comfy-Org/ComfyUI_frontend#7942](https://github.com/Comfy-Org/ComfyUI_frontend/issues/7942)  
“Gap between DOM and built-in widgets after node resize” — still open.

**Rule:** never call `addWidget` after `addDOMWidget` on the same node. Schema
widgets exist before JS runs; we append the DOM widget **last** (see comment in
each `w*.js` file).

## Widgets

Widgets read **UI payload keys** from `onExecuted` (not socket outputs):

| File | Node | UI keys |
|------|------|---------|
| `w1_box_trajectory.js` | `MiniMaxH3_TrackCrop` | `images`, `mmx_boxes` |
| `w2_token_grid.js` | `MiniMaxH3_MaskPrep` | `mmx_pixel_mask`, `mmx_token_preview` |
| `w3_sigma_plot.js` | `MiniMaxH3_SigmaInspector` | `mmx_sigma` (+ live shift recompute) |
| `w4_drift_curve.js` | `MiniMaxH3_DriftQC` | `mmx_drift` (`{drift_px, passed}`) |

V3 only forwards `NodeOutput.ui` to the browser (`execution.py:377-381`). Socket outputs
(`boxes_json`, `sigma_json`, etc.) remain for graph wiring; widgets do not read them.

Shared helpers: `shared.js` (rAF throttle, disposal, theme vars, DPR canvas).

## Theme CSS variables (read from installed bundle — do not guess)

Source files under `comfyui_frontend_package/static/assets/`:

| Variable | File |
|----------|------|
| `--bg-color`, `--fg-color` | `groupNode-CZraG87T.css` |
| `--comfy-menu-bg`, `--comfy-input-bg` | `groupNode-CZraG87T.css` |
| `--border-color`, `--input-text` | `groupNode-CZraG87T.css` |
| `--p-primary-color` | `GraphView-hWkD98Et.css` |
| `--text-primary` | `GraphView-hWkD98Et.css` |

`shared.js` `themeVar()` reads these with hardcoded fallbacks so missing vars
cannot render invisible text.

## GPU-box verification

1. Start ComfyUI with this pack on the path; hard-refresh the browser (Ctrl+Shift+R).
2. Add each of the four nodes; confirm the DOM panel appears **below** schema widgets
   with no resize gap (issue #7942).
3. **W1** — run Track+Crop with a plate; trajectory overlays preview; resize node —
   canvas reflows.
4. **W2** — wire Region Mask → Mask Prep and run Mask Prep; left = pixel mask, right =
   token preview. Both come from Mask Prep's own UI payload, so no separate upstream run
   is needed (an earlier draft graph-walked the input link; that was removed).
5. **W3** — run Sigma Inspector once; drag `shift_video` / `shift_audio` — curve updates
   without re-queue.
6. **W4** — run Drift QC; curve + PASS/FAIL badge.

Headless capture (when ComfyUI is running):

```bash
python tools/playwright_widget_screenshot.py --url http://127.0.0.1:8188 --out %TEMP%/mmx_widgets.png
```

## Serialization

DOM widgets use `serialize: false` — not saved in workflow JSON.

UI custom keys (`mmx_*`) travel via `NodeOutput.ui` on each execution. Socket STRING
outputs are still saved when wired in the graph, but widgets ignore them.
