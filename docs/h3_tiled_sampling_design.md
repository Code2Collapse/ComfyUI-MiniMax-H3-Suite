# H3 tiled sampling: design (L8.37, A15 Phase 1.5)

Written before the code (A7). Two new nodes, ported from bbaudio-2025/Comfyui-MMH3-UltimateUpscale (MIT) with the
learned upscaler of LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT code, Apache-2.0 weights):

- **H3 Tiled Diffusion**: a model patch. Every model call is split into spatial tiles, and the tile predictions are
  blended into one full-frame prediction. All tiles share the same sigmas and the same noise, so there is nothing to
  freeze and no seam to repair afterwards.
- **H3 Latent Upscale (Model)**: resizes the video half of an AV latent with the H3 3D latent upscaler (or plain
  interpolation). The audio half is passed through untouched.

The existing **H3 Split Upscale** (LBH port) stays as it is. It samples each tile to completion and stitches the
results; saved workflows keep working. Tiled Diffusion is the alternative to its spatial split: connect the patched
model to Split Upscale with `spatial_split_param` unconnected, or to any H3 sampler.

## What is already known (from core 0.39.0-43, `comfy/ldm/minimax/model.py`, `comfy/model_base.py`)

- The video latent is `[1, 24, T, H/16, W/16]`; the DiT patch is 2x2 latent cells (32 px). Batch size must be 1.
- Every model call gets a `minimax_payload` with a prebuilt `PackedLayout` (`position_ids` `[S, 3]` float64,
  `segments`, `img_update`, `audio_update`, `seq_len`, `signature`). `_forward` rebuilds the layout when the signature
  `(text_len, T, Hp, Wp, audio_t)` does not match.
- Spatial positions are **area-normalised** (`_frame_grid`): a tile given its own layout would be treated as a whole
  frame. Upstream v1 had that bug; v1.1 builds each tile's layout from the full frame's rows (global positions).
- The video denoise mask reaches the model at latent resolution `[1, 1, T, H, W]`, pooled per 2x2 patch
  (`_pool_masks_to_token_grid`); the audio mask is separate (`audio_denoise_mask`).
- Core's H3 Fun ControlNet is a **model patch** (`MiniMaxH3FunControlPatch`, a DIFFUSION_MODEL wrapper). It encodes the
  control video fitted to `x[0].shape` and caches it by shape. Called per tile it would squash the whole control video
  into every tile, and same-sized tiles would share one hint. Upstream only refuses the old `c["control"]` field, so on
  core this would be silently wrong.

## Which stages tile, and how

| Input | Per tile | Why |
|---|---|---|
| video latent | cropped to the tile window | the point of tiling |
| audio latent | whole, every tile; predictions averaged | audio has no spatial axis |
| text, refs (picture / video / audio) | whole | refs have their own grid; text has none |
| keyframe latents (FL2VA) | resized to the frame grid if needed, cropped, padded to the patch | they share the target grid |
| layout positions | rows of the full-frame layout for the window | global positions (see above) |
| video denoise mask | cropped like the latent | it is at latent resolution |
| audio denoise mask | whole | |
| core Fun ControlNet patch | the full-frame hint is encoded once, then cropped per tile and handed to the patch | correct hint per tile |
| old-style `control` (a ControlNet object in the conditioning) | refused with a sentence | it cannot be cropped |

## Grid alignment

Tile sizes and overlaps are in latent cells, snapped to 2 (32 px). Tile **starts are always even**, so every tile's
rows are whole DiT patches of the full frame. Upstream placed the last tile at `H - tile`, which is odd whenever the
latent height is odd (720 p: H = 45), and the window rows then straddled patches. Here the last tile starts at the
even cell at or below `H - tile` and runs to the frame edge (one cell larger, padded inside the model).

Placement per axis: the fewest tiles whose overlap is at least the requested overlap, first and last pinned to the
edges, the rest spread evenly (MultiDiffusion placement, the rule in LTX-Enhanced `ltx_enhanced/tiles.py`).

## Blending

- `trapezoid` (default): 1 inside, linear ramps over the overlaps; the ramps exclude their end points, so two
  neighbours sum to exactly 1 and a frame edge keeps full weight (LTX-Enhanced rule).
- `gaussian`: Mixture of Diffusers bump (upstream default; sigma = tile / 8).
- `uniform`: MultiDiffusion averaging.

Accumulation is fp32 whatever the model dtype (upstream summed in the model dtype; in bf16 a weight of 3e-4 loses most
of its bits). The result is divided by the summed weights, so three tiles over one cell are still exact.

## Budgets

- VRAM: weights + one tile's activations. Tiles run one after the other inside each step (H3 runs batch 1).
- Compute: (sum of tile areas) / frame area per step, reported in the log, e.g. 1.56x for 2x2 tiles of 1024 px with
  256 px overlap on 1792 px. This node trades time for memory and seam quality; it does not make sampling faster.
- CPU: one window layout per (payload, shape), cached; at most 8 cached plans (positive + negative per run).

## Failure modes

| Case | Behaviour |
|---|---|
| model without `minimax_payload` (not H3) | ValueError naming the node |
| old-style ControlNet in the conditioning | ValueError: disconnect it or use Split Upscale |
| an unknown model patch that also wraps the diffusion model | not detectable in general; documented limitation |
| overlap >= tile | ValueError at the node |
| tile >= frame on both axes | the model is called directly: bit-identical to the unpatched model |
| kill switch `MMX_TILED_DIFFUSION=off` | the patch calls the model directly (no tiling) |
| out of memory | the torch error, with a plain sentence: lower the tile size |

## H3 Latent Upscale (Model)

- Checkpoints: `models/latent_upscale_models/*.safetensors` (core registers the folder).
  `minimax_h3_latent_upscaler_3d_conv_v1_{fp16,bf16}.safetensors` from huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
  (Apache-2.0, 345 M parameters, verified 2026-10-22). Pickle formats are refused before the file is read
  (CVE-2025-32434). The architecture is read from the state dict and loaded strictly; a checkpoint that is not a
  24-channel H3 upscaler is refused.
- Normalisation: the per-channel mean / std of the H3 VAE latent (values from the repo's `config.json`).
- The model's GroupNorms take statistics over the whole clip, and its 3x3x3 convolutions see about 70 latent steps in
  each direction, so **temporal chunking is not exact**. Default: the whole clip in one pass. `chunk_latent_steps` > 0
  processes chunks with overlap and a linear cross-fade (reported as approximate).
- `method = model` or an interpolation (`bilinear`, `bicubic`, `area`, `nearest-exact`) without weights. Upscale only;
  a smaller target is refused for the model (it was trained for scales 1-4).
- The output width / height snap to the 16 px latent grid; the report states the effective scale.
- The model goes back to the CPU after use (`offload_model`), so H3 and the upscaler are never both on the GPU.

## Tests (CPU, no weights)

- Tile plan: even starts, edges pinned, every cell covered, overlap >= requested, odd dimensions (45 x 80).
- Weights: trapezoid neighbours sum to 1; edges keep weight 1.
- Window layout against core's own `PackedLayout`: the full frame's rows for the window, keyframe segments too, audio
  rows identical, signature matching `_forward`'s check.
- Wrapper with a stand-in model: linear model (prediction = f(x) per cell) -> tiled == untiled exactly; a model that
  reads its positions -> tiles see global positions; the audio average; masks cropped; the single-tile case calls the
  model once with the original arguments; the kill switch.
- Fun ControlNet stand-in patch: each tile gets the matching crop of the full hint; the full hint is restored after.
- Latent upscaler: a random-weight `LatentResizer3D` saved as safetensors -> architecture detected, strict load, output
  shape, audio untouched, pickle refused, wrong channels refused; chunked path covers every step once.
- Real checkpoint (testbed only, not in the repo): load the fp16 file on CPU and upscale a small latent.

GPU image quality (seams, sharpness, time and VRAM per tile) is the owner's measurement; the nodes ship marked
experimental until then.
