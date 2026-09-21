# Credits

## Comfyui_Minimax_h3_latent_Upscaler (split / tiled upscaler)

- **Source:** [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) by LBH-123-AI
- **Licence:** MIT
- **Ported into this pack:** `mmx_utils/split_upscale.py`, `mmx_nodes/split_upscale.py` — direct port of the split/tiled upsampler only (not the learned 2D/3D latent upscalers, which require weights not shipped here).

## ComfyUI-MiniMax-H3-Image-Studio (still-image generation)

- **Source:** [ComfyUI-MiniMax-H3-Image-Studio](https://github.com/) (upstream vendored at `third_party/ComfyUI-MiniMax-H3-Image-Studio`)
- **Licence:** Unlicense (public domain)
- **Ported into this pack:** `mmx_utils/image_studio.py`, `mmx_nodes/image_studio.py` — H3 video model used as a still-image generator via short frame packets, decode, and frame selection. `H3SamplingSettings` and `H3WorkflowNote` were not ported (see overlap notes in pack docs). Temporal grid helpers `decoded_frames_for_latent_t` / `latent_t_for_frame_count` live in `mmx_utils/h3_grid.py`.
