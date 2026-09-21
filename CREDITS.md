# Credits

## Comfyui_Minimax_h3_latent_Upscaler (split / tiled upscaler)

- **Source:** [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) by LBH-123-AI
- **Licence:** MIT
- **Ported into this pack:** `mmx_utils/split_upscale.py`, `mmx_nodes/split_upscale.py` — direct port of the split/tiled upsampler only (not the learned 2D/3D latent upscalers, which require weights not shipped here).

## ComfyUI-MiniMax-H3-Image-Studio (still-image generation)

- **Source:** [ComfyUI-MiniMax-H3-Image-Studio](https://github.com/) (upstream vendored at `third_party/ComfyUI-MiniMax-H3-Image-Studio`)
- **Licence:** Unlicense (public domain)
- **Ported into this pack:** `mmx_utils/image_studio.py`, `mmx_nodes/image_studio.py` — H3 video model used as a still-image generator via short frame packets, decode, and frame selection. `H3SamplingSettings` and `H3WorkflowNote` were not ported (see overlap notes in pack docs). Temporal grid helpers `decoded_frames_for_latent_t` / `latent_t_for_frame_count` live in `mmx_utils/h3_grid.py`.

## MiniMax-H3-NativeAudio-MusicVideo-Workflow (native audio lock + multishot)

- **Source:** vendored at `third_party/MiniMax-H3-NativeAudio-MusicVideo-Workflow` — sub-packs `ComfyUI-H3-NativeAudioLock` and `ComfyUI-H3-Multishot` (not `ComfyUI-Spectrum-MiniMax-H3`, already ported as `MiniMaxH3_SpectrumApply`)
- **Licence:** licence not stated by upstream (no LICENSE file, no SPDX header, no README statement) — ported at the repository owner's direction, 2026-09-21
- **Ported into this pack:** `mmx_utils/native_audio.py`, `mmx_utils/multishot.py`, `mmx_nodes/native_audio.py`, `mmx_nodes/multishot.py` — exact-audio lock (video-only denoise via nested noise mask), reference-audio stereo guard, multishot script split and chained samplers. `apply_gguf_arch_patch.py`, `h3_interior_patch.py`, loader-any wrappers, and `H3Keyframes` were not ported (see overlap notes in pack docs).


## ComfyUI-MiniMaxH3-Director (storyboard Director)

- **Source:** ComfyUI-MiniMaxH3-Director by CGlide (vendored at `third_party/ComfyUI-MiniMaxH3-Director`)
- **Licence:** **GPL-3.0**, declared in its LICENSE and README badge. This pack has been GPL-3.0 since its initial commit, so the licences agree and no relicensing was needed.
- **Ported into this pack:** `mmx_nodes/director/` — `MiniMaxH3_Director`, `MiniMaxH3_PreviewOverride`, `MiniMaxH3_RetakeStitch`, `MiniMaxH3_EnhancePrompt`, `MiniMaxH3_SaveLastFrame`. Their offline suites came too, as `tests/test_director_plan.py` (305 checks) and `tests/test_director_nodes.py` (50).
- **Not ported:** the canvas timeline front-end in `js/`. It is itself a fork of the LTX Director's canvas editor, and this workspace already has a better one — WanDirector's seven-track DOM timeline. A second canvas implementation would be the weaker of the two and would have to be maintained beside it. The Director works without it: an empty `timeline_data` falls back to `global_prompt`.
- **Also not registered:** `MiniMaxH3_DirectorChain`, exactly as upstream leaves it — the backend works, there is no usable way to give it a timeline.
- **Changed here:** node ids namespaced from the upstream `*CS` names, so installing the original alongside is a visible duplicate rather than a silent collision; and the eight HTTP routes now register through a guard instead of a bare decorator. Upstream's decorator is evaluated at import time against `PromptServer.instance`, which only exists once the server is up — so importing the module before that raised AttributeError and took all five nodes out of `/object_info`. Upstream's own test harness documents working around this by faking a server; that workaround is no longer needed.

## MiniMaxH3-Director / "Muse Minimax Director" (all-in-one chunked Director)

- **Source:** MiniMaxH3-Director (vendored at `third_party/MiniMaxH3-Director`)
- **Licence:** MIT, per the upstream README badge (no LICENSE file in the repository).
- **Ported into this pack:** `mmx_nodes/director/allinone.py` — `MiniMaxH3_DirectorAllInOne`. Kept alongside the other Director rather than instead of it: that one returns conditioning and a latent to wire into a sampler, this one runs the whole pipeline internally and **chunks**, so a video longer than H3's reliable ~15s per call is split with each continuation chunk seeded from the previous chunk's own last frames and last seconds of audio.
- **Changed here:** a `prompt` input was **added**. Upstream has no prompt socket at all — every word comes from `timeline_data`, written by its own canvas timeline, which is not ported. Without it the node would generate from an empty prompt. PyAV and the HTTP route are now imported and registered defensively for the same reason as above.
- **Registration:** it is a classic-API node, and this pack registers through the V3 `comfy_entrypoint`. It is adapted by `mmx_nodes/director/v1_adapter.py` rather than registered alongside, because ComfyUI's loader takes the V1 branch and **returns** if a pack exports `NODE_CLASS_MAPPINGS` — so exporting one mapping to register one node would have silently unregistered the other ninety-seven. `tests/test_v1_adapter.py` pins that, including a check that the pack root never grows such an export.


## comfyui-deno-custom-nodes (targeted text-encoder unload)

- **Source:** Deno Custom Nodes (vendored at `third_party/comfyui-deno-custom-nodes`)
- **Licence:** **GPL-3.0-only**, declared in its README and pyproject classifiers. This pack is GPL-3.0, so the licences agree.
- **Ported into this pack:** `mmx_utils/encoder_unload.py`, `mmx_nodes/encoder_unload.py` — `MiniMaxH3_TextEncoderUnload`. Taken because it matters more on H3 than anywhere else: the encoder is Qwen3-VL-32B, used once at the start and then resident for the whole sample, and a general "free memory" would unload the diffusion model that is about to run.
- **Not ported from that pack:** the LTX-specific nodes (this is not an LTX pack), the service-API nodes, the two RTX nodes (they need NVIDIA's Video Effects SDK, a separate Windows-only install), and its Image/Video Compare nodes — those have four modes against the sixteen in `ComfyUI-CustomNodePacks`' own `video_comparer` (scopes, false colour, bit-depth crush, audio waveform/spectrogram/loudness, synced player), so porting them would have been a downgrade. Recorded here so it is not re-audited.
