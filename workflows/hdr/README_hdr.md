# MiniMax H3 HDR workflows (A29)

All five use the Suite's **MiniMax H3/HDR** nodes plus ComfyUI core nodes (core 0.39+). Working space: scene-linear
Rec.709, 1.0 = diffuse white; log encoding `diffhdr_log_16384`. Nothing is clamped after **H3 HDR Log Decode**.

| File | What it does |
|---|---|
| `a_sdr_to_hdr_exr.json` | SDR clip -> Prepare Draft (24 fps, 1344x768, 17k+5 frames) -> H3 Reference to Video with the SDR clip as `<Video 1>` + the HDR IC-LoRA -> VAE Decode -> HDR Log Decode -> HDR Bias Correct -> keep the real frames -> Save EXR Sequence (half, ZIP) + Tonemap Preview. |
| `b_sdr_to_hdr_regen4k.json` | (a) plus the Suite's Open Regenerate STD at **4k** on the same SDR frames -> HDR Gain Map Join (draft HDR light x 4K detail) -> 4K Save EXR Sequence + preview; the draft-size EXR is saved too. |
| `c_dataset_build.json` | HDR Dataset: Fetch Poly Haven (20 CC0 HDRIs, 2k .hdr) -> HDRI -> Clips (2 seeded pan / tilt / push moves per HDRI, 73 frames, 1344x768, half EXR, median = 18 % grey) -> Make Pairs (`pairs/control/<clip>.mp4` SDR reference, `pairs/targets/<clip>.mp4` log target + `.txt` caption, `pairs/meta.jsonl`). Folders are under `output/hdr_dataset/`. |
| `d_train.json` | HDR Train IC-LoRA on `output/hdr_dataset/pairs` with ai-toolkit (ref2va, rank / alpha 16, qkv_proj / out_proj / fc1 / fc2). Ships with **dry_run on**: the first run only checks and writes the YAML; turn it off to train. |
| `e_eval.json` | Load EXR Sequence (ground truth) + Load EXR Sequence (decoded pipeline output) -> HDR Eval Report (stop error per luminance band). |

## Model files (names as in the widgets; pick yours from the dropdowns)

Hugging Face repo ids are **UNVERIFIED** here - check each download page before fetching.

| Widget | File | Folder |
|---|---|---|
| UNETLoader (a, b top) | `minimax_h3_ref2va_bf16.safetensors` (H3 Ref2VA model) | `models/diffusion_models` |
| LoraLoaderModelOnly | `h3_hdr_a29_iclora_diffhdr16384.safetensors` - the HDR IC-LoRA you train with node 16 (HDR Train IC-LoRA); no public file | `models/loras` |
| CLIPLoader (type minimax) | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` |
| VAELoader (video) | `minimax_h3_video_vae_fp16.safetensors` (fp16 is fine; Bias Correct handles its known bias) | `models/vae` |
| UNETLoader (b, Regenerate) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models` |
| H3 Open Regen LoRA | `open_h3_regenerate_2k_lora.safetensors` (+ `sharp_curve_branches.safetensors`) | `models/loras` |
| VAELoader (audio, b) | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` |
| H3 Open Regen Upscale | `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `models/latent_upscale_models` |
| H3 Open Regen Fast VAE Decode | `fast_h3_vae_noncausal_24l.safetensors` | `models/vae` |

Inputs: put the SDR clip in ComfyUI's `input/` folder (Load Video). Workflow (e) reads EXR folders relative to `input/`
then `output/` (or absolute paths). Outputs land under `output/hdr/` (a new numbered folder per run).

Status: the workflows are wire-checked against the registered node definitions on CPU (tests/test_hdr_a29_workflows.py).
They have NOT been run with real weights: HDR quality depends on the trained LoRA and is not verified.

## Dataset and training (c, d)

- **Background jobs.** Fetch, HDRI -> Clips, Make Pairs and Train run as background jobs. Each of these nodes has a
  jobs panel that shows the job, its progress, the end of its log and a **Cancel** button; the training node also
  shows the loss curve. With **wait** on, the node waits for the job and ComfyUI's Cancel button stops it. With
  **wait** off, the node returns at once and the job keeps running. Training defaults to wait off.
- **Re-runs skip finished work.** HDRIs, clips and pairs that are already done are skipped. An interrupted download
  resumes from its `.part` file.
- **Poly Haven** downloads use the public API. A file is checked against the size and md5 the API gives.
  **Hugging Face**: the total size of the matching files is checked before anything downloads, and the fetch is
  refused above `cap_gb`. Private repos need the `HF_TOKEN` environment variable.
- **Pairs.** For each clip the SDR control gets a random exposure shift (±1.5 EV). Half of the clipped controls are
  hard clipped and half get a soft roll-off; 15 % of clips are left unclipped. The control is sRGB, 8-bit h264, crf
  16-28. The target is `diffhdr_log_16384` as lossless RGB. The log code is stored as 8-bit values, because
  ai-toolkit reads video as 8-bit; `meta.jsonl` records this. Captions use the HDR Prompt template fields
  `{scene}` / `{lighting}` / `{camera}`, filled from the HDRI name, its Poly Haven categories and the camera move.
- **Disk** (rough estimate): about 0.4 GB of EXR per 73-frame clip and a few MB per pair. Delete `clips/` after
  Make Pairs if space is short (re-running HDRI -> Clips renders them again).
- **Training needs ai-toolkit with MiniMax H3** (upstream commit 4a7f285 or later). If the node says it is missing,
  run `git pull` in your ai-toolkit folder. Run the same job_name again to resume from the last checkpoint.
  Checkpoints are written to `output/hdr_lora/<job_name>/`, and the YAML, manifest and log to
  `output/hdr_lora/a29_runs/<job_name>/`. Copy the final `.safetensors` file to `models/loras` for workflows (a) and
  (b).
- NOT verified here: real downloads, a real training run (needs the H3 weights and a CUDA GPU), and HDR quality.
