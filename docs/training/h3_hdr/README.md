# Training H3 HDR adapters (A26 3.4)

This is the Suite's reproducible training integration for the two HDR goals. It uses **ai-toolkit** (ostris,
MIT) as an external program at **one pinned commit**. Nothing of ai-toolkit is copied into this repository.

| | |
|---|---|
| Trainer | github.com/ostris/ai-toolkit, commit `4a7f28532ed6fe81d9f8c00718878bb3b8739015` (main, 2026-11-10) |
| Model code used | `extensions_built_in/diffusion_models/minimax_h3/` at that commit (`arch: minimax_h3`) |
| Checker / launcher | `tools/h3_train.py` (`check`, `val-items`, `launch`, `export-check`) |
| Goal A config | `goal_a_log_lora.yaml`: H3 generates in a log code |
| Goal B config | `goal_b_sdr2hdr_iclora.yaml`: SDR-conditioned HDR reconstruction |
| Data | `tools/h3_hdr_dataset.py` (manifest v2: splits, integrity, precision audit) |

Status: the integration is written and checked against the pinned code, and the tool is tested. **No training run
has been done** (it needs the owner's GPU box). Nothing here claims a trained adapter exists.

## 1. Set up the pinned trainer

```
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
git checkout 4a7f28532ed6fe81d9f8c00718878bb3b8739015
python -m venv venv && venv/bin/pip install -r requirements.txt     # its own environment, not ComfyUI's
```

`requirements_base.txt` at the pin fixes transformers 5.5.3, diffusers at commit c943837, torchao 0.10.0 and
optimum-quanto 0.2.4, but leaves accelerate and bitsandbytes unpinned. `launch` therefore saves `pip freeze` with
every run.

## 2. Identity of what is trained

- **Base checkpoint:** `model_kwargs.partition` picks the H3 DiT file: `fl2va` (t2v + first-frame i2v) or
  `fl2va_pruned` (less VRAM; the trainer's default). Set `dit_<partition>_path`, `text_encoder_path`,
  `video_vae_path` and `audio_vae_path` to the files ComfyUI already uses, so nothing is downloaded twice.
  `launch --full-hash` records the name, size and SHA-256 of each in `run_manifest.json`.
- **VAE:** `minimax_h3_video_vae_fp16.safetensors`. It must be the same file the probe measured
  (`tools/h3_vae_log_probe.py`), and the code space must be the one the probe chose (`meta.c2c_dataset_code`).

## 3. What the adapter changes (LoRA target modules)

H3 runs text, video and audio rows through **one** sequence. The 50 blocks' `attn.qkv_proj`, `attn.out_proj`,
`mlp.fc1` and `mlp.fc2` are shared by all rows; only the AdaLN modulation is chosen per modality
(`MiniMaxH3Block` at the pin).

- **Adapted:** `blocks.N.attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` (50 x 4 = 200 modules;
  `network.transformer_only: true`).
- **Never adapted** (`network_kwargs.ignore_if_contains`): `audio` (audio_patch_proj, final_layer.audio_out),
  `adaln_proj` (the per-modality modulation table), `token_refiner` (text; it has its own `blocks`),
  `time_embedder`, `condition_proj`, `patch_proj`, `final_layer`.
- **Consequence:** the adapted layers also process audio rows. Training on video only gives no audio loss, but it
  does not guarantee that audio output is unchanged. The H3 application node (L9.210) applies the low-rank
  contribution to video rows only, and its tests check that audio is unchanged.
- **Rank / alpha:** 128 / 128 (LumiVid and Lightricks' HDR LoRA). Use 64 / 64 if VRAM is short.

## 4. What is optimised (no invented losses)

- `loss_type: mse`: the flow-matching velocity MSE on the video latents, with H3's own training scheduler (video
  flow shift 12, from `get_train_scheduler` at the pin).
- **No audio loss:** `do_audio: false`, so audio rides along as silence and does not enter the loss.
- **No temporal loss:** the pinned trainer has none. `check` rejects config keys such as `temporal_loss`.
- `pixelspace` is a real `loss_type` at the pin, but it decodes through the 33B model's VAE on every step and is
  not used here.
- **Goal B:** the loss covers the whole side-by-side clip, including the SDR half. ai-toolkit has no per-region loss
  mask for video at the pin.

## 5. Goal A vs Goal B

| | Goal A: log-domain generation | Goal B: SDR-conditioned reconstruction |
|---|---|---|
| Sample | one clip in the log code (`<dataset>/train`) | `[SDR rendering \| same frames in log code]` (`<dataset>/trackc/train`, built with `--trackc`) |
| Conditioning | the caption (text-to-video; first-frame i2v with `fl2va`) | the SDR source itself: frame-aligned, in the same clip; captions carry the `[HDRPAIR]` tag (dropout 0) |
| Inference | generate, then H3 HDR Roundtrip `to_scene` | SDR clip as the left half; H3 Masked Replace keeps it (denoise mask 0), regenerates the right half; crop + `to_scene` |
| Claim allowed after | the A26 3.8 checks on generated video | a measured conversion on the TEST split (alignment, per-stop error, flicker) |

HDRI orbit clips teach HDR appearance only. Their SDR renderings are frame-aligned, but they are not
source-conditioned plates; the manifest's `goal` field says so per clip.

## 6. Data, frames, batches

- `num_frames: 39` with `auto_frame_count: true` snaps every clip down to H3's 17k+5 grid. `batch_size: 1`,
  `gradient_accumulation: 1`.
- `resolution: R` caps a clip's area at R x R, keeps its aspect, never upscales, and rounds the sides to 32
  (`toolkit/buckets.py`). `check --dataset` prints the actual training size: Goal A 960x544 at R = 768; Goal B
  1664x480 at R = 896.
- **Precision:** ai-toolkit reads training video as 8-bit. The dataset keeps a 16-bit copy until the probe shows
  8-bit is precise enough (see the dataset manifest's `precision.decision`).
- **Memory:** `low_vram`, `gradient_checkpointing`, `bf16`, `adamw8bit`, cached latents and text embeddings.

## 7. Validation, checkpoints, seed, resume, logs, export

- **Checkpoints:** `save_every: 500`, keeping 10.
- **Validation:**
  - ai-toolkit's built-in `validation_config` measures loss on IMAGES at fixed sigmas. `h3_train.py val-items`
    turns the val split's first frames (the exact training codes) into those items. This is a cheap tracker only.
  - The real validation is per checkpoint: generate the val prompts (Goal A), or convert the val SDR clips
    (Goal B). Decode with H3 HDR Roundtrip, then run `tools/h3_hdr_qc.py` against the aligned HDR references.
  - The TEST split is touched once, for the final claim.
- **Seed:** ai-toolkit at the pin has no training seed setting; sampling uses `sample.seed`, and VAE latents use a
  fixed 42. `launch --seed N` seeds Python, NumPy and torch (CPU + CUDA) and sets PYTHONHASHSEED before the trainer
  starts. CUDA kernels can still be non-deterministic, so two runs are reproducible in setup, not bit-exact.
- **Resume:** ai-toolkit resumes from the newest checkpoint in `<training_folder>/<name>/`. Run `launch` again
  with the same `--run`; `run_manifest.json` records `resume_from`.
- **Logs:** `launch` tees the trainer's output to `<run>/train.log` and writes `run_manifest.json` with: pin and
  actual commit, config SHA-256, dataset-manifest SHA-256, model file identities, Python / torch / CUDA,
  pip freeze, seed, start / finish, exit code and last checkpoint.
- **Export:**
  - ai-toolkit saves `<name>.safetensors` with ComfyUI's original H3 keys (`lora_keys_use_comfy_prefix` at the pin:
    `diffusion_model.blocks.N....lora_A / lora_B`).
  - `h3_train.py export-check` confirms that: only the 200 video-path block linears, both factors present, finite
    values, one rank, coverage.
  - Core ComfyUI 0.39 uses the same H3 module names (`blocks.N.attn.qkv_proj`, `mlp.fc1` ...), and its generic LoRA
    key map (`comfy/lora.py`) takes `diffusion_model.<module>` keys with `lora_A` / `lora_B` factors. So the file
    fits core's LoRA loaders and the Suite's H3 HDR adapter node. This was established by reading the code; no
    trained file has been loaded yet.

## 8. Run

```
python tools/h3_train.py check --toolkit <ai-toolkit> --config my_goal_a.yaml --dataset <set>
python tools/h3_train.py val-items --dataset <set> --out <set>/val_items     # optional loss tracker
python tools/h3_train.py launch --toolkit <ai-toolkit> --python <ai-toolkit>/venv/bin/python \
    --config my_goal_a.yaml --dataset <set> --run runs/goal_a_01 --seed 1234 --full-hash
python tools/h3_train.py export-check runs/goal_a_01/output/h3_hdr_log_acescct_v1/h3_hdr_log_acescct_v1.safetensors
```

Train Goal A first: Goal B reuses its data and its code-space choice.
