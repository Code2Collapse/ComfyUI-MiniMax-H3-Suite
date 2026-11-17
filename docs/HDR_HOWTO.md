# HDR with the MiniMax H3 Suite - how-to

Everything is inside the Suite (menu **MiniMax H3 / HDR**). Example workflows are in `workflows/hdr/`.
To find a node, double-click the canvas and type **H3 HDR** (e.g. "H3 HDR · Log Decode", "H3 HDR Dataset · Make Pairs").
Values: scene-linear Rec.709, **1.0 = diffuse white**; brighter things are simply bigger numbers (10 = 10x white).
The H3 model sees HDR through one curve, **diffhdr_log_16384** (holds up to 16384x white).

## 1. Install
1. Close ComfyUI.
2. Replace `ComfyUI/custom_nodes/ComfyUI-MiniMaxSuite` with the folder from the zip (keep a copy of the old one).
3. Start ComfyUI. Nothing new needs installing: EXR, video and downloads use what ComfyUI already has.
   (Only if a save node says "this OpenCV cannot write EXR" run `pip install OpenEXR` in ComfyUI's Python.)

## 2. SDR video -> HDR EXR (workflow `a_sdr_to_hdr_exr.json`)
1. Load your SDR clip. Frame counts must be 17k+5 (22, 39, 56 ... 124); draft size 1344x768.
2. The HDR IC-LoRA turns it into a log-coded HDR clip (the SDR clip is `<Video 1>`).
3. **HDR Log Decode** turns the log picture back into linear light (values above 1.0 survive).
4. **HDR Bias Correct** removes the H3 VAE's small darkening (table editable; defaults measured in your
   `H3_HDR_Range_Comparison.xlsx`).
5. **HDR Save EXR Sequence** writes half-float EXR (ZIP), never clipped. **HDR Tonemap Preview** only shows a picture.
Optional: **HDR Gamut Convert** (Rec.2020 / ACEScg), **HDR Save HDR10 Video** (PQ, 203-nit paper white), **HDR Save
Log 16-bit** (16-bit PNG / TIFF of the log picture).

## 3. 4K: add Open-H3 Regenerate (workflow `b_sdr_to_hdr_regen4k.json`)
Regenerate STD makes a 4K SDR version of the same clip. **HDR Gain Map Join** puts the HDR brightness of the draft
onto the 4K picture: gain = HDR / (4K picture, shrunk to draft size), smoothed up to 4K along the 4K picture's edges.

## 4. Build a training set (workflow `c_dataset_build.json`)
1. **Fetch Poly Haven** (free CC0 HDRIs; choose category, resolution, how many) or **Fetch Hugging Face** (a dataset
   repo, with a size cap in GB). Downloads run in the background, can be cancelled, and skip what is already there.
2. **HDRI -> Clips** turns each panorama into short camera moves (pan / tilt / push) as EXR clips at 1344x768.
3. **Make Pairs** writes, per clip: the SDR control video (what a camera would have clipped), the log-HDR target
   video, a caption and a line in `meta.jsonl`.

## 5. Train the HDR IC-LoRA (workflow `d_train.json`)
Training runs through **ai-toolkit** (ostris). Your copy at `D:\PROJECT\ai-toolkit` is older than its MiniMax H3
support: open a terminal there and run `git pull` once. ai-toolkit already does what this LoRA needs - the SDR clip
is a reference video and only the HDR target is learned (checked in its code). **HDR Train IC-LoRA** writes the
training settings, starts ai-toolkit in the background, shows the log and the loss curve, and resumes after a stop.
The example workflow ships with **dry_run on**: the first run only checks everything and writes the settings file.
Turn dry_run off to train. Training runs in the background, so the ComfyUI queue is not blocked; cancel it from the
jobs panel.

## 6. Check the result (workflow `e_eval.json`)
**HDR Eval Report** compares decoded HDR with the ground truth and prints the error in stops for each brightness band
(the same table as the H3HDR test node).

## What was tested, and what was not
- Tested on this computer with made-up (synthetic) pictures: the curve round trip (error < 0.0001 relative from
  0.001x to 16384x white), EXR keeps values above 1, the bias table, gamut, PQ, the gain-map join, file writing.
- **Not tested here**: anything that needs the real H3 model, your LoRA, a GPU, a real download or a real training
  run. Picture quality is therefore not verified yet.
- The `comfyui-diffhdr` folder you mentioned is not on this computer; the curve comes from your H3HDR-Test
  `hdr_math.py`, as you asked.
- The HDR Prompt template you mentioned was not in the message; the node has a default you can edit.
