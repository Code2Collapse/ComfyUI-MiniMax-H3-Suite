# PORTED FROM: third_party/ComfyUI-Viggle-Remake/nodes.py (Apache-2.0, Viggle Remake authors) - the arithmetic
# (stamps, canvas, pins, audio padding, prefix rows, keyframe release); see CREDITS.md
"""Viggle Remake on H3: the pure part (torch only, no ComfyUI import, so it is tested on CPU without a model).

Remake (two rank-256 LoRAs on MiniMax H3) re-renders a source clip from 1 to 4 painted keyframes. What it does
differently from stock Reference to Video, reproduced from the reference nodes:

* the source clip is ``<Video 1>`` with a TEXT-ONLY presentation: one label per 2 fps block, no vision tokens;
* each image is ``<Picture N>``; with pins, also a keyframe at that real frame of the output;
* the source soundtrack is the clean target audio (noise_mask 0) for the whole denoise;
* a prefix (the previous take's frames) opens a render held clean, like the soundtrack.

The Suite adds one canvas rule for plate work (``framing_canvas``): the window's own framing, stretched, never cropped,
so the decoded frames map back onto the plate by the inverse stretch (docs/PLAN_h3_replacement_quality.md s3).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

FPS, AUDIO_SR, AUDIO_LATENT_FPS = 24, 32000, 40
AUDIO_FRAME_SAMPLES = 800          # one audio latent frame: 32000 / 40
CANVAS = 32                        # H3 canvas multiple (2x2 DiT patch of a 16x VAE)
MAX_PICTURES = 4                   # Remake is trained on 1 to 4 painted keyframes


class RemakeError(ValueError):
    """A wrong input, said in words the node shows."""


def check_frames(nf: int) -> int:
    nf = int(nf)
    if nf < 5 or (nf - 5) % 17:
        legal = ", ".join(str(17 * k + 5) for k in (5, 6, 7, 8, 10, 15, 20))
        raise RemakeError(f"H3 needs 17n + 5 frames (e.g. {legal}); got {nf}. Trim the window or let the Job "
                          "Controller cut it.")
    return nf


def target_latent_t(nf: int) -> int:
    """Video latent frames of an nf-frame H3 target (core: 2 for the first 5 frames, then 5 per 17)."""
    return (int(nf) - 5) // 17 * 5 + 2


def stamps(nf: int) -> list[float]:
    """The conditioner's 2 fps block labels for an nf-frame clip: every 12th frame, merged in pairs, each pair labelled
    with its mean time ("%.1f" rounds half to even, so the first block is <0.2 seconds>)."""
    idx, cursor = [], 0.0
    while round(cursor) < nf:
        if not idx or round(cursor) > idx[-1]:
            idx.append(round(cursor))
        cursor += FPS / 2
    t = [i / 2 for i in range(len(idx))]
    t += t[-1:] * (len(t) % 2)
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]


def remake_segments(nf: int, n_refs: int, prompt: str) -> list[str]:
    """The text-only presentation, one string per separately tokenized segment."""
    return ["<Video 1>: ", *["<%.1f seconds>" % s for s in stamps(nf)],
            *["<Picture %d>: " % (i + 1) for i in range(int(n_refs))], str(prompt)]


def parse_pins(text: str, n_refs: int, nf: int) -> list[int]:
    try:
        pins = [int(p) for p in str(text or "").replace(",", " ").split()]
    except ValueError as exc:
        raise RemakeError(f"pins must be whole frame numbers separated by commas, e.g. '0, 120' ({exc}).") from exc
    if len(pins) not in (0, int(n_refs)):
        raise RemakeError(f"give one pin per reference image or none: {len(pins)} pins, {n_refs} references.")
    bad = [p for p in pins if not 0 <= p < int(nf)]
    if bad:
        raise RemakeError(f"pins must be output frames in [0, {nf}); got {bad}.")
    return pins


def viggle_canvas(h: int, w: int, se: int) -> tuple[int, int] | None:
    """Remake's canvas: None = the source is used as is (short edge == se, both sides multiples of 32); otherwise the
    (width, height) of the 16:9-class canvas it is cover-cropped onto (832x480, 1280x736)."""
    if min(h, w) == se and not h % CANVAS and not w % CANVAS:
        return None
    long = se * 16 // 9 // CANVAS * CANVAS
    return (long, se) if w >= h else (se, long)


def framing_canvas(h: int, w: int, se: int) -> tuple[int, int]:
    """The window's own framing on an H3 canvas: short edge = se, the long side rounded to a multiple of 32. A pure
    stretch (never a crop): (width, height)."""
    h, w, se = int(h), int(w), int(se)
    if h <= 0 or w <= 0:
        raise RemakeError(f"empty frame {w}x{h}.")
    if w >= h:
        return max(CANVAS, round(w * se / h / CANVAS) * CANVAS), se
    return se, max(CANVAS, round(h * se / w / CANVAS) * CANVAS)


def ref_canvas(rh: int, rw: int, se: int) -> tuple[int, int]:
    """A <Picture N>'s size: its own aspect at short edge se, each side rounded to 32, no area cap: (width, height)."""
    s = min(rh, rw)
    th = max(CANVAS, round(rh * se / s / CANVAS) * CANVAS)
    tw = max(CANVAS, round(rw * se / s / CANVAS) * CANVAS)
    return tw, th


def prefix_rows(k: int) -> int:
    """Latent frames that hold k prefix frames (k a multiple of 17: 5 latents per 17 frames)."""
    return int(k) // 17 * 5


def check_prefix(k: int, nf: int) -> None:
    if k <= 0 or k % 17:
        raise RemakeError(f"the prefix must be 17, 34, 51 ... frames; got {k}. In a job, set H3 Task Spec's context "
                          "to a multiple of 17 - the window's first context frames ARE the accepted frames, so a "
                          "shorter prefix would generate accepted frames again.")
    if prefix_rows(k) >= target_latent_t(nf):
        raise RemakeError(f"a {k}-frame prefix fills the whole {nf}-frame render - shorten the context.")


def prepare_wav(audio: dict | None, nf: int, resample: Callable[[torch.Tensor, int, int], torch.Tensor]) -> torch.Tensor:
    """The soundtrack the target holds clean: nf/24 s, padded to (nf+1)/24 s, stereo, at 32 kHz, then zero-padded to
    whole 800-sample latent frames (VAE.encode would centre-crop instead and shift the track by up to 400 samples).
    Silence when there is no audio. Returns [2, n]."""
    n = round((int(nf) + 1) / FPS * AUDIO_SR)
    wav = torch.zeros(2, n)
    if audio is not None:
        wav, sr = audio["waveform"][0].float(), int(audio["sample_rate"])
        wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
        wav = wav[:, :round(int(nf) / FPS * sr)]
        wav = resample(wav, sr, AUDIO_SR) if sr != AUDIO_SR else wav
    wav = torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]
    return torch.nn.functional.pad(wav, (0, -n % AUDIO_FRAME_SAMPLES))


def audio_latent_len(nf: int) -> int:
    return round(int(nf) / FPS * AUDIO_LATENT_FPS)


def release_payload(p: dict) -> dict:
    """The H3 payload once the keyframes are released: only the references remain as condition latents; the layout
    is rebuilt by the model from the shapes (Remake's KeyframeUntil)."""
    refs = p.get("refs") or []
    return dict(p, keyframes=None, layout=None,
                cond_video_latents=[r["latent"] for r in refs if "latent" in r],
                cond_audio_latents=[r["audio_latent"] for r in refs if r.get("audio_latent") is not None])


def noise_skip_shapes(cond_extra: dict) -> list[tuple[int, ...]]:
    """Shapes the reference script draws noise for before the target's (refs, then keyframes)."""
    blocks = list(cond_extra.get("minimax_refs", [])) + list(cond_extra.get("minimax_keyframes", []))
    return [tuple(b["latent"].shape) for b in blocks if "latent" in b]


def is_placeholder_context(images: torch.Tensor | None) -> bool:
    """The Job Controller's 'no context' image ([1, 8, 8, 3] zeros) or nothing at all."""
    if images is None:
        return True
    if images.ndim == 4 and tuple(images.shape[:3]) == (1, 8, 8) and not bool(images.abs().sum() > 0):
        return True
    return False


def normalize_boxes(bboxes: Any, n: int) -> list[dict]:
    """A single box, a list of boxes, or a per-frame list of box lists -> one box dict per frame."""
    if isinstance(bboxes, dict):
        return [bboxes] * n
    if isinstance(bboxes, list) and bboxes:
        if all(isinstance(f, dict) for f in bboxes):
            frames = [[f] for f in bboxes]
        elif all(isinstance(f, list) and f and isinstance(f[0], dict) for f in bboxes):
            frames = bboxes
        else:
            frames = None
        if frames is not None:
            if len(frames) == 1:
                return [frames[0][0]] * n
            return [f[0] for f in frames]
    raise RemakeError("bboxes must be H3 Subject Crop's boxes: one box per frame.")


def crop_box(image: torch.Tensor, box: dict) -> torch.Tensor:
    """[1, H, W, C] cut to box {x, y, width, height} (clamped to the image)."""
    h, w = int(image.shape[1]), int(image.shape[2])
    x0, y0 = max(0, int(round(box["x"]))), max(0, int(round(box["y"])))
    x1, y1 = min(w, x0 + int(round(box["width"]))), min(h, y0 + int(round(box["height"])))
    if x1 <= x0 or y1 <= y0:
        raise RemakeError(f"the anchor box {box} lies outside the {w}x{h} anchor.")
    return image[:, y0:y1, x0:x1]


def full_boxes(n: int, w: int, h: int) -> list[list[dict]]:
    """One whole-window box per frame (H3 Plate Restore's per-frame box list)."""
    return [[{"x": 0, "y": 0, "width": int(w), "height": int(h)}] for _ in range(int(n))]


# ── default prompts (six sections, docs/PLAN_h3_replacement_quality.md s3) ────────────────────────────────────────

MODES = ("character", "face / head", "item / prop")


def default_sections(mode: str, subject: str, anchor_frame: int) -> dict[str, str]:
    subject = str(subject or "").strip() or ("the new object" if mode == MODES[2] else "the new character")
    pic = (f"<Picture 1> is frame {int(anchor_frame)} of <Video 1> repainted: it shows exactly how the result must "
           "look.")
    plate = ("<Video 1> is the source video: it defines framing, camera motion, timing, light and composition.")
    keep = ("<Video 1> (everything not named above): fully_preserved - background, other people, lighting, grain, "
            "camera, framing and timing are retained.")
    sound = "Copied from <Video 1>."
    if mode == MODES[1]:
        return {
            "subject_definitions": (f"<Subject 1> is {subject}: the head and face shown in <Picture 1>.\n"
                                    f"{pic}\n<Subject 2> is the performer in <Video 1>, whose head is replaced.\n"
                                    f"{plate}"),
            "summary": ("[video editing + reference generation] The target video is <Video 1> with only the head and "
                        "face of <Subject 2> replaced by those of <Subject 1>, inside the masked region. Expression, "
                        "eye and lip movement, head pose and timing follow <Subject 2>."),
            "retention_analysis": ("<Subject 1> (head and face): attribute_transfer - identity from <Picture 1> in "
                                   "every frame.\n<Subject 2> (body, wardrobe, performance): fully_preserved.\n" + keep),
            "detailed_description": ("The target video matches the live-action plate in lens, lighting and grade.\n"
                                     "[Shot 1] Continuous take from <Video 1>; only the head is <Subject 1>, lit by the "
                                     "plate's light."),
            "overall_soundscape": sound, "non_diegetic_music": "none.",
        }
    if mode == MODES[2]:
        return {
            "subject_definitions": (f"<Subject 1> is {subject} as shown in <Picture 1>.\n{pic}\n"
                                    f"<Subject 2> is the object in the masked region of <Video 1>; it is replaced.\n"
                                    f"{plate}"),
            "summary": ("[video editing + reference generation] The target video is <Video 1> with only the masked "
                        "object replaced by <Subject 1>, in its place, scale, perspective and motion, held and moved "
                        "exactly as <Subject 2> was."),
            "retention_analysis": ("<Subject 1> (the object): attribute_transfer - look from <Picture 1>.\n"
                                   "Hands and people touching it: fully_preserved.\n" + keep),
            "detailed_description": ("The target video matches the live-action plate in lens, lighting and grade.\n"
                                     "[Shot 1] Continuous take from <Video 1>; <Subject 1> sits where <Subject 2> was, "
                                     "with the plate's light, shadows and reflections."),
            "overall_soundscape": sound, "non_diegetic_music": "none.",
        }
    return {
        "subject_definitions": (f"<Subject 1> is {subject} as shown in <Picture 1>.\n{pic}\n"
                                f"<Subject 2> is the performer in <Video 1>; the performer is replaced.\n{plate}"),
        "summary": ("[video editing + reference generation] The target video is <Video 1> with <Subject 2> replaced by "
                    "<Subject 1> in every frame, the same person throughout. Pose, expression, lip movement and "
                    "timing follow <Subject 2>."),
        "retention_analysis": ("<Subject 1> (appears in [Shot 1], every frame): attribute_transfer - identity and "
                               "appearance from <Picture 1>.\n<Subject 2> (performance): partially_preserved - pose, "
                               "expression and timing kept; identity replaced.\n" + keep),
        "detailed_description": ("The target video matches the live-action plate in lens, lighting and grade.\n"
                                 "[Shot 1] Continuous take from <Video 1>; <Subject 1> performs exactly as <Subject 2> "
                                 "did, lit by the plate's light."),
        "overall_soundscape": sound, "non_diegetic_music": "none.",
    }
