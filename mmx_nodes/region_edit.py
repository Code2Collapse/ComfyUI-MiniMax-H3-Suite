"""H3 Region Edit - ONE node for a masked edit steered by a ControlNet (owner A1, ORDERS 11.6, ledger L8.18).

Plate + mask + ControlNet image in, a patched model and the masked AV latent out. The two signals split the frame:
inside the mask the ControlNet steers what is regenerated, outside it the latent keeps the plate, and the ControlNet
adds nothing there (core's H3 Fun ControlNet adds its residual to EVERY video row, also the ones the latent mask holds,
which is how the two fought). It is the composition of two tested pieces: H3 Source AV Latent (the plate encoded on
H3's own grid, the mask reduced on H3's frame grouping, sound kept per 1/40 s) and H3 Mask-Aware ControlNet (the
residual gated by the sampler's own denoise mask).

Reading of the owner's words: "ControlNet fades to zero inside the regenerated mask, the masked latent keeps the
plate outside, so they never fight" - taken as the split above (ControlNet zero where the latent holds the plate),
because ControlNet zero inside the mask would leave it nowhere to act. `controlnet_where = everywhere` gives core's
behaviour for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet  # noqa: E402
from mmx_nodes.source_av_latent import AUDIO_MODES, SIZE_POLICIES, build_source_av_latent  # noqa: E402
from mmx_utils.mask_aware_control import SCHEDULES  # noqa: E402

CATEGORY = "MiniMax H3/Conditioning"
WHERE = ("inside the mask", "everywhere")
HINTS = ("auto", "control video", "inpaint from the plate", "both")


def _out(o):
    return o.result if hasattr(o, "result") else (o.args if hasattr(o, "args") else o)


def region_edit(model, control_net, video_vae, images, mask, control_video=None, audio_vae=None, audio=None,
                strength=1.0, controlnet_where=WHERE[0], hint=HINTS[0], boundary_softness=1.0,
                schedule=SCHEDULES[0], audio_mode="keep", size_policy=SIZE_POLICIES[0], control_region=None,
                sigma_start=1.0, sigma_end=0.0):
    if mask is None:
        raise ValueError("H3 Region Edit needs the mask of what to regenerate (1 = regenerate, 0 = keep the plate).")
    latent, latent_report = build_source_av_latent(images, video_vae, audio_vae, audio, mask, audio_mode, size_policy)
    lines = ["Latent: " + latent_report.replace("\n", "\n  ")]
    if control_net is None or float(strength) == 0.0:
        lines.append("No ControlNet (none connected or strength 0): a masked edit only, the plate is kept outside "
                     "the mask.")
        return model, latent, "\n".join(lines)
    if hint == "auto":
        hint = "control video" if control_video is not None else "inpaint from the plate"
    if hint in ("control video", "both") and control_video is None:
        raise ValueError(f"hint '{hint}' needs a control_video (depth, pose, edges); connect one or choose "
                         "'inpaint from the plate'.")
    inpaint = hint in ("inpaint from the plate", "both")
    res = _out(MiniMaxH3_MaskAwareControlNet.execute(
        model=model, control_net=control_net, vae=video_vae, strength=float(strength),
        preserved_strength=0.0 if controlnet_where == WHERE[0] else 1.0,
        boundary_softness=float(boundary_softness), affect_text_rows=False, inpaint=inpaint,
        control_video=control_video if hint in ("control video", "both") else None,
        mask=mask if inpaint else None, source_video=images if inpaint else None,
        sigma_start=float(sigma_start), sigma_end=float(sigma_end), control_region=control_region,
        schedule=schedule))
    patched, control_report = res[0], res[1]
    where = ("ControlNet steers only the regenerated area; outside the mask the latent keeps the plate and the "
             "ControlNet adds nothing" if controlnet_where == WHERE[0] else
             "ControlNet everywhere (core's behaviour): it also pushes the rows the latent keeps")
    lines += [f"ControlNet: {hint}, strength {float(strength):.2f}. {where}.",
              "  " + str(control_report).replace("\n", "\n  ")]
    return patched, latent, "\n".join(lines)


class MiniMaxH3_RegionEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RegionEdit",
            display_name="H3 Region Edit (plate + mask + ControlNet)",
            category=CATEGORY,
            description=(
                "One node for a masked edit steered by a ControlNet. Inside the mask the ControlNet steers what is "
                "regenerated; outside it the latent keeps the plate and the ControlNet adds nothing, so the two "
                "never fight. Feed `model` to the sampler's model, `latent` to its latent_image. Plate encoded on "
                "H3's own grid (no crop, no offset), mask reduced on H3's frame grouping, sound kept per 1/40 s."),
            search_aliases=["region edit", "masked controlnet", "controlnet inpaint", "nkd av latent", "plate mask"],
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model."),
                io.Vae.Input("video_vae", tooltip="The MiniMax H3 video VAE (encodes the plate and the hint)."),
                io.Image.Input("images", tooltip="The plate: source frames (17k+5 of them) at the generation size."),
                io.Mask.Input("mask", tooltip="What to regenerate, per frame at the frame size (1 = regenerate)."),
                io.ModelPatch.Input("control_net", optional=True,
                                    tooltip="MiniMax H3 Fun ControlNet (ModelPatchLoader). Without it: a masked "
                                            "edit only."),
                io.Image.Input("control_video", optional=True, tooltip="The hint: depth, pose, edges, per frame."),
                io.Vae.Input("audio_vae", optional=True, tooltip="The MiniMax H3 audio VAE (needed with audio)."),
                io.Audio.Input("audio", optional=True, tooltip="The plate's sound (silent track without it)."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="ControlNet strength, as in core."),
                io.Combo.Input("controlnet_where", options=list(WHERE), default=WHERE[0],
                               tooltip="inside the mask: the ControlNet steers only the regenerated area (the fix). "
                                       "everywhere: core's behaviour, for comparison."),
                io.Combo.Input("hint", options=list(HINTS), default=HINTS[0],
                               tooltip="What the ControlNet is given. auto: the control video if connected, else "
                                       "inpaint from the plate. both: structure and the inpaint channels."),
                io.Float.Input("boundary_softness", default=1.0, min=0.0, max=16.0, step=0.5,
                               tooltip="Soften the ControlNet's edge at the mask border, in latent patches."),
                io.Combo.Input("schedule", options=list(SCHEDULES), default=SCHEDULES[0],
                               tooltip="ControlNet strength over the steps: constant, fade out (layout early, detail "
                                       "to the model), fade in, smooth fade out."),
                io.Combo.Input("audio_mode", options=list(AUDIO_MODES), default="keep",
                               tooltip="keep the plate's sound, regenerate it, or follow the mask per 1/40 s."),
                io.Combo.Input("size_policy", options=list(SIZE_POLICIES), default=SIZE_POLICIES[0], advanced=True,
                               tooltip="A frame that is not a multiple of 16: refuse, or pad right / bottom."),
                io.Mask.Input("control_region", optional=True, advanced=True,
                              tooltip="Restrict the ControlNet further, e.g. pose only on the person (1 = steer)."),
                io.Float.Input("sigma_start", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               advanced=True),
                io.Float.Input("sigma_end", default=0.0, min=0.0, max=1.0, step=0.01, optional=True, advanced=True),
            ],
            outputs=[io.Model.Output(display_name="model"), io.Latent.Output(display_name="latent"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, video_vae, images, mask, control_net=None, control_video=None, audio_vae=None, audio=None,
                strength=1.0, controlnet_where=WHERE[0], hint=HINTS[0], boundary_softness=1.0,
                schedule=SCHEDULES[0], audio_mode="keep", size_policy=SIZE_POLICIES[0], control_region=None,
                sigma_start=1.0, sigma_end=0.0) -> io.NodeOutput:
        m, latent, report = region_edit(model, control_net, video_vae, images, mask, control_video, audio_vae, audio,
                                        strength, controlnet_where, hint, boundary_softness, schedule, audio_mode,
                                        size_policy, control_region, sigma_start, sigma_end)
        return io.NodeOutput(m, latent, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_RegionEdit", "region_edit"]
