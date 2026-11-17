# PORTED FROM: ComfyUI-H3HDR-Test/hdr_math.py (the owner's own test pack, no licence file; ported on the owner's order A29,
# 2026-11-16). The ported part below the "ported" banner is unchanged; the Suite additions follow the "added" banner.
"""Math for the H3 HDR round-trip test: encodings + metrics."""

import math

import torch

EPS = 1e-4

# ---------------------------------------------------------------- encodings
ENCODINGS = [
    "diffhdr_log_65536",   # DiffHDR curve, range 65536 (what DiffHDR uses)
    "diffhdr_log_16384",   # same curve, range 16384
    "diffhdr_log_4096",    # same curve, range 4096
    "diffhdr_log_1024",    # same curve, range 1024 (more code values for shadows/mids)
    "logc3_ei800",         # ARRI LogC3 (what the LTX HDR IC-LoRA uses), range ~55x white
    "srgb_clip_control",   # CONTROL: ordinary clipped sRGB image, no HDR at all
]
_LOG_GAMMA = 2.2

# ARRI LogC3, EI 800
_C = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)


def _diffhdr_range(name: str) -> float:
    return float(name.rsplit("_", 1)[1])


def encode(x: torch.Tensor, name: str) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    if name.startswith("diffhdr"):
        r = _diffhdr_range(name)
        x = torch.clamp(x, max=r)
        n = math.log(_LOG_GAMMA * r + 1.0)
        return torch.pow(torch.log(_LOG_GAMMA * x + 1.0) / n, 1.0 / _LOG_GAMMA)
    if name == "logc3_ei800":
        c = _C
        y = torch.where(x > c["cut"],
                        c["c"] * torch.log10(c["a"] * x + c["b"]) + c["d"],
                        c["e"] * x + c["f"])
        return torch.clamp(y, 0.0, 1.0)
    if name == "srgb_clip_control":
        x = torch.clamp(x, max=1.0)
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x.clamp_min(1e-12), 1 / 2.4) - 0.055)
    raise ValueError(name)


def decode(y: torch.Tensor, name: str) -> torch.Tensor:
    y = torch.clamp(y, 0.0, 1.0)
    if name.startswith("diffhdr"):
        r = _diffhdr_range(name)
        n = math.log(_LOG_GAMMA * r + 1.0)
        return (torch.exp(torch.pow(y, _LOG_GAMMA) * n) - 1.0) / _LOG_GAMMA
    if name == "logc3_ei800":
        c = _C
        lin_hi = (torch.pow(10.0, (y - c["d"]) / c["c"]) - c["b"]) / c["a"]
        lin_lo = (y - c["f"]) / c["e"]
        return torch.where(y > c["e"] * c["cut"] + c["f"], lin_hi, lin_lo).clamp_min(0.0)
    if name == "srgb_clip_control":
        return torch.where(y <= 0.04045, y / 12.92, torch.pow((y + 0.055) / 1.055, 2.4))
    raise ValueError(name)


def enc_max(name: str) -> float:
    """Largest linear value the encoding can represent."""
    if name.startswith("diffhdr"):
        return _diffhdr_range(name)
    if name == "srgb_clip_control":
        return 1.0
    return float(decode(torch.tensor(1.0), name))


# ------------------------------------------------------------------ metrics
def luma709(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def signed_stops(ref: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
    """log2(rec/ref). Positive = VAE output is brighter than the original."""
    return torch.log2(rec.clamp_min(0) + EPS) - torch.log2(ref.clamp_min(0) + EPS)


BANDS = [
    ("shadows   0.001-0.01", 0.001, 0.01),
    ("low       0.01-0.1", 0.01, 0.1),
    ("mid       0.1-1", 0.1, 1.0),
    ("high      1-10", 1.0, 10.0),
    ("very high 10-100", 10.0, 100.0),
    ("extreme   100-1000", 100.0, 1000.0),
    ("ultra     1000+", 1000.0, float("inf")),
]


def _q(x: torch.Tensor, q: float) -> float:
    x = x.flatten().float()
    if x.numel() > 2_000_000:
        x = x[:: max(1, x.numel() // 2_000_000)]
    return float(torch.quantile(x, q))


def band_stats(ref: torch.Tensor, rec: torch.Tensor) -> list:
    """[(label, pixels, luma mean|err|, luma p99|err|, luma bias, worst-channel p99)] in stops.
    Bands come from the original's luma. Luma error is what you see; worst-channel error also counts
    tiny channels inside saturated colours, which exaggerates dark-band numbers on synthetic patterns."""
    luma_ref, luma_rec = luma709(ref), luma709(rec)
    ls = signed_stops(luma_ref, luma_rec)
    cs = signed_stops(ref, rec).abs().amax(dim=-1)
    rows = []
    for label, lo, hi in BANDS:
        sel = (luma_ref >= lo) & (luma_ref < hi)
        n = int(sel.sum())
        if n < 50:
            rows.append((label, n, None, None, None, None))
        else:
            rows.append((label, n, float(ls[sel].abs().mean()), _q(ls[sel].abs(), 0.99), float(ls[sel].mean()), _q(cs[sel], 0.99)))
    return rows


def code_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 0 else -10.0 * math.log10(mse)


def quantize8(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0, 1) * 255.0) / 255.0


def format_report(name, ref_lin, rec_lin, ref_code, rec_code, base_lin, base_code) -> str:
    vae, base = band_stats(ref_lin, rec_lin), band_stats(ref_lin, base_lin)
    lines = [
        f"H3 VAE round trip | encoding: {name} | encoder max = {enc_max(name):.1f}x white",
        "Y = luminance error in photographic stops (lower is better). bias: + = VAE output brighter.",
        "chan p99 = worst single colour channel (harsh on synthetic saturated patches). 8bit = rounding only, no VAE.",
        "",
        f"{'luminance band':<22}{'pixels':>9}  {'Y mean':>7}{'Y p99':>7}{'Y bias':>8}{'chan p99':>9}  {'8bit Y p99':>10}",
    ]
    for (label, n, vm, vp, vb, vc), (_, _, _, bp, _, _) in zip(vae, base):
        if vm is None:
            lines.append(f"{label:<22}{n:>9}  (too few pixels)")
        else:
            lines.append(f"{label:<22}{n:>9}  {vm:>7.3f}{vp:>7.3f}{vb:>+8.3f}{vc:>9.3f}  {bp:>10.3f}")
    lines += ["", f"PSNR in code domain (what the VAE actually sees): VAE {code_psnr(ref_code, rec_code):.2f} dB | 8-bit rounding {code_psnr(ref_code, base_code):.2f} dB"]
    return "\n".join(lines)


# ======================================================================================================================
# added - MiniMax H3 Suite HDR (owner order A29, 2026-11-16; design docs/hdr/A29_DESIGN.md in the C2C workspace).
# Working space: scene-linear Rec.709, 1.0 = diffuse white. Nothing below clamps linear light to 1.
# ======================================================================================================================

DEFAULT_ENCODING = "diffhdr_log_16384"          # fixed decision (A29)
SUITE_ENCODINGS = [DEFAULT_ENCODING] + [e for e in ENCODINGS if e != DEFAULT_ENCODING]


# ------------------------------------------------------------------------------------------------------- sRGB / PQ
def srgb_to_linear(v: torch.Tensor) -> torch.Tensor:
    """sRGB code values (0..1) -> linear. The input is an SDR picture, so it is clamped to 0..1 first."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.04045, v / 12.92, torch.pow((v + 0.055) / 1.055, 2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """linear -> sRGB code (display): clamps to 0..1 - use only for previews and SDR outputs."""
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x.clamp_min(1e-12), 1 / 2.4) - 0.055)


PQ_M1, PQ_M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
PQ_C1, PQ_C2, PQ_C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
PAPER_WHITE_NITS = 203.0                           # BT.2408 reference white


def pq_encode(nits: torch.Tensor) -> torch.Tensor:
    """SMPTE ST 2084 inverse EOTF: absolute nits (0..10000) -> PQ signal 0..1. Above 10000 nits is clipped (the
    format cannot hold it)."""
    y = (nits.clamp_min(0.0) / 10000.0).clamp(max=1.0)
    p = torch.pow(y, PQ_M1)
    return torch.pow((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2)


def pq_decode(e: torch.Tensor) -> torch.Tensor:
    """PQ signal -> absolute nits."""
    p = torch.pow(e.clamp(0.0, 1.0), 1.0 / PQ_M2)
    return 10000.0 * torch.pow((p - PQ_C1).clamp_min(0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1)


# ---------------------------------------------------------------------------------------------------------- gamut
GAMUTS = {"rec709": "sRGB / Rec.709", "rec2020": "Rec.2020", "acescg": "ACEScg (AP1)"}


def gamut_matrix(src: str, dst: str) -> torch.Tensor:
    """3x3 RGB matrix src -> dst (names as in GAMUTS), from the vendored primaries (Bradford / ACES hub)."""
    if src not in GAMUTS or dst not in GAMUTS:
        raise ValueError(f"gamut must be one of {', '.join(GAMUTS)}")
    if src == dst:
        return torch.eye(3)
    try:
        from mmx_vendor.hdr_curves._primaries import gamut_matrix as _gm
    except ImportError:                                            # imported as <pack>.mmx_utils.hdr_math
        from ..mmx_vendor.hdr_curves._primaries import gamut_matrix as _gm
    return torch.tensor(_gm(GAMUTS[src], GAMUTS[dst]), dtype=torch.float32)


def gamut_convert(rgb: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    m = gamut_matrix(src, dst).to(rgb.device, torch.float32)
    return torch.einsum("...c,dc->...d", rgb[..., :3].float(), m)


# ------------------------------------------------------------------------------------------------------ bias table
# Mean luma bias of the H3 VAE round trip, diffhdr_log_16384 column of the owner's H3_HDR_Range_Comparison.xlsx
# (fp32 VAE, synthetic pattern, 1 frame), at each band's geometric centre (luma, stops; + = VAE output brighter).
DEFAULT_BIAS_TABLE = """# luma (linear, 1.0 = diffuse white): stops the H3 VAE adds (negative = it darkens)
0.0032: -0.877
0.032: -0.406
0.32: -0.143
3.2: -0.086
32: -0.177
316: -0.145
1430: -0.397"""


def parse_bias_table(text: str) -> list:
    """'luma: stops' lines (# comments allowed) -> sorted [(luma, stops)]. Errors name the bad line."""
    pts = []
    for n, line in enumerate(str(text or "").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            a, b = line.replace(",", ":").split(":", 1)
            luma, stops = float(a), float(b)
        except ValueError:
            raise ValueError(f"bias table line {n} '{line}': write it as 'luma: stops', e.g. '3.2: -0.09'") from None
        if not (luma > 0 and math.isfinite(luma) and math.isfinite(stops)):
            raise ValueError(f"bias table line {n}: luma must be > 0 and both numbers finite")
        pts.append((luma, stops))
    if not pts:
        raise ValueError("the bias table is empty: give at least one 'luma: stops' line")
    return sorted(pts)


def bias_at(luma: torch.Tensor, pts: list) -> torch.Tensor:
    """Stops of bias at each luma: linear interpolation in log2(luma), held flat outside the table."""
    xs = torch.tensor([math.log2(p[0]) for p in pts], dtype=torch.float32, device=luma.device)
    ys = torch.tensor([p[1] for p in pts], dtype=torch.float32, device=luma.device)
    if len(pts) == 1:
        return torch.full_like(luma, float(ys[0]))
    lx = torch.log2(luma.clamp_min(1e-8)).clamp(float(xs[0]), float(xs[-1]))
    i = torch.searchsorted(xs, lx.contiguous()).clamp(1, len(pts) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    w = ((lx - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return y0 + (y1 - y0) * w


def bias_correct(rgb: torch.Tensor, pts: list, strength: float = 1.0) -> torch.Tensor:
    """Luminance-only correction: every pixel is multiplied by 2^(-bias(luma) * strength), so colour ratios are kept.
    The bias is looked up at the DECODED luma (the true luma is unknown); the table is smooth, so the error is small."""
    lin = rgb[..., :3].float()
    gain = torch.pow(2.0, -bias_at(luma709(lin).clamp_min(0.0), pts) * float(strength))
    return lin * gain[..., None]


# -------------------------------------------------------------------------------------------- tonemap / clip masks
def tonemap_preview(lin: torch.Tensor, exposure_ev: float = 0.0, shoulder: float = 0.5) -> torch.Tensor:
    """Display-only sRGB preview: exposure, then a soft shoulder that rolls highlights off above the knee
    (knee = 1 - shoulder; shoulder 0 = hard clip at white)."""
    x = lin[..., :3].float().clamp_min(0.0) * (2.0 ** float(exposure_ev))
    knee = max(1e-3, 1.0 - float(shoulder))
    if knee < 1.0:
        span = 1.0 - knee
        x = torch.where(x > knee, knee + span * (1.0 - torch.exp(-(x - knee) / span)), x)
    return linear_to_srgb(x)


def clip_masks(sdr: torch.Tensor, over: float = 0.98, under: float = 0.02):
    """(over, under) MASKs from an SDR picture (0..1 code values): over = any channel at or above `over`; under =
    luma (of the code values) at or below `under`."""
    s = sdr[..., :3].float()
    return (s.amax(dim=-1) >= float(over)).float(), (luma709(s) <= float(under)).float()


# ------------------------------------------------------------------------------------------------- gain-map join
def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Box mean over a (2r+1)^2 window; edges average only the pixels inside. x: [N, C, H, W]."""
    k = 2 * int(r) + 1
    return torch.nn.functional.avg_pool2d(x, k, 1, int(r), count_include_pad=False)


def guided_upsample(gain_lo: torch.Tensor, guide_hi: torch.Tensor, radius: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """Fast guided-filter upsampling (He & Sun 2015): fit gain = a * guide + b in windows at LOW resolution against the
    downscaled guide, upsample a and b, apply them to the full-resolution guide. gain_lo [N, C, h, w], guide_hi
    [N, 1, H, W] -> [N, C, H, W]. A constant gain comes back exactly constant."""
    H, W = guide_hi.shape[-2:]
    h, w = gain_lo.shape[-2:]
    g_lo = torch.nn.functional.interpolate(guide_hi, size=(h, w), mode="area")
    mI, mp = _box(g_lo, radius), _box(gain_lo, radius)
    cov = _box(g_lo * gain_lo, radius) - mI * mp
    var = _box(g_lo * g_lo, radius) - mI * mI
    a = cov / (var + float(eps))
    b = mp - a * mI
    a = torch.nn.functional.interpolate(_box(a, radius), size=(H, W), mode="bilinear", align_corners=False)
    b = torch.nn.functional.interpolate(_box(b, radius), size=(H, W), mode="bilinear", align_corners=False)
    return a * guide_hi + b


def gain_map_join(draft_lin: torch.Tensor, regen_srgb: torch.Tensor, eps: float = 1e-3, radius: int = 4,
                  guide_eps: float = 1e-3):
    """Draft-resolution linear HDR + the Regenerate 4K output (sRGB) -> 4K linear HDR.
    regen_lin = sRGB decode of the 4K frames; gain = hdr / max(downscale(regen_lin), eps) per channel; the gain is
    upsampled with a guided filter on the 4K luma (in log2, so edges in shadows and highlights weigh alike); out =
    regen_lin * gain. Both IMAGE [B, H, W, 3]; returns (hdr_4k, report). Nothing is clamped to 1."""
    if draft_lin.shape[0] != regen_srgb.shape[0]:
        raise ValueError(f"frame counts differ: draft HDR {draft_lin.shape[0]}, Regenerate output "
                         f"{regen_srgb.shape[0]} - trim them to the same frames first")
    hdr = draft_lin[..., :3].float().permute(0, 3, 1, 2)
    regen_lin = srgb_to_linear(regen_srgb[..., :3].float()).permute(0, 3, 1, 2)
    h, w = hdr.shape[-2:]
    down = torch.nn.functional.interpolate(regen_lin, size=(h, w), mode="area")
    gain_lo = hdr / down.clamp_min(float(eps))
    guide = torch.log2(luma709(regen_lin.permute(0, 2, 3, 1)).clamp_min(1e-4))[:, None]
    gain_hi = guided_upsample(gain_lo, guide, radius, guide_eps).clamp_min(0.0)
    out = (regen_lin * gain_hi).permute(0, 2, 3, 1).contiguous()
    rep = (f"gain-map join: draft {w}x{h} HDR + Regenerate {regen_lin.shape[-1]}x{regen_lin.shape[-2]} -> linear HDR; "
           f"gain range {float(gain_lo.min()):.3g}..{float(gain_lo.max()):.3g}; output max "
           f"{float(out.max()):.4g}x white (not clamped)")
    return out, rep
