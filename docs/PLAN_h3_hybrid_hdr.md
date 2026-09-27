# H3 hybrid: Singularity's HDR without the FL2VA damage

## The user's report

H3 colours are not rich or HDR. Mixing Ref2VA with Singularity looked like the
fix, but "FL2VA and Ref2VA mixture is damaging". It has to hold for every H3
build: full, pruned, fp8, int8 and GGUF.

## What the models are (verified, 2026-09-27)

| Model | What it is | Source |
|---|---|---|
| Ref2VA | Official, reference-conditioned (identity from up to 9 images, 3 videos, 3 voices) | `Comfy-Org/MiniMax-H3`: bf16 66.3 GB, int8_convrot 34.0 GB, pruned bf16 40.2 GB, pruned fp8 21.0 GB, pruned int8 21.0 GB |
| FL2VA | Official, first/last-frame + text | same repo, same variants |
| hybrid bNN-49 | FL2VA base with blocks NN-49 taken from Ref2VA; "weight-selection merge", validated by "empirical, subjective comparison" | `smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models` |
| Singularity v1.3 | Community merge of "ref, fl, b25-49" + HDR fine-tune + 3 days of pruning; merge ratios undisclosed; Ref2VA format | `WarmBloodAban/Minimax-h3_Singularity` (int8 34.0 GB, pruned int8 21.0 GB, pruned w4a8 11.8 GB), GGUF at `Abiray/MiniMax-H3-Singularity-GGUF` |
| GGUF bases | Ref2VA / FL2VA pruned, Q2-Q8 | `Abiray/MiniMax-H3-Pruned-GGUF`, `unsloth/MiniMax-H3-GGUF` |

Layout (read from the safetensors header, no download): 50 blocks; per block
`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` (int8 + per-row
`weight_scale` + `comfy_quant`), `adaln_proj.linear`, norms. Full hidden 5376.
**Pruned narrows the layers** (out_proj in 7168, fc2 in 14336), so a
difference is only defined within one variant: pruned with pruned, full with full.
`int8_convrot` weights are stored after a grouped rotation (group 256); they
must be decoded with ComfyUI's own `comfy_kitchen` layouts, never by hand.

So Singularity's trunk is largely FL2VA (the b25-49 hybrid it folded in is an
FL2VA base). That is consistent with the user's finding.

### Measured on the real weights (2026-09-27)

Single layers fetched from Hugging Face by HTTP range (pruned variant:
`Comfy-Org` ref2va/fl2va `pruned_bf16` and `ref2va_pruned_int8_convrot`,
Singularity `ref2va_Pruned_v1.3_int8`), decoded by this pack's own reader.

Decoder check: the official int8_convrot layer decodes to within 1.0-1.2 % of
the official bf16 layer at every depth tested - ordinary int8 error (an undone
rotation would read ~140 %).

`attn.out_proj`, FL share of Singularity's change:

| block | a | cos(S−R, F−R) | ‖H‖/‖R‖ after removing FL | int8 floor |
|---|---|---|---|---|
| 0 | 1.00 | 0.86 | 1.42 % | 1.20 % |
| 12 | 1.00 | 0.94 | 1.24 % | 1.04 % |
| 25 | 1.00 | 0.94 | 1.13 % | 0.98 % |
| 37 | 1.00 | 0.91 | 1.13 % | 1.00 % |
| 49 | 1.00 | 0.75 | 1.20 % | 1.12 % |

`adaln_proj.linear` (the per-block modality and reference routing, bf16, exact):

| blocks | Singularity equals | ‖S − parent‖/‖parent‖ |
|---|---|---|
| 0-24 | **FL2VA** | 0.17 % (bf16 rounding) |
| 25-49 | **Ref2VA** | 0.17 % |

FL2VA and Ref2VA routing differ by 185 % of their norm - they are different
routings, not a small change.

**Conclusion.** Singularity v1.3 = the b25-49 hybrid (FL2VA trunk, FL2VA
routing in blocks 0-24, Ref2VA routing in 25-49) + an HDR fine-tune on the
trunk. For Ref2VA work, half the network routes references with FL2VA's
routing: that is the damage. The fine-tune itself is Singularity − FL2VA on the
trunk, about 0.4-0.8 % of the weight norm above the int8 noise floor. The
hybrid applies exactly that to Ref2VA and leaves every routing weight Ref2VA's.

## Method

For each weight tensor W, with R = Ref2VA, F = FL2VA, S = Singularity (same variant):

    D = F − R            the FL2VA direction
    Δ = S − R            everything Singularity changed
    a = ⟨Δ, D⟩ / ⟨D, D⟩  how much of Δ is the FL2VA mixture (clamped to [0, 1])
    H = Δ − a·D          the rest: HDR fine-tune, pruning repair, b25-49 remainder

`H` is compressed per layer into a rank-r LoRA (randomised SVD), with the
captured energy reported. Applied to any Ref2VA build of the same variant -
including GGUF, since LoRA patches apply on top of GGUF weights - it gives
Ref2VA's reference fidelity with Singularity's HDR training.

The per-block `a` and `‖H‖` report is itself the evidence: it shows where the FL
mixture sits and how large the HDR part is, instead of guessing.

## Deliverables (MiniMaxSuite)

1. `tools/h3_hybrid_extract.py` - offline, streams tensor by tensor (peak RAM
   about three copies of the largest layer, ~2 GB fp32), reads safetensors
   (bf16 / fp16 / fp8_scaled / int8 incl. convrot / w4a8) and GGUF, writes the
   LoRA + a JSON/Markdown report. `--no-fl-projection` gives the plain
   Singularity − Ref2VA delta for comparison.
2. Node `MiniMaxH3_HybridHDR` - applies that LoRA to a Ref2VA MODEL: strength,
   block range, and an optional late-steps ramp (HDR where texture forms, Ref2VA
   alone while composition and identity settle).
3. Node `MiniMaxH3_ColorQC` - scores a clip: colourfulness (Hasler-Süsstrunk),
   dynamic range and highlight headroom in linear light, texture energy,
   temporal flicker, and Oklab distance to a reference image's palette. For A/B
   of base vs Singularity vs hybrid by number.
4. A workflow running the three side by side, and a README section.

## Limits, stated plainly

- Nothing here can run the 33B model on the dev box; the tools are tested on
  small synthetic checkpoints with the real key layout, and the A/B runs on the
  user's GPU box.
- The projection removes the part of Singularity that points along FL2VA − Ref2VA.
  If the damage came from something not on that direction (e.g. the pruning
  repair), the report will show a small `a` and the LoRA will still carry it;
  the strength and block-range controls exist for that.
