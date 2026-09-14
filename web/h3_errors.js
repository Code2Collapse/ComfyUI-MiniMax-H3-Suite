// h3_errors.js — plain-English translation of H3 runtime failures.
//
// Deliberately free of any ComfyUI import so plain node can unit-test it.
// kit.js imports it; nothing else should restate these rules.
//
// These are not generic torch messages. They are the failures THIS pack
// actually produces, each one met while building it, and each message names
// the specific thing to change rather than the thing that broke.

// First match wins, so specific patterns sit above generic ones.
const RULES = [
  // ── H3 support missing from the ComfyUI build ────────────────────────────
  [/No module named ['"]?comfy\.ldm\.minimax/i,
   "This ComfyUI build does not ship MiniMax H3 support (comfy.ldm.minimax is " +
   "absent). Update ComfyUI to a build that includes H3 - the node itself is " +
   "installed correctly."],
  [/comfy_kitchen.*has no attribute|int8_attention_is_available/i,
   "ComfyUI's comfy_kitchen module is a different version than this build " +
   "expects, so the H3 model code will not import. Update ComfyUI and " +
   "comfy_kitchen together."],
  [/requires a native MiniMax H3 diffusion model/i,
   "That MODEL input is not a MiniMax H3 model. This node only patches H3 - " +
   "check which loader the wire comes from."],

  // ── the frame grid: the most common silent mistake ───────────────────────
  [/frame count|n % ?17|must be .*17|not on H3'?s grid/i,
   "The frame count is not on H3's grid. A legal length is n % 17 == 5 " +
   "(5, 22, 39, 56, ... 90 ...). Use the Context Windows node, which snaps it " +
   "for you and says what it changed."],
  [/SplitSigmas|sigma schedule stops at/i,
   "This sigma schedule is one half of a split schedule - it does not end at " +
   "zero. H3 needs a complete schedule; remove the SplitSigmas node."],

  // ── AV pairing ───────────────────────────────────────────────────────────
  [/audio.*(length|frames|samples).*(mismatch|must match)|video.*audio.*match/i,
   "The audio and video lengths disagree. H3 samples them jointly, so they " +
   "must describe the same duration - check the frame count against the " +
   "sample rate."],
  [/NestedTensor|joint (av|audio-video)/i,
   "The joint audio/video tensor is malformed. Both streams must be present " +
   "and the same duration for an AV pass."],

  // ── resources ────────────────────────────────────────────────────────────
  [/CUDA out of memory|CUDA error: out of memory/i,
   "Out of GPU memory. Shorten the window (Context Windows), lower the " +
   "resolution, or enable the block cache."],
  [/DefaultCPUAllocator|not enough memory|MemoryError/i,
   "Out of system RAM. Reduce the frame count per pass - long clips should be " +
   "tiled with Context Windows rather than generated in one go."],

  // ── dependencies ─────────────────────────────────────────────────────────
  [/ModuleNotFoundError: No module named ['"]([\w.]+)/i,
   (m) => `A required Python package is missing: ${m[1]}. Install it and ` +
          "restart ComfyUI."],

  // ── ordinary wiring ──────────────────────────────────────────────────────
  [/NoneType.*has no attribute|NoneType.*not subscriptable/i,
   "A required input is not connected."],
  [/Sizes of tensors must match|size mismatch|must have the same (dimensions|shape)/i,
   "Two inputs are different sizes. Match them before this node - the mask and " +
   "the frames must share a resolution and a frame count."],
  [/all masks are empty|mask is empty|nothing to crop/i,
   "The mask is empty, so there is nothing to work on. Check the node that " +
   "produces it."],
  [/Interrupted|execution was interrupted|KeyboardInterrupt/i,
   "Cancelled."],
];

export function humaniseH3Error(raw) {
  if (!raw) return "Something went wrong.";
  const text = String(raw);
  for (const [re, msg] of RULES) {
    const m = text.match(re);
    if (m) return typeof msg === "function" ? msg(m) : msg;
  }
  // Fall back to the exception line, which is LAST. The first line is always
  // "Traceback (most recent call last):" and says nothing.
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    if (/^[A-Za-z_.]*(Error|Exception)\b/.test(lines[i])) return lines[i];
  }
  return lines[lines.length - 1] || "Something went wrong.";
}
