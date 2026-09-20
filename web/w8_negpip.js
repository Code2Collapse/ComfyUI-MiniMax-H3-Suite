/**
 * W8 — H3 NegPiP term ledger.
 *
 * WHY: this node's behaviour is one number per term that the user never sees.
 * What actually reaches the model is  V x (-strength x weight)  on that term's
 * token rows, and that product is spread across two places: a weight typed at
 * the end of a line in a text box, and a global strength slider somewhere else
 * on the node. Reading "1.5" on line 3 and "2.0" on the slider and arriving at
 * "-3.0" is arithmetic the node should be doing, not the person.
 *
 * So the panel shows, per line, the multiplier that line will produce, drawn
 * against the range where the method behaves. Past about -2 the flipped values
 * start to dominate the attention output and the frame degrades - that reads
 * as burn-in rather than as a bad prompt, so it is easy to misdiagnose, and it
 * is marked.
 *
 * Nothing here estimates TOKEN counts. How many rows "motion blur" encodes to
 * is the tokeniser's business and is not known until the node runs; a bar
 * width that confident would be a guess. The real counts land in the node's
 * report output.
 *
 * Also hides prompt_prefix unless context_mode needs it.
 *
 * Plain ES module, no Vue, rAF-throttled, chained onRemoved.
 */
import {
  addDomWidgetLast,
  app,
  chainOnRemoved,
  disposeState,
  rafThrottle,
  setupDpiCanvas,
  themeVar,
  widgetByName,
} from "./shared.js";

const NODE_ID = "MiniMaxH3_NegPiP";
const STATE = "_mmxNegPiP";
const ROW_H = 17;
const HEAD_H = 26;
const MIN_H = 62;
const MAX_ROWS = 9;
// Mirrors MAX_WEIGHT in mmx_utils/negpip.py. The python side refuses past it.
const MAX_WEIGHT = 10.0;
// Where the method stops behaving; not a hard limit, a drawn one.
const SAFE = 2.0;

/**
 * Same grammar as parse_terms() in mmx_utils/negpip.py: one phrase per line,
 * "# " comments and blanks dropped, an optional trailing ":number" weight.
 * Kept deliberately small and mirrored rather than shared - if the two ever
 * disagree the python side is authoritative and will say so on execute.
 */
function parseTerms(text) {
  const out = [];
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = /^(.*?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$/.exec(line);
    if (m && m[1].trim()) out.push({ phrase: m[1].trim(), weight: parseFloat(m[2]) });
    else out.push({ phrase: line, weight: 1.0 });
  }
  return out;
}

function problemWith(term) {
  if (term.weight < 0) return "negative weight — refused; the weight is how hard to push away";
  if (term.weight > MAX_WEIGHT) return `over the ${MAX_WEIGHT} ceiling — refused`;
  return null;
}

function build(node) {
  disposeState(node, STATE);

  const wrap = document.createElement("div");
  wrap.style.cssText = "width:100%;box-sizing:border-box;padding:2px 2px 0;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;display:block;border-radius:4px;";
  const caption = document.createElement("div");
  caption.style.cssText =
    "font:10px system-ui,sans-serif;opacity:.75;padding:3px 2px 0;line-height:1.35;";
  wrap.append(canvas, caption);

  const st = { canvas, caption, rows: 0, capH: 30 };
  node[STATE] = st;

  const paint = () => {
    const termsW = widgetByName(node, "terms");
    const strengthW = widgetByName(node, "strength");
    const modeW = widgetByName(node, "context_mode");
    const prefixW = widgetByName(node, "prompt_prefix");

    // prompt_prefix is dead weight in standalone mode; hiding it is the whole
    // difference between "which of these two do I fill in" and a clear node.
    if (prefixW) {
      const needed = String(modeW?.value ?? "standalone") === "prompt_prefix";
      if (prefixW.hidden !== !needed) {
        prefixW.hidden = !needed;
        if (prefixW.element) prefixW.element.hidden = !needed;
        node.setDirtyCanvas(true, true);
      }
    }

    const terms = parseTerms(termsW?.value);
    const strength = Number(strengthW?.value ?? 1);
    const shown = terms.slice(0, MAX_ROWS);
    const h = Math.max(MIN_H, HEAD_H + shown.length * ROW_H + 6);
    if (st.rows !== shown.length) {
      st.rows = shown.length;
      node.setDirtyCanvas(true, true);
    }

    const cssW = Math.max(120, (node.size?.[0] || 320) - 24);
    const ctx = setupDpiCanvas(canvas, cssW, h);
    const ink = themeVar("inputText") || "#ddd";
    const w = cssW;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(0, 0, w, h);

    // axis: 0 on the left, -MAX on the right. Everything here is negative, so
    // a longer bar is a harder push and the direction never has to be read.
    const padL = 6;
    const padR = 44;
    const axisW = Math.max(20, w - padL - padR);
    const span = Math.max(SAFE * 1.5, Math.abs(strength) * Math.max(
      1, ...terms.map((t) => Math.abs(t.weight) || 1)));
    const xOf = (mag) => padL + (Math.min(mag, span) / span) * axisW;

    ctx.font = "9px system-ui,sans-serif";
    ctx.textBaseline = "middle";

    // the safe/degrading boundary, drawn once. Not drawn with no terms:
    // an empty red zone floating next to a placeholder reads as damage,
    // not as a scale.
    if (shown.length && SAFE < span) {
      const sx = xOf(SAFE);
      ctx.fillStyle = "rgba(209,106,106,0.13)";
      ctx.fillRect(sx, HEAD_H - 4, padL + axisW - sx, h - HEAD_H);
      ctx.strokeStyle = "rgba(209,106,106,0.5)";
      ctx.beginPath();
      ctx.moveTo(sx, HEAD_H - 4);
      ctx.lineTo(sx, h - 3);
      ctx.stroke();
      ctx.fillStyle = "rgba(209,106,106,0.8)";
      ctx.fillText("degrades", sx + 3, HEAD_H - 11);
    }

    ctx.fillStyle = ink;
    ctx.globalAlpha = 0.55;
    ctx.fillText(
      terms.length
        ? `V x  0${" ".repeat(2)}→  -${span.toFixed(1)}`
        : "one phrase per line — 'blurry' or 'blurry : 1.5'",
      padL, HEAD_H - 11);
    ctx.globalAlpha = 1;

    shown.forEach((term, i) => {
      const y = HEAD_H + i * ROW_H + ROW_H / 2 - 2;
      const bad = problemWith(term);
      const mag = Math.abs(strength) * Math.abs(term.weight);
      const x = xOf(mag);

      ctx.fillStyle = bad ? "#d16a6a" : (mag > SAFE ? "#d1a33a" : "#5aa8c8");
      ctx.fillRect(padL, y - 4, Math.max(1, x - padL), 8);

      ctx.fillStyle = ink;
      ctx.globalAlpha = bad ? 0.55 : 0.92;
      // the reason is long; the phrase must survive the truncation, so the
      // reason is dropped rather than eating it.
      const room = w - padR - padL - 10;
      const full = bad ? `${term.phrase} — ${bad}` : term.phrase;
      const label = (bad && ctx.measureText(full).width > room)
        ? `${term.phrase} — refused` : full;

      // The label normally sits just after the bar. A long bar is exactly the
      // row you most need to read - the one past the degrade line - and that
      // is where the least space is left, so once the label no longer fits
      // after the bar it moves INSIDE it and flips to dark ink for contrast.
      const lw = ctx.measureText(label).width;
      const after = x + 4;
      const inside = after + lw > w - padR - 4;
      const tx = inside ? padL + 6 : after;
      if (inside) ctx.fillStyle = "#14171a";
      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, y - 8, w - padL - padR, 16);
      ctx.clip();
      ctx.fillText(label, tx, y);
      ctx.restore();

      ctx.globalAlpha = 1;
      ctx.fillStyle = bad ? "#d16a6a" : ink;
      ctx.textAlign = "right";
      ctx.fillText(bad ? "—" : `${(-mag).toFixed(2)}`, w - 5, y);
      ctx.textAlign = "left";
    });

    const hidden = terms.length - shown.length;
    const bad = terms.filter(problemWith).length;
    const parts = [];
    if (!terms.length) {
      parts.push("No terms — the node passes the model and conditioning straight through.");
    } else {
      parts.push(
        `${terms.length} term${terms.length === 1 ? "" : "s"} appended to the prompt; ` +
        "their attention values are negated, their keys are not.");
      if (hidden > 0) parts.push(`${hidden} more not drawn.`);
      if (bad) parts.push(`${bad} line${bad === 1 ? "" : "s"} will be refused on run.`);
      if (Math.abs(strength) === 0) {
        parts.push("strength 0 — the terms are in the sequence but contribute nothing.");
      }
      parts.push("Rides in the same forward pass as the prompt, so unlike CFG it costs no extra step time.");
    }
    caption.style.color = bad ? "#e06c6c" : "";
    caption.textContent = parts.join(" ");

    // The caption wraps, and how far depends on the node's width and on how
    // many terms there are. A fixed allowance clipped the last line off the
    // bottom of the node, so measure it and re-lay-out when it changes.
    const capH = Math.max(16, caption.offsetHeight || 0) + 6;
    if (Math.abs(capH - st.capH) > 1) {
      st.capH = capH;
      node.setDirtyCanvas(true, true);
      node.setSize?.(node.computeSize?.() || node.size);
    }
  };

  st.paint = rafThrottle(paint);
  addDomWidgetLast(node, "mmx_negpip_ledger", wrap,
    () => Math.max(MIN_H, HEAD_H + Math.min(st.rows, MAX_ROWS) * ROW_H + 6) + st.capH);

  for (const name of ["terms", "strength", "context_mode"]) {
    const wdg = widgetByName(node, name);
    if (!wdg) continue;
    const prev = wdg.callback;
    wdg.callback = function (...args) {
      const r = prev?.apply(this, args);
      st.paint();
      return r;
    };
  }
  // the terms box is a textarea: typing does not fire the widget callback
  const termsW = widgetByName(node, "terms");
  if (termsW?.element) {
    const onInput = () => st.paint();
    termsW.element.addEventListener("input", onInput);
    st.detach = () => termsW.element.removeEventListener("input", onInput);
  }

  chainOnRemoved(node, () => {
    st.detach?.();
    disposeState(node, STATE);
  });
  paint();
}

app.registerExtension({
  name: "MiniMaxSuite.NegPiPLedger",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try {
        build(this);
      } catch (_e) {
        /* a broken ledger must never take the node down */
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure?.apply(this, args);
      this[STATE]?.paint?.();
      return r;
    };
  },
});
