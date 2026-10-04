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
 * Plain ES module, no Vue, chained onRemoved.
 */
import {
  app,
  chainOnRemoved,
  disposeState,
  widgetByName,
} from "./shared.js";
import {
  mountPanel,
  statusLine,
} from "./c2c_ui/index.js";

const NODE_ID = "MiniMaxH3_NegPiP";
const STATE = "_mmxNegPiP";
const MIN_H = 100;
const NODE_MIN_W = 380;
const MAX_ROWS = 9;
const MAX_WEIGHT = 10.0;
const SAFE = 2.0;

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

function rebuild(st, node) {
  const termsW = widgetByName(node, "terms");
  const strengthW = widgetByName(node, "strength");
  const modeW = widgetByName(node, "context_mode");
  const prefixW = widgetByName(node, "prompt_prefix");

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
  const span = Math.max(SAFE * 1.5, Math.abs(strength) * Math.max(
    1, ...terms.map((t) => Math.abs(t.weight) || 1)));

  st.axisHint.textContent = terms.length
    ? `V × 0 → −${span.toFixed(1)}`
    : "one phrase per line — 'blurry' or 'blurry : 1.5'";

  const degradePct = shown.length && SAFE < span
    ? `${((1 - SAFE / span) * 100).toFixed(1)}%`
    : "0%";
  st.tbody.innerHTML = "";
  st.table.style.display = shown.length ? "" : "none";

  shown.forEach((term) => {
    const bad = problemWith(term);
    const mag = Math.abs(strength) * Math.abs(term.weight);
    const pct = Math.min(100, (mag / span) * 100);
    const tr = document.createElement("tr");

    const tdPhrase = document.createElement("td");
    tdPhrase.className = "c2c-ui-ledger__phrase";
    tdPhrase.textContent = bad ? `${term.phrase} — ${bad}` : term.phrase;
    if (bad) tdPhrase.style.opacity = "0.65";

    const tdBar = document.createElement("td");
    tdBar.className = "c2c-ui-ledger__bar";
    if (shown.length && SAFE < span) {
      tdBar.classList.add("c2c-ui-ledger__bar--degrade");
      tdBar.style.setProperty("--ledger-degrade-pct", degradePct);
    }
    const fill = document.createElement("div");
    fill.className = "c2c-ui-ledger__fill";
    fill.style.width = `${pct}%`;
    fill.style.background = bad ? "var(--cu-danger, #f27a92)"
      : (mag > SAFE ? "var(--cu-warn, #f3d288)" : "var(--cu-accent, #b494ff)");
    tdBar.appendChild(fill);

    const tdVal = document.createElement("td");
    tdVal.className = "c2c-ui-ledger__val";
    tdVal.textContent = bad ? "—" : `${(-mag).toFixed(2)}`;
    if (bad) tdVal.style.color = "var(--cu-danger, #f27a92)";

    tr.append(tdPhrase, tdBar, tdVal);
    st.tbody.appendChild(tr);
  });

  if (st.rows !== shown.length) {
    st.rows = shown.length;
    node.setDirtyCanvas(true, true);
  }

  const hidden = terms.length - shown.length;
  const badCount = terms.filter(problemWith).length;
  const parts = [];
  if (!terms.length) {
    parts.push("No terms — the node passes the model and conditioning straight through.");
  } else {
    parts.push(
      `${terms.length} term${terms.length === 1 ? "" : "s"} appended to the prompt; ` +
      "their attention values are negated, their keys are not.");
    if (hidden > 0) parts.push(`${hidden} more not drawn.`);
    if (badCount) parts.push(`${badCount} line${badCount === 1 ? "" : "s"} will be refused on run.`);
    if (Math.abs(strength) === 0) {
      parts.push("strength 0 — the terms are in the sequence but contribute nothing.");
    }
    parts.push("Rides in the same forward pass as the prompt, so unlike CFG it costs no extra step time.");
  }
  st.caption.setText(parts.join(" "), badCount ? "danger" : "default");
}

function build(node) {
  disposeState(node, STATE);

  const root = document.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "2px";
  root.style.width = "100%";

  const axisHint = document.createElement("div");
  axisHint.className = "c2c-ui-status";
  axisHint.style.cssText = "font-variant-numeric:tabular-nums;padding:0 2px;margin:0;";

  const table = document.createElement("table");
  table.className = "c2c-ui-ledger";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const h of ["term", "magnitude", "value"]) {
    const th = document.createElement("th");
    th.textContent = h;
    if (h === "magnitude") th.style.width = "38%";
    if (h === "value") th.style.width = "52px";
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  table.append(thead, tbody);

  const caption = statusLine();
  root.append(axisHint, table, caption.el);

  const st = { root, axisHint, table, tbody, caption, rows: 0 };
  node[STATE] = st;

  const panelWidget = mountPanel(node, "mmx_negpip_ledger", root, { minHeight: MIN_H });
  panelWidget.computeSize = (w) => {
    const rowH = 22;
    const headH = st.rows > 0 ? 18 : 0;
    const tableH = headH + Math.min(st.rows, MAX_ROWS) * rowH;
    const capH = 22;
    return [w, 18 + tableH + capH + 4];
  };

  const repaint = () => rebuild(st, node);

  for (const name of ["terms", "strength", "context_mode"]) {
    const wdg = widgetByName(node, name);
    if (!wdg) continue;
    const prev = wdg.callback;
    wdg.callback = function (...args) {
      const r = prev?.apply(this, args);
      repaint();
      return r;
    };
  }
  const termsW = widgetByName(node, "terms");
  if (termsW?.element) {
    const onInput = () => repaint();
    termsW.element.addEventListener("input", onInput);
    st.detach = () => termsW.element.removeEventListener("input", onInput);
  }

  chainOnRemoved(node, () => {
    st.detach?.();
    disposeState(node, STATE);
  });
  repaint();
}

app.registerExtension({
  name: "MiniMaxSuite.NegPiPLedger",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (String(nodeData?.name || "") !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;
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
      const st = this[STATE];
      if (st) rebuild(st, this);
      return r;
    };
  },
});
