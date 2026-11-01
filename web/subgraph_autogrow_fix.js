// subgraph_autogrow_fix.js — Convert to Subgraph must not re-wire nodes that have autogrow input groups.
//
// Upstream ComfyUI frontend fault, reproduced on 1.53.10, 1.55.14 and 1.57.0 with core nodes only
// (--disable-all-custom-nodes; core MiniMax H3 Reference to Video). The node's copy inside the new subgraph gets its
// inputs in schema order, with the grown slots (ref_image_1, ref_video_1 ...) moved to the end, while every link keeps
// the slot INDEX it had. Each link then lands on whatever input sits at that index now. Measured: a reference video on
// an AUDIO input, width's link on length, length's on ref_image_size. The server refuses the run, or worse, runs it
// with the values swapped.
//
// The fix: just before the conversion, note each node's input names by index. Afterwards, put every link back on the
// input with the name it had. Only links are moved, never values; nodes whose order did not change are not touched.
// One patch per prototype, so another pack shipping the same fix does not wrap it twice. Ledger L9.96.

import { app } from "./shared.js";

const FLAG = "__c2cAutogrowSubgraphFix";

function linkOf(graph, id) {
    return graph?.links?.get ? graph.links.get(id) : graph?.links?.[id];
}

const slotIndex = (node, name) => (node.inputs || []).findIndex((s) => s.name === name);

function inputLinkOf(graph, node, i) {
    // 1.53+: connectivity is derived from a store; node.getInputLink is the public read
    return node.getInputLink ? node.getInputLink(i) : linkOf(graph, node.inputs[i]?.link);
}

// oldNames: the node's input names, by index, before the conversion. Returns how many links were moved.
// Through the public API only (disconnectInput / connect): assigning slot.link is ignored by newer frontends, and an
// autogrow group can grow or shrink while we work, so every slot is found by NAME at the moment it is used.
export function relinkByName(graph, node, oldNames) {
    const now = (node?.inputs || []).map((s) => s.name);
    if (!oldNames || !now.length || now.every((n, i) => n === oldNames[i])) return 0;
    const moves = [];
    now.forEach((name, i) => {
        const l = inputLinkOf(graph, node, i);
        if (!l) return;
        const want = oldNames[l.target_slot ?? i];
        if (want == null || want === name || slotIndex(node, want) < 0) return;
        moves.push({ originId: l.origin_id, originSlot: l.origin_slot, wrong: name, want });
    });
    for (const m of moves) {
        const i = slotIndex(node, m.wrong);
        if (i >= 0) node.disconnectInput(i);
    }
    let done = 0;
    for (const m of moves) {
        const j = slotIndex(node, m.want);
        if (j < 0) continue;
        const inputNode = graph.inputNode;
        if (inputNode && m.originId === inputNode.id) {
            const sin = inputNode.slots?.[m.originSlot] ?? graph.inputs?.[m.originSlot];
            sin?.connect(node.inputs[j], node);
        } else {
            graph.getNodeById(m.originId)?.connect(m.originSlot, node, j);
        }
        done += inputLinkOf(graph, node, slotIndex(node, m.want)) ? 1 : 0;
    }
    if (moves.length) trimGrownSlots(graph, node);
    return done;
}

// A link that landed on the wrong autogrow slot made that group grow; after the repair each group keeps what an
// untouched node has - its connected slots and ONE empty slot after the last of them (at least its first slot).
// Empty surplus slots only, removed through the public API, highest index first.
export function trimGrownSlots(graph, node) {
    const groups = new Map();
    (node.inputs || []).forEach((s, i) => {
        const m = /^(.+\.[^.]*_)(\d+)$/.exec(s.name || "");
        if (!m) return;
        const g = groups.get(m[1]) ?? [];
        g.push({ i, n: Number(m[2]), linked: !!inputLinkOf(graph, node, i) });
        groups.set(m[1], g);
    });
    const drop = [];
    for (const slots of groups.values()) {
        const first = Math.min(...slots.map((s) => s.n));
        const last = Math.max(first - 1, ...slots.filter((s) => s.linked).map((s) => s.n));
        for (const s of slots) if (!s.linked && s.n > last + 1) drop.push(s.i);
    }
    for (const i of drop.sort((a, b) => b - a)) node.removeInput(i);
    return drop.length;
}

function install() {
    const g = app.rootGraph ?? app.graph;
    let proto = g && Object.getPrototypeOf(g);
    while (proto && !Object.prototype.hasOwnProperty.call(proto, "convertToSubgraph")) proto = Object.getPrototypeOf(proto);
    if (!proto || proto[FLAG]) return;
    const orig = proto.convertToSubgraph;
    proto.convertToSubgraph = function (items, ...rest) {
        const before = new Map();
        try {
            for (const it of items || []) if (it && Array.isArray(it.inputs)) before.set(String(it.id), it.inputs.map((s) => s.name));
        } catch { /* an unusual selection: convert without the repair */ }
        const res = orig.call(this, items, ...rest);
        try {
            const sg = res?.subgraph ?? res?.node?.subgraph;
            let moved = 0;
            for (const n of sg?.nodes ?? sg?._nodes ?? []) moved += relinkByName(sg, n, before.get(String(n.id)));
            if (moved) {
                console.info(`[MiniMax H3] Convert to Subgraph: ${moved} link(s) put back on the inputs they were on (autogrow slot order).`);
                app.canvas?.setDirty?.(true, true);
            }
        } catch (e) {
            console.warn("[MiniMax H3] subgraph link repair skipped:", e);
        }
        return res;
    };
    Object.defineProperty(proto, FLAG, { value: true });
}

app.registerExtension({
    name: "MiniMaxH3.SubgraphAutogrowFix",
    setup() {
        install();
    },
});
