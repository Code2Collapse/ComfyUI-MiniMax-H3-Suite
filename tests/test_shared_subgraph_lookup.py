"""web/shared.js findNodeByExecId / forEachNode - the kit's status strip follows nodes INSIDE subgraphs (C2C L2.47).

The server names a node inside a subgraph "outer:inner[:deeper]"; Number() of that is NaN and app.graph is the root
only. The two functions are lifted out of the shipped file and run in node against a mock graph. CPU-only."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SHARED = Path(__file__).resolve().parents[1] / "web" / "shared.js"


def _fn(src: str, name: str) -> str:
    m = re.search(rf"^export function {name}\(.*?^}}", src, re.S | re.M)
    assert m, f"{name} not found in web/shared.js"
    return m.group(0).replace("export ", "", 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_exec_ids_resolve_inside_nested_subgraphs_and_walk_reaches_them():
    # INVARIANT: "12:5" and "12:6:9" resolve; a plain root id resolves; a wrong path is null; forEachNode sees all.
    src = SHARED.read_text(encoding="utf-8")
    script = r"""
const deep = { _nodes: [{ id: "9", type: "Deep" }] };
const inner = { _nodes: [{ id: "5", type: "Inner" }, { id: "6", type: "Holder", subgraph: deep }] };
const app = { graph: { _nodes: [{ id: "1", type: "A" }, { id: "12", type: "SG", subgraph: inner }] } };
""" + _fn(src, "findNodeByExecId") + "\n" + _fn(src, "forEachNode") + r"""
const seen = []; forEachNode((n) => seen.push(n.type));
process.stdout.write(JSON.stringify({
  inner: findNodeByExecId("12:5")?.type, deep: findNodeByExecId("12:6:9")?.type, root: findNodeByExecId(1)?.type,
  bad: findNodeByExecId("12:77"), seen: seen.sort() }));
"""
    with tempfile.TemporaryDirectory() as d:
        js = Path(d) / "probe.js"
        js.write_text(script, encoding="utf-8")
        p = subprocess.run([shutil.which("node"), str(js)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:800]
    out = json.loads(p.stdout)
    assert out == {"inner": "Inner", "deep": "Deep", "root": "A", "bad": None,
                   "seen": sorted(["A", "SG", "Inner", "Holder", "Deep"])}
